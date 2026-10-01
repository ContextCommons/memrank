# Copyright 2026 AtomicStrata
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.
"""A run that dies in the judge stage keeps the half that was expensive.

The artifact is written once, after judging, so a failure there discarded every ingest and retrieve
that paid for it: 4h50m on 2026-08-12, with 272 ingests and 4,620 retrieves complete and nothing on
disk but a progress heartbeat. `run_cell` now writes `retrieval.json` before the first judge call.

THE PROPERTY, and the only one that matters: judging from the checkpoint produces what an
uninterrupted run produced. Everything else here defends that.

Documents are stored once by content hash -- 126x duplication measured on a real artifact -- with the
engine's own id kept beside the hash, because 14 ids in that artifact carried more than one content
and collapsing identity into storage would have hidden it.
"""

from __future__ import annotations

import json

import pytest

from memrank import runner
from memrank.core import Document
from memrank.judging.judge import JudgeConfig
from memrank.runs import checkpoint
from tests.fakes import FakeAdapter, JudgeFakeBenchmark, make_fake_completer

DOC = Document(id="d1", content="favorite drink is green tea", user_id="u1")


def _run(tmp_path, *, judge_workers=1, checkpoint_path=None):
    """One judged cell against fakes -- no network, no engine."""
    bench = JudgeFakeBenchmark()
    adapter = FakeAdapter("fake", {q["text"]: [DOC] for q in bench.load()[0].queries})
    cfg = JudgeConfig(no_context_control=False, completer=make_fake_completer())
    return runner.run_cell(
        adapter, bench, k=5, repeats=1, run_id_prefix="t", model="gpt-4o-mini",
        token_budget=5000, judge=cfg, judge_workers=judge_workers,
        checkpoint_path=checkpoint_path).to_dict()


# ------------------------------------------------------------------ #
# The property
# ------------------------------------------------------------------ #

def test_judging_from_the_checkpoint_reproduces_the_run(tmp_path):
    path = tmp_path / checkpoint.CHECKPOINT_FILE
    uninterrupted = _run(tmp_path, checkpoint_path=path)

    cell, budget_mode = checkpoint.read(path)
    bench = JudgeFakeBenchmark()
    resumed = runner._apply_judge(
        bench.load(), cell["per_query"],
        JudgeConfig(no_context_control=False, completer=make_fake_completer()),
        make_fake_completer(), budget_mode, shape=bench.judge_shape())

    assert resumed == uninterrupted["judged_metrics"]


def test_the_checkpoint_carries_the_whole_cell_not_just_the_rows(tmp_path):
    """A resumed run must be able to produce the artifact the run WOULD have produced. Metrics
    alone would make it a second-class result nobody trusts."""
    path = tmp_path / checkpoint.CHECKPOINT_FILE
    _run(tmp_path, checkpoint_path=path)
    cell, _ = checkpoint.read(path)

    for field in ("benchmark", "adapter", "per_unit", "latency_metrics", "token_metrics",
                  "receipt", "k", "repeats", "corpus_documents"):
        assert field in cell, f"{field} missing -- rejudge could not rebuild the artifact"
    # Absent rather than null -- `_aggregate_cell` omits the key when there are no verdicts. Either
    # way the point holds: the checkpoint is written BEFORE judging and carries no grades.
    assert cell.get("judged_metrics") is None


def test_no_checkpoint_is_written_for_an_unjudged_run(tmp_path):
    """Nothing to resume: an unjudged run's artifact is complete when retrieval ends."""
    bench = JudgeFakeBenchmark()
    adapter = FakeAdapter("fake", {q["text"]: [DOC] for q in bench.load()[0].queries})
    path = tmp_path / checkpoint.CHECKPOINT_FILE
    runner.run_cell(adapter, bench, k=5, repeats=1, run_id_prefix="t", model="gpt-4o-mini",
                    token_budget=5000, judge=None, checkpoint_path=path).to_dict()
    assert not path.exists()


# ------------------------------------------------------------------ #
# Content addressing
# ------------------------------------------------------------------ #

def test_pack_stores_repeated_content_once():
    """The measured case: an engine that returns its whole bank per query serialises the same
    text once per query. On a real artifact this takes the rows from 12.89 MB to 2.09 MB."""
    rows = [{"query_id": f"q{i}", "retrieved": [{"id": "m1", "content": "same text", "score": 1.0}]}
            for i in range(50)]
    documents, packed = checkpoint.pack(rows)

    assert len(documents) == 1
    assert all("content" not in doc for row in packed for doc in row["retrieved"])
    # Per-entry fields an adapter set (score, and anything else) stay on the entry; only the
    # document moves to the table.
    assert all(doc["score"] == 1.0 for row in packed for doc in row["retrieved"])
    assert checkpoint.unpack(documents, packed) == rows


def test_the_reference_is_an_index_not_a_digest():
    """A sha256 hex is no smaller than the ~193-character document it replaces, which is why the
    first version of this only reached 1.8x on a real artifact. The references are the bulk."""
    documents, packed = checkpoint.pack(
        [{"query_id": "q1", "retrieved": [{"id": "m1", "content": "x" * 500}]}])
    reference = packed[0]["retrieved"][0]

    assert reference[checkpoint._REF] == 0
    assert len(json.dumps(reference)) < 20, "the reference should be a couple of characters"
    assert checkpoint.unpack(documents, packed)[0]["retrieved"][0]["content"] == "x" * 500


def test_one_id_carrying_two_contents_survives_the_round_trip():
    """Hindsight rewrote 14 memory ids mid-run while being queried. Identity and storage keys are
    different layers; collapsing them would render a consolidation as a disappearance."""
    rows = [
        {"query_id": "q0", "retrieved": [{"id": "m1", "content": "before consolidation"}]},
        {"query_id": "q9", "retrieved": [{"id": "m1", "content": "after consolidation"}]},
    ]
    documents, packed = checkpoint.pack(rows)

    # Two table entries under ONE id: keyed on (id, content), so the consolidation is preserved
    # rather than one version silently winning.
    assert len(documents) == 2, "two distinct contents must be stored distinctly"
    assert [d["id"] for d in documents] == ["m1", "m1"]
    assert {d["content"] for d in documents} == {"before consolidation", "after consolidation"}
    assert checkpoint.unpack(documents, packed) == rows


def test_missing_content_is_refused_not_defaulted():
    """Judging against silently-empty context yields a complete, plausible, wrong verdict."""
    with pytest.raises(checkpoint.CheckpointError, match="does not contain"):
        checkpoint.unpack({}, [{"query_id": "q1", "retrieved": [{"id": "m1", "hash": "deadbeef"}]}])


def test_a_checkpoint_from_another_schema_is_refused(tmp_path):
    path = tmp_path / checkpoint.CHECKPOINT_FILE
    path.write_text(json.dumps({"schema_version": 99, "cell": {}, "budget_mode": "matched",
                                "documents": {}}), encoding="utf-8")
    with pytest.raises(checkpoint.CheckpointError, match="schema"):
        checkpoint.read(path)


# ------------------------------------------------------------------ #
# What the checkpoint must not lose
# ------------------------------------------------------------------ #

def test_the_budget_mode_survives(tmp_path):
    """`budget_mode` is the one thing the judge needs that no receipt field records. Lost, a
    resumed uncapped arm is re-judged as matched and reports the cap instead of what it carried
    (the defect fixed in 9a1cfea, arriving again through the resume path)."""
    bench = JudgeFakeBenchmark()
    adapter = FakeAdapter("fake", {q["text"]: [DOC] for q in bench.load()[0].queries})
    adapter.context_budget = "uncapped"
    path = tmp_path / checkpoint.CHECKPOINT_FILE
    runner.run_cell(adapter, bench, k=5, repeats=1, run_id_prefix="t", model="gpt-4o-mini",
                    token_budget=5000,
                    judge=JudgeConfig(no_context_control=False, completer=make_fake_completer()),
                    checkpoint_path=path).to_dict()

    _, budget_mode = checkpoint.read(path)
    assert budget_mode == "uncapped"


def test_the_checkpoint_is_not_a_result_cell(tmp_path):
    """`cell_files` matches `<target>__<benchmark>.json` positively and so already excludes this --
    pinned because the blocklist version of that function swallowed `status.json` the day the
    heartbeat started writing one, and made every run list twice."""
    from memrank.runs import registry

    (tmp_path / checkpoint.CHECKPOINT_FILE).write_text("{}", encoding="utf-8")
    (tmp_path / "fake__judgefake.json").write_text("{}", encoding="utf-8")
    assert [p.name for p in registry.cell_files(tmp_path)] == ["fake__judgefake.json"]
