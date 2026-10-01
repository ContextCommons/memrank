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
"""LongMemEval ships its ground truth inside its session ids. We must not pass them on.

Every evidence session id in `longmemeval_s_cleaned.json` begins with the literal prefix
`answer`, and the official retrieval scorer keys on exactly that
(src/retrieval/run_retrieval.py:272, `[d for d in corpus_ids if "answer" in d]`). The prefix is
a NOISE-FREE oracle: the count of `answer`-prefixed sessions equals the gold count at every
cardinality, so ranking on it alone scores recall_all@5 = 0.994 and recall_all@10 = 1.000 --
above the paper's best published retriever (0.862).

memrank forwarded those ids to every engine twice: as `Document.id`, which every adapter passes
through as metadata (docs/system-contract.md:71 makes it contractual), and again inside
`Document.context`, which supermemory maps to `entity_context` and hindsight to `item["context"]`
-- semantically indexed fields.

That made it a VALIDITY defect rather than a comparability one, and an engine-dependent one:
how much a given engine benefited turned on whether it indexed ids, so it corrupted the ORDERING
between engines, not merely the level. These tests are the guard.
"""
import pytest

from memrank.benchmarks.longmemeval import LongMemEvalBenchmark
from tests.benchmarks.test_longmemeval_methodology import _item, _one_per_type, _write_fixture


@pytest.fixture()
def loaded(tmp_path, monkeypatch):
    _write_fixture(tmp_path, monkeypatch, _one_per_type())
    return LongMemEvalBenchmark().load()


def test_no_engine_visible_field_carries_a_real_session_id(loaded, tmp_path, monkeypatch):
    """The precise requirement: the dataset's own session ids never reach the engine.

    Scoped to `id` and `context` rather than `content`, because conversation text may contain
    the word "answer" legitimately -- the leak is the identifier, not the vocabulary.
    """
    real_ids = {sid for unit in loaded
                for sid in unit.metadata["session_handles"].values()}
    assert real_ids, "fixture must exercise real session ids"

    for unit in loaded:
        for doc in unit.documents:
            for field in (doc.id, doc.context or ""):
                assert not any(sid in field for sid in real_ids), \
                    f"{field!r} carries a dataset session id"


def test_handles_are_positional_and_stable(loaded):
    for unit in loaded:
        expected = [f"{unit.unit_id}_s{i:03d}" for i in range(len(unit.documents))]
        assert [d.id for d in unit.documents] == expected


def test_context_leads_with_a_position_not_an_identifier(loaded):
    """`context` reaches semantically indexed fields on two adapters, so it must carry no
    identifier. It leads with the position; the session DATE it also carries is a legitimate
    input the protocol hands the reader (see test_longmemeval_reader_context.py)."""
    for unit in loaded:
        for i, doc in enumerate(unit.documents):
            assert doc.context.startswith(f"Session {i + 1}")


def test_handles_round_trip_to_the_real_session_ids(loaded):
    """M7 scores recall against `answer_session_ids`, so the map must be complete and injective."""
    for unit in loaded:
        handles = unit.metadata["session_handles"]
        assert set(handles) == {d.id for d in unit.documents}
        assert len(set(handles.values())) == len(handles)


def test_gold_ids_are_handles_not_raw_ids(loaded):
    for unit in loaded:
        gold = unit.queries[0]["gold_ids"]
        assert gold, "fixture marks session 0 as evidence"
        assert set(gold) <= {d.id for d in unit.documents}


def test_the_prefix_cheat_is_no_longer_expressible(tmp_path, monkeypatch):
    """The audit's reproduction: rank documents by `id.startswith("answer")`.

    On the pre-M2 loader this retrieved every gold document. It must now select nothing.
    """
    _write_fixture(tmp_path, monkeypatch, [_item("q1", "multi-session", n_sessions=6)])
    unit = LongMemEvalBenchmark().load()[0]

    cheat = [d.id for d in unit.documents if "answer" in d.id.lower()]

    assert cheat == []
    assert unit.queries[0]["gold_ids"], "there is still gold to find, just not by filename"
