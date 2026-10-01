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
"""The observable contract of a run, pinned before the runner split.

``runner.py`` is scheduled to be split into an eval library and a thin CLI (AGENTS.md).
These tests freeze what that split must not change: the artifact's exact key sequence, the
``status.json`` write sequence, which stream carries what, and the fact that a library call
touches nothing on disk. Each pins what IS, down to the byte where bytes are deterministic --
a failure here during the split means behavior drifted, not that this file needs loosening.

The sweep golden uses ``word-overlap,no-context x demo`` because it is the one full local
run that needs no engine, no credentials, and no network beyond the (cached) tokenizer --
the same shape ``test_local_sweep`` builds on.
"""
from __future__ import annotations

import json
import re
import signal

import pytest
from typer.testing import CliRunner

from memrank.judging.judge import JudgeConfig
from memrank.runner import app, run_cell
from memrank.runs import registry
from tests.fakes import FakeAdapter, FakeBenchmark, JudgeFakeBenchmark, make_fake_completer

runner = CliRunner()


@pytest.fixture(autouse=True)
def _restore_sigterm():
    """Execute mode installs a real SIGTERM handler; tests must not leak it into each other."""
    original = signal.getsignal(signal.SIGTERM)
    yield
    signal.signal(signal.SIGTERM, original)


# ------------------------------------------------------------------ #
# The artifact schema
# ------------------------------------------------------------------ #

#: Every key ``run_cell`` produces, in construction order. The first block is the closed
#: schema every cell carries; the tail comes from the two open spreads -- ``benchmark.rollup``
#: (empty for the fake) and ``benchmark.config_for_receipt()``. ``judged_metrics`` is
#: appended only when a judge ran, and the orchestrator injects ``target`` after the fact.
CELL_KEYS = (
    "adapter", "benchmark", "composite", "per_unit", "per_query", "ingested_documents",
    "latency_metrics", "retrieve_latency_summary", "ingest_latency_summary",
    "ingest_throughput", "token_metrics", "corpus_documents", "corpus_bytes",
    "corpus_tokens", "workers", "latency_contended", "judge_workers",
    "context_tokens_mean", "est_dollars_per_query", "est_dollars_per_cell", "cost_basis",
    "cleanup_is_destructive", "substring_recall_supported", "composite_rankable",
    "quality_metric", "question_text_public", "receipt", "n_units", "k", "repeats",
    "unit_outcomes", "units_total", "units_failed", "unit_failure_rate",
    # FakeBenchmark's config_for_receipt() -- the open tail, pinned so a reordering of the
    # spreads (they must come LAST, later keys shadowing earlier ones) cannot pass unseen.
    "tier", "slice", "task_version",
)


def _cell(**kwargs):
    a = FakeAdapter(name="fake", responses={"q1": [], "q2": []})
    return run_cell(a, FakeBenchmark(), k=10, repeats=1, run_id_prefix="g",
                    model="gpt-4o-mini", token_budget=5000, **kwargs).to_dict()


def test_run_cell_returns_the_typed_result_whose_dict_is_the_artifact():
    """`run_cell` answers with an `EvalResult`; `.to_dict()` is the artifact schema."""
    from memrank.evaluation.result import EvalResult

    a = FakeAdapter(name="fake", responses={"q1": [], "q2": []})
    result = run_cell(a, FakeBenchmark(), k=10, repeats=1, run_id_prefix="g",
                      model="gpt-4o-mini", token_budget=5000)

    assert isinstance(result, EvalResult)
    assert result.is_applicable
    assert result.to_dict()["composite"] == result.composite


def test_the_cell_carries_exactly_the_agreed_keys_in_order():
    """Ten modules read this dict (registry, record, leaderboard, API projection, the
    uploader); its key sequence IS the contract, including where the spreads land."""
    assert tuple(_cell()) == CELL_KEYS


def test_an_unjudged_cell_omits_judged_metrics_rather_than_nulling_it():
    """Absent and null mean different things to every reader: absent is "no judge ran",
    null would be "a judge ran and produced nothing"."""
    assert "judged_metrics" not in _cell()


def test_a_judged_cell_appends_judged_metrics_last():
    bench = JudgeFakeBenchmark()
    docs = {q["text"]: [] for q in bench.load()[0].queries}
    adapter = FakeAdapter("fake", docs)
    cfg = JudgeConfig(no_context_control=False, completer=make_fake_completer())

    result = run_cell(adapter, bench, k=5, repeats=1, run_id_prefix="g",
                      model="gpt-4o-mini", token_budget=5000, judge=cfg).to_dict()

    assert list(result)[-1] == "judged_metrics"


def test_the_cell_serializes_the_way_the_artifact_is_written():
    """`indent=2, sort_keys=True` is the writer's exact form; a cell that stops
    round-tripping through it would corrupt the artifact, not just a test."""
    result = _cell()
    assert json.loads(json.dumps(result, indent=2, sort_keys=True)) == result


def test_an_inapplicable_cell_is_the_five_key_shell():
    """A graph benchmark against a non-graph adapter skips with a reason -- the shell's
    shape is load-bearing for every reader that keys on `status`."""
    class GraphBenchmark(FakeBenchmark):
        requires_graph = True

    a = FakeAdapter(name="fake", responses={})
    result = run_cell(a, GraphBenchmark(), k=10, repeats=1, run_id_prefix="g",
                      model="gpt-4o-mini", token_budget=5000).to_dict()

    assert result == {
        "adapter": "fake", "benchmark": "fake", "status": "not_applicable",
        "composite": None,
        "reason": "fake requires a graph-capable adapter; fake is not",
    }


def test_a_library_call_leaves_the_registry_and_config_untouched(tmp_path, monkeypatch):
    """The decoupling promise, already true and pinned so it stays true: `run_cell`
    without a checkpoint path writes no run dir, no status, no settings -- nothing."""
    monkeypatch.setenv("MEMRANK_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("MEMRANK_CONFIG_DIR", str(tmp_path / "cfg"))

    _cell()

    assert not (tmp_path / "runs").exists()
    assert not (tmp_path / "cfg").exists()


# ------------------------------------------------------------------ #
# The sweep's heartbeat and streams
# ------------------------------------------------------------------ #

_SWEEP = ["submit", "word-overlap,no-context", "demo", "--on", "none",
          "--run-id", "r1", "--run-id", "r2"]


def _execute(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMRANK_RUNS_DIR", str(tmp_path / "runs"))
    return runner.invoke(app, [*_SWEEP, "--output-dir", str(tmp_path / "out")])


def test_the_status_writes_follow_the_lifecycle_in_order(tmp_path, monkeypatch):
    """Every status.json write, journaled at the one chokepoint all of them share.
    The deduplicated state sequence is each run's whole lifecycle, and run 2 must not
    start until run 1 is done -- the sweep is serial and its records say so."""
    from memrank.runs import status as status_mod

    journal: list[tuple[str, str]] = []
    real = status_mod.write_json

    def spy(path, data, **kwargs):
        journal.append((path.parent.name, data.get("state")))
        return real(path, data, **kwargs)

    monkeypatch.setattr(status_mod, "write_json", spy)
    result = _execute(tmp_path, monkeypatch)

    assert result.exit_code == 0, result.output
    lifecycle = ["initializing", "queued", "running", "ingesting", "retrieving",
                 "writing", "done"]
    for run_id in ("r1", "r2"):
        states = [s for rid, s in journal if rid == run_id]
        deduped = [s for i, s in enumerate(states) if i == 0 or s != states[i - 1]]
        assert deduped == lifecycle, f"{run_id}: {deduped}"
    first_r2_run = next(i for i, (rid, s) in enumerate(journal)
                        if rid == "r2" and s == "running")
    r1_done = next(i for i, (rid, s) in enumerate(journal)
                   if rid == "r1" and s == "done")
    assert r1_done < first_r2_run, "the sibling must stay queued until run 1 is done"


def _normalized(text: str, tmp_path) -> list[str]:
    """The stderr transcript with its only volatile parts made stable: tmp paths and the
    per-run isolation suffix (`word-overlap-71fe4e`) minted from a random hex."""
    text = text.replace(str(tmp_path / "out"), "<out>")
    text = text.replace(str(tmp_path / "runs"), "<runs>")
    return [re.sub(r"-[0-9a-f]{6} × ", " × ", line) for line in text.splitlines()]


def _unit_lines(target: str) -> list[str]:
    """demo through one target: 1 unit, 3 docs, 5 queries x 3 repeats, echoed per item
    (the throttle step is max(1, total//10) = 1 at this size)."""
    return [
        "ingesting unit 1/1 (3 docs) ...",
        *[f"[{target} × demo] ingested {i}/3 docs" for i in (1, 2, 3)],
        *[f"[{target} × demo] retrieved {q}/5 queries (pass {p}/3)"
          for p in (1, 2, 3) for q in (1, 2, 3, 4, 5)],
        f"[{target} × demo] unit 1/1 (5/5 queries)",
    ]


def test_stderr_narrates_the_run_and_stdout_stays_silent(tmp_path, monkeypatch):
    """The whole transcript, line for line -- scores included: word-overlap earns 0.800 on
    demo and the no-context control 0.200, so a scoring drift fails here too. Execute mode
    prints no ids: stdout is submit mode's data channel and stays empty here."""
    result = _execute(tmp_path, monkeypatch)

    assert result.exit_code == 0, result.output
    assert result.stdout == ""
    assert _normalized(result.stderr, tmp_path) == [
        "warning: `memrank submit` is deprecated and will be removed; evaluate an agent with "
        "`memrank run` instead (see the README).",
        "preparing tokenizer 'o200k_base' (first run downloads ~4 MB, cached afterwards) ...",
        "tokenizer ready",
        "run r1  (word-overlap × demo)",
        "run r2  (no-context × demo)",
        *_unit_lines("word-overlap"),
        "[word-overlap × demo] recall=0.800 -> <out>/word-overlap__demo.json",
        # No sync note in either run: the suite runs signed out (conftest), and
        # local-first means a signed-out run says nothing about syncing.
        "run recorded -> <runs>/r1",
        "note: 'no-context' retrieves nothing, so unjudged it scores only on negative "
        "queries (the ones a system should decline) and its number says nothing about "
        "memory. The no-memory arm tests the hypothesis only with --judge, where a reader "
        "answers from an empty context.",
        *_unit_lines("no-context"),
        "[no-context × demo] recall=0.200 -> <out>/no-context__demo.json",
        "run recorded -> <runs>/r2",
        "Summary written to <out>/summary__demo.json",
    ]


def test_the_recorded_artifact_is_the_cell_plus_target_in_writer_form(tmp_path, monkeypatch):
    """What lands on disk is the run_cell dict with the orchestrator's one addition --
    `target` -- serialized `indent=2, sort_keys=True`, byte for byte."""
    result = _execute(tmp_path, monkeypatch)

    assert result.exit_code == 0, result.output
    path = registry.runs_root() / "r1" / "word-overlap__demo.json"
    raw = path.read_text(encoding="utf-8")
    cell = json.loads(raw)
    assert raw == json.dumps(cell, indent=2, sort_keys=True)
    assert set(cell) == set(CELL_KEYS) | {"target"}
    assert cell["target"] == "word-overlap"
    assert cell["composite"] == 0.8
