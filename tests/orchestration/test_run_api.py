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
"""``memrank.run`` -- the whole eval, in a script, with nothing saved.

The decoupling promise this package refactor exists for: get the adapter, run the
eval, get a typed result -- no run directory, no status file, no config, no terminal
output. What the CLI adds on top (registry, heartbeat, sync, cloud) is wiring, and a
library caller who asked for none of it gets none of it.
"""
from __future__ import annotations

import pytest

# `memrank.run` is now the typed run over the seven (memrank/instrument/); the cell run
# that produces an `EvalResult` is imported from its own module, which is where the CLI
# and the cloud reach it too.
import memrank
from memrank.evaluation.api import run as run_cell
from memrank.judging.judge import JudgeConfig
from tests.fakes import FakeAdapter, FakeBenchmark, JudgeFakeBenchmark, make_fake_completer

# These tests exercise the deprecated in-process route on purpose; its notice is expected here.
pytestmark = pytest.mark.filterwarnings(
    "ignore:.*is deprecated and will be removed:DeprecationWarning")


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """Fresh, EMPTY state dirs -- the test asserts they stay that way."""
    monkeypatch.setenv("MEMRANK_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("MEMRANK_CONFIG_DIR", str(tmp_path / "cfg"))
    return tmp_path


def test_a_catalog_ref_runs_end_to_end_and_writes_nothing(isolated, capsys):
    result = run_cell("word-overlap", "demo", repeats=1)

    assert isinstance(result, memrank.EvalResult)
    assert result.composite == 0.8, "the same score the CLI's demo sweep records"
    assert result.target == "word-overlap"
    assert not (isolated / "runs").exists(), "no run registry entry may appear"
    assert not (isolated / "cfg").exists(), "no config may be created"
    out = capsys.readouterr()
    assert out.out == "" and out.err == "", "a library call narrates only via an observer"


def test_instances_run_without_any_resolution(isolated):
    adapter = FakeAdapter(name="fake", responses={"q1": [], "q2": []})

    result = run_cell(adapter, FakeBenchmark(), repeats=1)

    assert isinstance(result, memrank.EvalResult)
    assert result.target is None, "no ref was given, so none is invented"
    assert result.repeats == 1


def test_the_judge_default_asks_the_benchmark(isolated):
    """`demo` scores itself, so judge=None means unjudged -- same rule as the CLI's."""
    result = run_cell("word-overlap", "demo", repeats=1)

    assert result.judged_metrics is None


def test_an_explicit_judge_config_is_used_as_given(isolated):
    bench = JudgeFakeBenchmark()
    adapter = FakeAdapter("fake", {q["text"]: [] for q in bench.load()[0].queries})
    cfg = JudgeConfig(no_context_control=False, completer=make_fake_completer())

    result = run_cell(adapter, bench, repeats=1, judge=cfg)

    assert result.judged_metrics is not None
    assert result.judged_metrics["n_judged"] == 2


def test_judge_false_forces_an_unjudged_run(isolated):
    bench = JudgeFakeBenchmark()
    adapter = FakeAdapter("fake", {q["text"]: [] for q in bench.load()[0].queries})

    result = run_cell(adapter, bench, repeats=1, judge=False)

    assert result.judged_metrics is None


def test_an_observer_hears_the_run(isolated):
    events: list[str] = []

    class Collector(memrank.EvalObserver):
        def planned(self, plan):
            events.append(f"planned:{plan.units}u/{plan.documents}d/{plan.retrievals}r")

        def unit_started(self, *, index, total, documents):
            events.append(f"unit_started:{index}/{total}")

        def item_done(self, stage, *, seconds, done=0, total=0, **kwargs):
            events.append(f"{stage}:{done}/{total}")

        def unit_finished(self, *, label, index, total, queries_done, queries_total):
            events.append(f"unit_finished:{queries_done}/{queries_total}")

    adapter = FakeAdapter(name="fake", responses={"q1": [], "q2": []})
    run_cell(adapter, FakeBenchmark(), repeats=1, observer=Collector())

    assert events == ["planned:1u/1d/2r", "unit_started:1/1", "ingest:1/1",
                      "retrieve:1/2", "retrieve:2/2", "unit_finished:2/2"]


def test_the_lazy_exports_resolve_and_dir_lists_them():
    """`run` is deliberately absent: ATO-2247 moved the verb onto the evaluation.

    It was never one of these lazy exports anyway -- it was a plain import that `dir()` picked
    up beside them -- and it left the package surface outright rather than becoming an alias,
    so neither the attribute nor the `dir()` entry survives.
    """
    assert memrank.Evaluation.run.__module__ == "memrank.instrument.evaluation"
    assert memrank.EvalResult.__module__ == "memrank.evaluation.result"
    assert {"EvalResult", "EvalObserver", "EvalPlan", "JudgeConfig"} <= set(dir(memrank))
    assert "run" not in dir(memrank)
    with pytest.raises(AttributeError):
        _ = memrank.no_such_symbol
