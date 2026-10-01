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
"""A cell attempts every unit and records what happened to each.

The failure this pins: `run_cell` used to end at the first unit that raised, so a run that died
on unit 12 of 50 discarded the eleven that had worked and reported nothing about any of them.
The unit failure RATE is a measurement of the engine, so it is recorded rather than thrown.

Every failure here is injected deterministically by unit id -- no timing, no live backend.
"""

from __future__ import annotations

from functools import partial

import pytest

from memrank.adapters.word_overlap import WordOverlap
from memrank.evaluation.cell import RateLimitExhausted, run_cell
from memrank.evaluation.observer import EvalObserver
from tests.fakes import MultiUnitFakeBenchmark

#: Which unit the injected failure lands on. `u1` of `u0..u2`: a middle unit, so a run that
#: continues is visibly different from one that stops (u0 only) and one that never started.
DOOMED = "u1"


class _FailingAdapter(WordOverlap):
    """Word-overlap retrieval that raises for ONE unit, at a chosen stage.

    The unit is identified by ``prepare``'s isolation unit, which carries the unit's
    ``isolation_id`` as its suffix -- the only handle an adapter has on which unit it is in.
    """

    def __init__(self, *, stage: str = "ingest", unit: str | None = DOOMED,
                 error: type[BaseException] = RuntimeError) -> None:
        super().__init__()
        #: ``None`` dooms EVERY unit -- the all-fail case, which is not the same shape as one.
        self._stage, self._unit, self._error = stage, unit, error

    def _doomed(self) -> bool:
        return self._unit is None or (self._isolation or "").endswith(f"-{self._unit}")

    def _raise(self, stage: str) -> None:
        if stage == self._stage and self._doomed():
            raise self._error(f"{self.name} refused {stage} for {self._unit}")

    def ingest(self, documents):
        self._raise("ingest")
        return super().ingest(documents)

    def retrieve(self, query, k, user_id, query_timestamp=None):
        self._raise("retrieve")
        return super().retrieve(query, k, user_id, query_timestamp)


class _FailingScoreBenchmark(MultiUnitFakeBenchmark):
    """Scores every unit except one, which raises -- the third stage a unit can fail in."""

    def score(self, unit, responses):
        if unit.unit_id == DOOMED:
            raise ValueError(f"cannot score {unit.unit_id}")
        return super().score(unit, responses)


class _RecordingObserver(EvalObserver):
    """Collects the warnings the loop emits, which is how an operator hears about a lost unit."""

    def __init__(self) -> None:
        self.warnings: list[str] = []

    def warning(self, message: str) -> None:
        self.warnings.append(message)


def _run(factory, *, benchmark=None, workers=1, n_units=3, **kwargs):
    """One cell over ``n_units`` independent units, as a dict.

    ``factory`` builds the adapter, and is handed on as ``make_adapter``: the concurrent path
    gives each unit its OWN instance, and sharing one here would race on the store the fake
    keeps per isolation unit.
    """
    bench = benchmark if benchmark is not None else MultiUnitFakeBenchmark(n_units=n_units)
    return run_cell(
        factory(), bench, k=10, repeats=1, run_id_prefix="run-u", model="gpt-4o-mini",
        token_budget=5000, workers=workers, make_adapter=factory, **kwargs,
    ).to_dict()


def _outcomes(cell) -> dict[str, str]:
    return {row["unit_id"]: row["outcome"] for row in cell["unit_outcomes"]}


# ------------------------------------------------------------------ #
# One unit fails, the rest are measured
# ------------------------------------------------------------------ #

#: (stage the unit dies in, how the cell is built). One case per stage a unit can fail in, and
#: the concurrent path repeated for the stage that reaches it through a different merge.
FAILURE_CASES = (
    ("ingest", 1), ("retrieve", 1), ("ingest", 3), ("retrieve", 3),
)


@pytest.mark.parametrize(("stage", "workers"), FAILURE_CASES)
def test_a_failed_unit_is_recorded_and_the_rest_still_run(stage, workers):
    cell = _run(partial(_FailingAdapter, stage=stage), workers=workers)

    assert _outcomes(cell) == {"u0": "ok", DOOMED: "failed", "u2": "ok"}
    assert [row["unit_id"] for row in cell["per_unit"]] == ["u0", "u2"]
    assert {row["unit_id"] for row in cell["per_query"]} == {"u0", "u2"}


@pytest.mark.parametrize(("stage", "workers"), FAILURE_CASES)
def test_the_failed_unit_records_where_and_how_it_failed(stage, workers):
    cell = _run(partial(_FailingAdapter, stage=stage), workers=workers)

    failed = next(row for row in cell["unit_outcomes"] if row["outcome"] == "failed")
    assert failed["stage"] == stage
    assert failed["error"] == "RuntimeError"
    assert DOOMED in failed["message"]


def test_a_unit_whose_scorer_raises_fails_in_score():
    cell = _run(WordOverlap, benchmark=_FailingScoreBenchmark(n_units=3))

    failed = next(row for row in cell["unit_outcomes"] if row["outcome"] == "failed")
    assert (failed["stage"], failed["error"]) == ("score", "ValueError")


@pytest.mark.parametrize("workers", [1, 3])
def test_the_counts_say_what_the_composite_was_computed_over(workers):
    cell = _run(_FailingAdapter, workers=workers)

    assert (cell["units_total"], cell["units_failed"]) == (3, 1)
    assert cell["unit_failure_rate"] == pytest.approx(1 / 3)
    # n_units stays the number ATTEMPTED, which is what every existing reader means by it.
    assert cell["n_units"] == 3
    assert cell["composite"] == 1.0        # the two units that ran both scored 1.0


@pytest.mark.parametrize("workers", [1, 3])
def test_a_lost_unit_is_narrated_to_whoever_is_listening(workers):
    observer = _RecordingObserver()

    _run(_FailingAdapter, workers=workers, observer=observer)

    assert len(observer.warnings) == 1
    assert DOOMED in observer.warnings[0] and "1/3 units failed" in observer.warnings[0]


# ------------------------------------------------------------------ #
# Every unit fails
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("workers", [1, 3])
def test_a_cell_whose_every_unit_failed_completes_with_no_composite(workers):
    """None, never 0.0: an engine that measured nothing did not score zero."""
    cell = _run(partial(_FailingAdapter, unit=None), workers=workers)

    assert cell["composite"] is None
    assert set(_outcomes(cell).values()) == {"failed"}
    assert (cell["units_total"], cell["units_failed"]) == (3, 3)
    assert cell["unit_failure_rate"] == 1.0


# ------------------------------------------------------------------ #
# What stays fatal
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("workers", [1, 3])
def test_fail_fast_restores_the_original_ending(workers):
    """The exception the caller always saw, not a wrapper around it."""
    with pytest.raises(RuntimeError, match="refused ingest"):
        _run(_FailingAdapter, workers=workers, fail_fast=True)


@pytest.mark.parametrize("workers", [1, 3])
def test_an_exhausted_rate_limit_ends_the_run_whatever_the_flag_says(workers):
    """Account-wide, so every remaining unit would spend the same ten-minute deadline to
    arrive at the same nothing."""
    with pytest.raises(RateLimitExhausted):
        _run(partial(_FailingAdapter, error=RateLimitExhausted), workers=workers,
             fail_fast=False)


@pytest.mark.parametrize("workers", [1, 3])
def test_an_interrupt_is_not_a_unit_failure(workers):
    """`KeyboardInterrupt` is a person stopping the run; recording it as one unit's problem
    would swallow it once per remaining unit."""
    with pytest.raises(KeyboardInterrupt):
        _run(partial(_FailingAdapter, error=KeyboardInterrupt), workers=workers)


# ------------------------------------------------------------------ #
# What the artifact and the checkpoint carry
# ------------------------------------------------------------------ #

def test_a_clean_cell_records_every_unit_as_ok():
    cell = _run(WordOverlap)

    assert _outcomes(cell) == {"u0": "ok", "u1": "ok", "u2": "ok"}
    assert (cell["units_failed"], cell["unit_failure_rate"]) == (0, 0.0)
    # A unit that ran carries two keys; the failure fields would be three nulls claiming a stage.
    assert cell["unit_outcomes"][0] == {"unit_id": "u0", "outcome": "ok"}


def test_the_receipt_and_artifact_survive_a_json_round_trip(tmp_path):
    """The artifact is written and read as JSON by ten modules; nothing here may be unserialisable."""
    import json

    cell = _run(_FailingAdapter)
    path = tmp_path / "cell.json"
    path.write_text(json.dumps(cell, indent=2, sort_keys=True), encoding="utf-8")

    assert json.loads(path.read_text(encoding="utf-8"))["unit_outcomes"] == cell["unit_outcomes"]


class _JudgedMultiUnitBenchmark(MultiUnitFakeBenchmark):
    """Multi-unit and judgeable: every query carries a category and a gold answer."""

    def load(self):
        units = super().load()
        for unit in units:
            for query in unit.queries:
                query.update(category="single-hop", kind="positive",
                             gold_answers=[query["text"]])
        return units


def test_judging_skips_a_failed_unit_instead_of_asserting_on_its_missing_rows():
    """The judge indexes drill rows by query id and asserts every judgeable query has one; a
    failed unit has none, and must leave the judged denominator rather than trip that."""
    from memrank.judging.judge import JudgeConfig
    from tests.fakes import make_fake_completer

    cfg = JudgeConfig(no_context_control=False, completer=make_fake_completer())
    cell = _run(_FailingAdapter, benchmark=_JudgedMultiUnitBenchmark(n_units=3), judge=cfg)

    assert cell["units_failed"] == 1
    # Two units' worth of queries were graded, and the third is absent rather than unjudged.
    assert cell["judged_metrics"]["n_judged"] == 2
    assert cell["judged_metrics"]["n_unjudged"] == 0


def test_the_checkpoint_round_trips_a_cell_with_failed_units(tmp_path):
    """`memrank ops rejudge` reads this back; the new fields must survive pack/unpack."""
    from memrank.runs import checkpoint

    cell = _run(_FailingAdapter)
    path = tmp_path / checkpoint.CHECKPOINT_FILE
    checkpoint.write(path, cell=cell, budget_mode="matched")

    restored, budget_mode = checkpoint.read(path)
    assert budget_mode == "matched"
    assert restored["unit_outcomes"] == cell["unit_outcomes"]
    assert (restored["units_total"], restored["units_failed"]) == (3, 1)
