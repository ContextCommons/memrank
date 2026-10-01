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
"""Two results, laid side by side. Checked against a hand-built pair with counted flips.

The worked case is 50 tasks: 25 both right, 6 both wrong, 8 only A, 11 only B. That is a gap
of -6 points, 19 discordant pairs and an exact two-sided McNemar p of 0.648, which is computed
here by hand from the binomial rather than taken from the code being tested.
"""
from __future__ import annotations

import pytest

from memrank.instrument.measure import Decider, Value
from memrank.instrument.paired import TOO_FEW, PairingRefused, paired
from memrank.instrument.result import EvaluationRecord, Result, SystemRecord
from memrank.instrument.statistics import mcnemar_exact
from memrank.instrument.task import Task
from memrank.instrument.trace import Trace

WHEN = "2026-09-21T10:00:00Z"


def result(marks: dict[str, float], *, system="A", name="demo", version="1",
           measure="word-match", groups: dict[str, str] | None = None) -> Result:
    return Result(
        system=SystemRecord(name=system, kind="memory"),
        evaluation=EvaluationRecord(name=name, version=version, clearing="per-group"),
        traces=tuple(Trace(task=Task(id=task, prompt="?"),
                           group=(groups or {}).get(task), started=WHEN, finished=WHEN)
                     for task in marks),
        values=tuple(Value(measure=measure, decider=Decider.RULE, task_id=task, value=value)
                     for task, value in marks.items()),
        started=WHEN, finished=WHEN)


def worked_pair() -> tuple[Result, Result]:
    """50 tasks: 25 both right, 6 both wrong, 8 only A, 11 only B."""
    a: dict[str, float] = {}
    b: dict[str, float] = {}
    for kind, count, left, right in (("both", 25, 1.0, 1.0), ("neither", 6, 0.0, 0.0),
                                     ("onlya", 8, 1.0, 0.0), ("onlyb", 11, 0.0, 1.0)):
        for i in range(count):
            a[f"{kind}{i}"] = left
            b[f"{kind}{i}"] = right
    return result(a, system="A"), result(b, system="B")


def test_the_worked_pair_reports_the_counts_the_gap_and_the_exact_mcnemar_p():
    reading = paired(*worked_pair())

    [word_match] = reading.measures
    assert (word_match.both, word_match.neither, word_match.only_a, word_match.only_b) == (
        25, 6, 8, 11)
    assert word_match.tasks == 50 and word_match.discordant == 19
    assert word_match.mean_a == pytest.approx(0.66) and word_match.mean_b == pytest.approx(0.72)
    assert word_match.gap == pytest.approx(0.06)
    assert word_match.p_value == pytest.approx(0.648, abs=0.001)
    assert len(word_match.flips) == 19
    assert word_match.caution is None


def test_the_exact_test_is_the_binomial_tail_and_not_an_approximation():
    assert mcnemar_exact(8, 11) == pytest.approx(2 * sum(
        __import__("math").comb(19, i) for i in range(9)) / 2 ** 19)
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(0, 4) == pytest.approx(0.125)


def test_a_split_too_small_to_characterise_is_said_rather_than_refused():
    a = result({"t1": 1.0, "t2": 0.0, "t3": 1.0}, system="A")
    b = result({"t1": 0.0, "t2": 0.0, "t3": 1.0}, system="B")

    [reading] = paired(a, b).measures

    assert reading.discordant == 1 and reading.caution == TOO_FEW
    assert reading.gap == pytest.approx(-1 / 3)


def test_a_continuous_measure_gets_a_paired_bootstrap_interval_clustered_by_group():
    tasks = {f"t{i}": 0.4 + (i % 5) / 10 for i in range(20)}
    groups = {task: f"g{i % 4}" for i, task in enumerate(tasks)}
    a = result(tasks, system="A", measure="latency-ish", groups=groups)
    b = result({task: value + 0.05 + (i % 3) / 100 for i, (task, value) in
                enumerate(tasks.items())}, system="B", measure="latency-ish", groups=groups)

    [reading] = paired(a, b, resamples=200, seed=7).measures

    assert reading.kind == "continuous"
    assert reading.gap == pytest.approx(0.06, abs=0.005)
    assert reading.ci_low < reading.gap < reading.ci_high, (
        "an interval the resampled means actually straddle")
    assert (reading.resamples, reading.seed) == (200, 7)
    assert reading.p_value is None, "no McNemar over a measure that is not binary"


def test_the_bootstrap_is_deterministic_for_the_same_seed():
    a = result({f"t{i}": i / 30 for i in range(30)}, system="A", measure="score")
    b = result({f"t{i}": (i + 3) / 30 for i in range(30)}, system="B", measure="score")

    first = paired(a, b, resamples=200, seed=11).measures[0]
    again = paired(a, b, resamples=200, seed=11).measures[0]

    assert (first.ci_low, first.ci_high) == (again.ci_low, again.ci_high)


def test_pairing_refuses_two_runs_of_different_evaluations():
    with pytest.raises(PairingRefused, match="different evaluations"):
        paired(result({"t1": 1.0}, name="demo"), result({"t1": 1.0}, name="locomo"))


def test_pairing_refuses_two_runs_of_the_same_evaluation_at_different_versions():
    with pytest.raises(PairingRefused, match="different versions"):
        paired(result({"t1": 1.0}, version="1"), result({"t1": 1.0}, version="2"))


def test_pairing_refuses_a_refused_run():
    refused = result({}, system="B").model_copy(update={"refusal": "no retrieve verb"})

    with pytest.raises(PairingRefused, match="refused"):
        paired(result({"t1": 1.0}), refused)


def test_tasks_only_one_run_has_are_named_rather_than_paired():
    reading = paired(result({"t1": 1.0, "t2": 0.0}), result({"t2": 1.0, "t3": 1.0}))

    assert reading.only_in_a == ("t1",) and reading.only_in_b == ("t3",)
    assert reading.measures[0].tasks == 1, "only the shared task is paired"


def test_the_reading_never_says_better():
    reading = paired(*worked_pair())

    assert "better" not in reading.model_dump_json()
    assert set(reading.measures[0].model_dump()) >= {"gap", "flips", "p_value", "caution"}
