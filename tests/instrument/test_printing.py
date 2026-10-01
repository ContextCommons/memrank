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
"""`print(result)` and `print(paired)`, held to the lines a reader relies on.

Printing moved into the package because it was living in `examples/shared/show.py`, which meant
the reader of an example had to be shown a formatter before being shown a number. What that
formatter printed is the specification here: the measure's name on every value, who decided it,
the caution when a gap cannot be characterised, and the closing line that refuses to name a
winner. Every object is hand-built, so nothing below runs a system or reads a clock.
"""
from __future__ import annotations

from memrank.instrument.measure import Decider, Value
from memrank.instrument.paired import NOT_BETTER, TOO_FEW, paired
from memrank.instrument.result import EvaluationRecord, Result, SystemRecord
from memrank.instrument.task import Task
from memrank.instrument.trace import Failure, Trace

WHEN = "2026-09-21T10:00:00Z"


def _result(marks: dict[str, float], *, system: str = "A", values: tuple[Value, ...] = (),
            broken: str | None = None) -> Result:
    return Result(
        system=SystemRecord(name=system, kind="memory", version="1"),
        evaluation=EvaluationRecord(name="demo", version="v1", clearing="per-group",
                                    task_count=len(marks)),
        traces=tuple(Trace(task=Task(id=task, prompt="?"), started=WHEN, finished=WHEN,
                           error=Failure(step="retrieve", kind="RuntimeError", message="no")
                           if task == broken else None)
                     for task in marks),
        values=values or tuple(
            Value(measure="word-match", decider=Decider.RULE, task_id=task, value=value,
                  why="a retrieval proxy, not answer correctness")
            for task, value in marks.items()),
        started=WHEN, finished=WHEN)


def test_a_result_names_its_system_its_evaluation_and_how_many_traces_it_holds():
    printed = str(_result({"q1": 1.0, "q2": 0.0}, broken="q2"))
    assert "system:     A (memory), version 1" in printed
    assert "evaluation: demo at v1, 2 task(s), cleared per-group" in printed
    assert "traces:     2 recorded, 1 with errors" in printed


def test_every_printed_value_carries_its_measure_its_decider_and_what_it_is_about():
    printed = str(_result({"q1": 1.0}))
    assert "word-match" in printed
    assert "1.000" in printed
    assert "decided by rule" in printed
    assert "[q1]" in printed
    assert "a retrieval proxy, not answer correctness" in printed


def test_a_run_scope_value_says_whole_run_where_a_task_id_would_go():
    value = Value(measure="failure-rate", decider=Decider.MEMRANK, value=0.0)
    printed = str(_result({"q1": 1.0}, values=(value,)))
    assert "[whole run]" in printed
    assert "decided by memrank" in printed


def test_a_refusal_prints_its_reason_and_nothing_it_did_not_measure():
    refused = Result(
        system=SystemRecord(name="A", kind="memory"),
        evaluation=EvaluationRecord(name="demo", version="v1", clearing="per-group"),
        refusal="the system has no verb to be told things with",
        started=WHEN, finished=WHEN)
    printed = str(refused)
    assert "REFUSED before the system was touched: the system has no verb" in printed
    assert "traces:" not in printed


def test_the_fields_are_still_there_for_a_debugger():
    """`__str__` is the read; `__repr__` stays pydantic's, so nothing was traded away."""
    result = _result({"q1": 1.0})
    assert "schema_version=" in repr(result)
    assert repr(result) != str(result)


def _pair() -> tuple[Result, Result]:
    marks = {f"q{i}": float(i % 2) for i in range(10)}
    flipped = {task: 1.0 - value for task, value in marks.items()}
    return _result(marks, system="A"), _result(flipped, system="B")


def test_a_paired_reading_prints_the_measure_the_means_the_gap_and_the_counts():
    printed = str(paired(*_pair()))
    assert "demo at v1" in printed
    assert "A = A    B = B" in printed
    assert "word-match (binary, 10 paired task(s))" in printed
    assert "mean A 0.500   mean B 0.500   gap +0.000   10 discordant" in printed
    assert "both 0  neither 0  only A 5  only B 5  McNemar exact p =" in printed


def test_a_paired_reading_names_the_tasks_that_flipped():
    printed = str(paired(*_pair()))
    assert "flipped: q0  0.0 -> 1.0" in printed


def test_a_thin_pairing_prints_the_caution_rather_than_characterising_the_gap():
    a, b = _result({"q1": 1.0, "q2": 1.0}), _result({"q1": 1.0, "q2": 0.0}, system="B")
    assert f"caution: {TOO_FEW}" in str(paired(a, b))


def test_a_paired_reading_ends_by_refusing_to_say_which_system_is_better():
    printed = str(paired(*_pair()))
    assert printed.endswith(NOT_BETTER)
    assert "better" not in printed[:printed.index(NOT_BETTER)]
