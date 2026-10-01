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
"""Every one of the seven constructs, and says what it is without a run.

The point of typing them is that a person reading the class knows what they get. These are the
cases that would break silently if a field were renamed, retyped or given a default that lies:
a declaration that declared nothing must read as `None` and never as zero, a trace must carry
its task, and a value must never be a bare number.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

import memrank
from memrank.contract import Document, Recall
from memrank.instrument.evaluation import UNFREEZABLE, Clearing, Evaluation
from memrank.instrument.measure import Decider, Measure, Scope, Value
from memrank.instrument.result import EvaluationRecord, Result, SystemRecord
from memrank.instrument.system import Assistant, Model, Retriever, System, kind_of, missing_verbs
from memrank.instrument.task import Expected, Task
from memrank.instrument.trace import Answered, Declared, Failure, Trace

WHEN = "2026-09-21T10:00:00Z"


def a_trace(**over) -> Trace:
    fields = {"task": Task(id="t1", prompt="what?"), "started": WHEN, "finished": WHEN}
    return Trace(**{**fields, **over})


def test_a_task_is_frozen_data_and_carries_what_a_correct_outcome_looks_like():
    task = Task(id="t1", prompt="what?", expected=Expected(answers=("blue",)),
                group="g1", category="single-hop")

    assert task.expected.answers == ("blue",)
    assert task.expected.polarity == "positive"
    with pytest.raises(ValidationError):
        task.id = "t2"


def test_a_trace_is_one_task_one_attempt_and_carries_the_task_it_ran():
    trace = a_trace(
        recalled=Recall(documents=[Document(id="d1", content="Alex is a marine biologist.")],
                        declared={"results": ["d1"]}),
        answered=Answered(text="a marine biologist", produced_by="system"),
        timings_ms={"retrieve": 1.5})

    assert trace.task_id == "t1"
    assert trace.attempt == 1
    assert trace.recalled.documents[0].id == "d1", "the order is the rank"
    assert trace.recalled.declared == {"results": ["d1"]}
    assert trace.answered.produced_by == "system"


def test_a_failed_task_is_a_trace_with_a_reason_not_a_missing_row():
    trace = a_trace(error=Failure(step="retrieve", message="ConnectionError: refused"))

    assert trace.error.step == "retrieve"
    assert trace.recalled is None, "nothing recalled is not the same as recalling nothing"


def test_recalling_nothing_and_never_being_asked_are_different_traces():
    assert a_trace(recalled=Recall(documents=[])).recalled.documents == []
    assert a_trace().recalled is None


def test_a_declaration_that_declared_nothing_is_none_and_never_zero():
    declared = Declared()

    assert declared.version is None
    assert declared.tokens is None
    assert declared.engine_timings is None


def test_a_value_carries_its_measure_and_its_decider():
    value = Value(measure="word-match", decider=Decider.RULE, task_id="t1", value=1.0)

    assert (value.measure, value.decider) == ("word-match", Decider.RULE)
    assert Decider.MEMRANK.value == "memrank" and Scope.TASK.value == "task"


def test_an_evaluation_bundles_tasks_measures_and_a_clearing_rule():
    class Counted(Measure):
        name = "counted"
        reads = ("recalled",)

        def measure(self, traces, values):
            return [Value(measure=self.name, decider=self.decider, value=float(len(traces)))]

    evaluation = Evaluation(name="mine", version="1", clearing=Clearing.PER_TASK,
                            tasks=(Task(id="t1", prompt="a", group="g"),
                                   Task(id="t2", prompt="b", group="g"),
                                   Task(id="t3", prompt="c")),
                            measures=(Counted(),))

    assert evaluation.groups() == ["g", "t3"], "a task with no group is its own group"
    assert evaluation.measures[0].measure((), ()) == [
        Value(measure="counted", decider=Decider.RULE, value=0.0)]


def test_an_evaluation_may_state_that_its_material_cannot_be_frozen():
    assert Evaluation(name="live", version=UNFREEZABLE).version == UNFREEZABLE


def test_a_result_is_traces_and_named_values_or_a_refusal():
    result = Result(system=SystemRecord(name="Mine", kind="memory"),
                    evaluation=EvaluationRecord(name="demo", version="1", clearing="per-group"),
                    refusal="Mine is a memory system and does not supply retrieve",
                    started=WHEN, finished=WHEN)

    assert result.refused and result.traces == ()


def test_the_kinds_are_the_base_classes_and_each_names_its_own_verbs():
    class Mine(Model):
        def complete(self, prompt): return "an answer"

    assert kind_of(Mine()) == "model" and missing_verbs(Mine()) == ()
    assert issubclass(memrank.Memory, System)
    for kind in (Model, Retriever, Assistant):
        assert issubclass(kind, System)


def test_an_optional_declaration_is_a_plain_method_returning_none():
    class Mine(Model):
        def complete(self, prompt): return "an answer"

    system = Mine()
    assert system.declared_version() is None
    assert system.declared_tokens() is None
    assert system.declared_timings() is None
    assert system.declared_state() is None
    assert system.address is None


def test_the_memory_kind_is_the_one_memory_contract_under_the_new_name():
    from memrank.core import MemoryAdapter

    assert memrank.Memory is MemoryAdapter, "one memory contract, two names while it is renamed"
