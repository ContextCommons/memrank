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
"""The run: what it refuses before it starts, and what it guarantees once it does.

Four guarantees are checked here, because each one has failed somewhere before: a run that
cannot be set up refuses BEFORE the system is touched; every task yields exactly one trace per
attempt; a task's failure does not stop the loop; and state is cleared where the evaluation's
rule says it is.
"""
from __future__ import annotations

import pytest

from memrank.instrument.evaluation import Clearing, Evaluation
from memrank.instrument.task import Task
from tests.instrument.fakes import (
    Broke,
    Counts,
    FirstPassage,
    NeedsAnAnswer,
    ReadsTheUnknown,
    TinyAssistant,
    TinyMemory,
    TinyModel,
    TinyRetriever,
    two_task_evaluation,
)

# These tests exercise the deprecated in-process route on purpose; its notice is expected here.
pytestmark = pytest.mark.filterwarnings(
    "ignore:.*is deprecated and will be removed:DeprecationWarning")


def test_a_run_produces_one_trace_per_task_and_values_that_name_their_decider():
    system = TinyMemory()

    result = two_task_evaluation(measures=(Broke(),)).run(system=system)

    assert result.refusal is None
    assert [t.task_id for t in result.traces] == ["t_job", "t_visit"]
    assert [(v.measure, v.task_id, v.value) for v in result.values] == [
        ("broke", "t_job", False), ("broke", "t_visit", False)]
    assert all(v.decider.value == "memrank" for v in result.values)


def test_the_system_is_given_the_group_s_documents_once_and_the_trace_says_which():
    system = TinyMemory()

    result = two_task_evaluation().run(system=system)

    assert system.prepared == ["g1"], "one group, one preparation"
    assert len(system.store.get("g1", [])) == 0, "cleared at the end of the group"
    assert result.traces[0].given.document_ids == ("d1", "d2")
    assert result.traces[0].given.count == 2


def test_a_task_that_breaks_is_a_trace_with_a_reason_and_the_loop_carries_on():
    system = TinyMemory(fails_on="What is Alex?")

    result = two_task_evaluation(measures=(Broke(),)).run(system=system)

    broke, ran = result.traces
    assert broke.error.step == "retrieve" and "retrieve refused" in broke.error.message
    assert ran.error is None and ran.recalled
    assert [v.value for v in result.values_of("broke")] == [True, False]


def test_a_group_that_cannot_be_prepared_still_yields_one_trace_per_task():
    system = TinyMemory(fails_on="g1", fails_at="prepare")

    result = two_task_evaluation().run(system=system)

    assert len(result.traces) == 2
    assert {t.error.step for t in result.traces} == {"prepare"}


def test_attempts_produce_one_trace_per_task_per_attempt():
    result = two_task_evaluation().run(system=TinyMemory(), attempts=3)

    assert len(result.traces) == 6
    assert sorted(t.attempt for t in result.traces) == [1, 1, 2, 2, 3, 3]
    assert len(result.traces_of("t_job")) == 3


def test_the_clearing_rule_in_force_is_recorded_and_observed():
    system = TinyMemory()

    per_task = two_task_evaluation(clearing=Clearing.PER_TASK).run(system=system)

    assert system.prepared == ["t_job", "t_visit"]
    assert system.cleaned == 2
    assert per_task.evaluation.clearing == "per-task"
    assert per_task.evaluation.cleared is True


def test_clearing_at_the_end_prepares_once_over_the_whole_evaluation():
    system = TinyMemory()

    two_task_evaluation(clearing=Clearing.AT_END).run(system=system)

    assert system.prepared == ["tiny"] and system.cleaned == 1


def test_a_run_refuses_before_touching_a_system_that_lacks_a_required_verb():
    class NotEvenASystem:
        address = None

    result = two_task_evaluation().run(system=NotEvenASystem())

    assert result.traces == () and result.values == ()
    assert "not a memrank System" in result.refusal


def test_a_run_refuses_a_measure_that_reads_a_name_nothing_produces():
    system = TinyMemory()

    result = two_task_evaluation(measures=(ReadsTheUnknown(),)).run(system=system)

    assert "reads vibes" in result.refusal
    assert system.prepared == [], "the refusal happened before the first prepare"


def test_a_measure_may_read_another_measure_s_name():
    class Doubles(Counts):
        name = "doubles"
        reads = ("counts",)

    result = two_task_evaluation(measures=(Counts(), Doubles())).run(system=TinyMemory())

    assert result.refusal is None
    assert [v.measure for v in result.values] == ["counts", "doubles"]


def test_a_run_refuses_when_a_measure_needs_an_answer_and_nothing_writes_one():
    system = TinyMemory()

    result = two_task_evaluation(measures=(NeedsAnAnswer(),)).run(system=system)

    assert "answerer=" in result.refusal and system.prepared == []


def test_an_answer_writer_satisfies_that_refusal_and_is_named_on_the_trace():
    result = two_task_evaluation(measures=(NeedsAnAnswer(),)).run(
        system=TinyMemory(), answerer=FirstPassage())

    assert result.refusal is None
    assert result.traces[0].answered.produced_by == "first-passage"
    assert all(v.value is True for v in result.values_of("needs-an-answer"))


def test_a_system_that_answers_for_itself_needs_no_writer():
    for system in (TinyModel(), TinyAssistant()):
        evaluation = Evaluation(name="asks", version="1",
                                tasks=(Task(id="t1", prompt="What about whales?"),),
                                measures=(NeedsAnAnswer(),))

        result = evaluation.run(system=system)

        assert result.refusal is None
        assert result.traces[0].answered.produced_by == "system"


def test_a_system_that_cannot_be_told_things_refuses_material_rather_than_ignoring_it():
    result = two_task_evaluation().run(system=TinyRetriever())

    assert "no verb to be told things with" in result.refusal


def test_a_retriever_ranks_its_own_corpus():
    evaluation = Evaluation(name="ranking", version="1",
                            tasks=(Task(id="t1", prompt="largest animal"),))

    result = evaluation.run(system=TinyRetriever())

    assert result.refusal is None
    assert [d.id for d in result.traces[0].recalled.documents] == ["c1"]


def test_the_result_records_the_system_and_the_evaluation_version():
    result = two_task_evaluation().run(system=TinyMemory())

    assert result.system.name == "TinyMemory" and result.system.kind == "memory"
    assert result.system.version == "tiny-1", "the system's own word, recorded as its word"
    assert (result.evaluation.name, result.evaluation.version) == ("tiny", "1")
