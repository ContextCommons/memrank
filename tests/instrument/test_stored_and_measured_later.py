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
"""A result persists, and a measure thought of afterwards runs over the traces it stored.

This is the case decision 0009 moved scoring out of the run loop for (C23): a person who
thinks of a metric after the run must be able to get it from what was recorded, without
running the system again. So the test that matters is the equality: measuring afterwards, over
a result that has been through JSON, gives what measuring in the run gave.
"""
from __future__ import annotations

import pytest

from memrank.instrument.catalog import evaluation
from memrank.instrument.measures import FailureRate, WordMatch
from memrank.instrument.result import Result
from memrank.instrument.run import measure
from tests.instrument.fakes import TinyMemory, two_task_evaluation

# These tests exercise the deprecated in-process route on purpose; its notice is expected here.
pytestmark = pytest.mark.filterwarnings(
    "ignore:.*is deprecated and will be removed:DeprecationWarning")


def test_a_saved_result_loads_back_with_its_traces_typed(tmp_path):
    original = two_task_evaluation(measures=(WordMatch(),)).run(system=TinyMemory())

    path = original.save(tmp_path / "run.json")
    loaded = Result.load(path)

    assert loaded == original
    assert loaded.traces[0].task.expected.required_spans == ("marine biologist",)
    assert loaded.traces[0].recalled.documents[0].content.startswith("Alex")
    assert loaded.values[0].decider is original.values[0].decider


def test_measuring_afterwards_over_a_saved_result_equals_measuring_in_the_run(tmp_path):
    system = TinyMemory()
    measured_in_the_run = two_task_evaluation(measures=(WordMatch(),)).run(system=system)

    without = two_task_evaluation().run(system=TinyMemory())
    later = measure(Result.load(without.save(tmp_path / "run.json")), WordMatch())

    assert [(v.measure, v.task_id, v.value) for v in later.values] == [
        (v.measure, v.task_id, v.value) for v in measured_in_the_run.values]


def test_a_measure_added_later_keeps_what_was_already_measured():
    result = two_task_evaluation(measures=(WordMatch(),)).run(system=TinyMemory())

    extended = measure(result, FailureRate())

    assert [v.measure for v in extended.values] == ["word-match", "word-match", "failure-rate"]
    assert result.values == tuple(extended.values[:2]), "the original result is unchanged"


def test_measuring_a_refused_run_says_there_is_nothing_to_measure():
    class NotASystem:
        address = None

    refused = two_task_evaluation().run(system=NotASystem())

    with pytest.raises(ValueError, match="no traces to measure"):
        measure(refused, WordMatch())


def test_a_stored_result_states_the_schema_it_was_written_under(tmp_path):
    path = two_task_evaluation().run(system=TinyMemory()).save(tmp_path / "run.json")
    written = path.read_text(encoding="utf-8")
    path.write_text(written.replace('"instrument-1"', '"instrument-99"'), encoding="utf-8")

    with pytest.raises(ValueError, match="instrument-99"):
        Result.load(path)


def test_a_demo_result_round_trips_and_re_measures_to_the_same_numbers(tmp_path):
    first = evaluation("demo").run(system=TinyMemory())

    again = measure(Result.load(first.save(tmp_path / "demo.json")), FailureRate())

    assert [v.value for v in again.values_of("word-match")] == [
        v.value for v in first.values_of("word-match")]
    assert again.values_of("failure-rate")[0].value == 0.0
