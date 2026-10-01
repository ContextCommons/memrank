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
"""The shipped engine on the shipped evaluation, through the top-level run. No fakes.

The smoke run that found the defect this file now holds shut: every registered adapter returns
a `memrank.contract.Recall`, and a run loop still unpacking retrieve's old tuple turned all
five demo tasks into errors -- while the benchmark measure went on printing a number over
them, because a negative query with nothing retrieved scores as a hit. A run whose every task
failed must say so, not score it.

So this asserts the two halves together: zero errors, and a value per task that is not None.
"""
from __future__ import annotations

import pytest

import memrank
from memrank.adapters import REGISTRY
from memrank.contract import Recall

# These tests exercise the deprecated in-process route on purpose; its notice is expected here.
pytestmark = pytest.mark.filterwarnings(
    "ignore:.*is deprecated and will be removed:DeprecationWarning")


def test_word_overlap_runs_demo_end_to_end_with_no_errors():
    result = memrank.evaluation("demo").run(system=REGISTRY["word-overlap"]())

    assert result.refusal is None
    assert [t.error for t in result.traces] == [None] * 5, (
        "\n".join(f"{t.task_id}: {t.error.step} {t.error.message}"
                  for t in result.traces if t.error))
    assert all(isinstance(t.recalled, Recall) for t in result.traces)


def test_every_task_gets_a_value_that_is_not_none():
    result = memrank.evaluation("demo").run(system=REGISTRY["word-overlap"]())

    marks = result.values_of("word-match")
    assert len(marks) == 5
    assert [v.value for v in marks] == [1.0, 1.0, 1.0, 0.0, 1.0]
    assert result.values_of("demo-score")[0].value == 0.8
    assert result.values_of("failure-rate")[0].value == 0.0


def test_a_group_whose_every_task_failed_is_not_given_a_number():
    class Broken(type(REGISTRY["word-overlap"]())):
        def retrieve(self, query, k, user_id, query_timestamp=None):
            raise RuntimeError("engine unreachable")

    result = memrank.evaluation("demo").run(system=Broken())

    assert [t.error.step for t in result.traces] == ["retrieve"] * 5
    scored = result.values_of("demo-score")[0]
    assert scored.value is None, "a benchmark cannot score responses that never happened"
    assert "nothing to score" in scored.why
    assert result.values_of("failure-rate")[0].value == 1.0


def test_the_constructor_wins_over_the_package_of_the_same_name():
    """`memrank.evaluation` is a verb of the seven AND the package the cell run lives in.

    An imported submodule is set as an attribute of its package, and that beats a module's
    `__getattr__` -- so once anything had imported `memrank.evaluation.api`, this call reached
    the module and raised `'module' object is not callable`, depending on import order.
    """
    import memrank.evaluation.api  # noqa: F401 - the import is the precondition being tested

    assert callable(memrank.evaluation)
    assert memrank.evaluation("demo").name == "demo"

    from memrank.evaluation.api import run as cell_run

    assert cell_run.__module__ == "memrank.evaluation.api", "the package is still reachable"
