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
"""Hindsight scopes memory the way AMB does, and nothing may quietly collapse that.

The audit recorded hindsight's partitioning as DRIFT -- "bank per question" for AMB against "one
bank" for us (F3). It is not drift. AMB keys banks by USER --
``"bank_id": self._bank_id_for(user_id)`` -- and memrank creates one bank per benchmark unit.
On LongMemEval a unit IS a question, so both produce one bank per question; on LoCoMo both
produce one bank per conversation.

So no `partitioning:` block was added for this engine -- it would express what already happens. What
was missing is anything that FAILS if the scoping changes, which is what these tests are. They pin
the property, not the implementation: the runner deriving its scope from ``unit.isolation_id``
(runner.py:_run_unit_repeats) is the mechanism today, and a future change that scoped by run alone
would silently merge every question's haystack into one bank and read as a much worse engine.
"""

from __future__ import annotations

from memrank.adapters.hindsight import _safe_bank_id
from memrank.benchmarks.locomo import LoCoMoBenchmark
from memrank.benchmarks.longmemeval import LongMemEvalBenchmark

RUN_PREFIX = "memrank-run-2f9a1c"


def _bank_ids(units) -> list[str]:
    """The bank each unit gets, derived exactly as the runner + adapter derive it."""
    return [_safe_bank_id(f"{RUN_PREFIX}-{unit.isolation_id}") for unit in units]


def test_every_longmemeval_question_gets_its_own_bank():
    """A LongMemEval unit is one question, so per-unit scoping IS AMB's bank-per-question."""
    units = LongMemEvalBenchmark(slice="smoke").load()
    banks = _bank_ids(units)
    assert len(units) > 1, "a single-unit slice could not detect a collapse"
    assert len(set(banks)) == len(units), (
        f"{len(units)} questions share {len(set(banks))} banks. Merged haystacks make every "
        f"question searchable from every other one's sessions.")


def test_a_bank_id_is_derived_from_the_unit_not_only_the_run():
    """The regression that would be invisible: same run, different units, same bank."""
    units = LongMemEvalBenchmark(slice="smoke").load()
    for unit, bank in zip(units, _bank_ids(units), strict=True):
        assert unit.isolation_id in bank, (
            f"bank {bank!r} does not carry unit {unit.isolation_id!r}, so two units in one run "
            f"would collide")


def test_a_locomo_conversation_is_one_bank():
    """LoCoMo units are conversations, and AMB keys by user there too -- one bank, many questions."""
    units = LoCoMoBenchmark(slice="smoke").load()
    banks = _bank_ids(units)
    assert len(set(banks)) == len(units)
    assert len(units[0].queries) > 1, (
        "the point of this assertion is that many questions share one conversation's bank")


def test_bank_ids_stay_within_what_the_api_accepts():
    """hindsight's bank_id parser rejects anything outside [A-Za-z0-9_-], capped at 64."""
    units = LongMemEvalBenchmark(slice="smoke").load()
    for bank in _bank_ids(units):
        assert bank and len(bank) <= 64
        assert all(character.isalnum() or character in "-_" for character in bank)
