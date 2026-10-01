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
"""An uncapped arm's receipt reports what it carried, not the cap it was exempt from.

The defect, found in a real run rather than reasoned about: the faithful `hindsight` declares
`context_budget: uncapped`, was correctly handed its full context by `runner._context_text`, and
recorded `context_tokens_mean: 5000` for queries that actually carried **8,886** tokens
(`20260812-001623__locomo__smoke__811b17`, 204 results per query, recomputed from `per_query`).

Two functions were answering the same question differently: `_context_text` honoured the mode and
`cost.context_tokens` capped unconditionally. Only the record was wrong -- but the record is the
product, and this particular field is the one faithful mode exists to compare: AMB's published
LoCoMo row spends 36,235 context tokens per question, a number our receipt could not represent.

These tests pin the property at the seam that broke: the mode reaching `_drill_unit` from the
adapter, not just `cost.context_tokens` handling it once it arrives.
"""

from __future__ import annotations

import pytest

from memrank import runner
from memrank.core import AdapterResponse, BenchmarkUnit, Document

#: Comfortably more than the budget below, so an uncapped arm and a matched one cannot agree
#: by accident.
_LONG = "word " * 400
_BUDGET = 50


def _unit() -> BenchmarkUnit:
    return BenchmarkUnit(
        unit_id="u1", isolation_id="u1",
        documents=[Document(id="d1", content=_LONG)],
        queries=[{"id": "q1", "text": "what did they say?", "required_spans": ["word"]}])


def _responses() -> list[AdapterResponse]:
    return [AdapterResponse(query_id="q1",
                            documents=[Document(id=str(i), content=_LONG) for i in range(3)])]


def _tokens(budget_mode: str) -> int:
    rows = runner._drill_unit(_unit(), _responses(), _BUDGET, budget_mode)
    return rows[0]["context_tokens"]


def test_a_matched_arm_records_the_cap_that_bit_it():
    """The fairness control, unchanged: every leaderboard-eligible target is held to one budget."""
    assert _tokens("matched") == _BUDGET


def test_an_uncapped_arm_records_what_it_carried():
    assert _tokens("uncapped") > _BUDGET


def test_the_two_modes_disagree_which_is_the_whole_point():
    """A fix that made both paths return the same number would have fixed nothing."""
    assert _tokens("uncapped") != _tokens("matched")


@pytest.mark.parametrize("ref,expected", [("hindsight", "uncapped"),
                                         ("hindsight:matched", "matched")])
def test_the_mode_the_receipt_uses_comes_from_the_target(ref: str, expected: str):
    """The seam that broke was plumbing, not arithmetic.

    `cost.context_tokens` can be perfectly correct and the receipt still wrong if the runner never
    passes it the target's mode -- which is exactly what happened. This asserts the value
    `_drill_unit` is given is the one the adapter carries.
    """
    from memrank.targets import resolve_target
    from memrank.targets.factory import build_adapter

    adapter = build_adapter(resolve_target(ref), verify_engine=False)
    assert getattr(adapter, "context_budget", "matched") == expected
