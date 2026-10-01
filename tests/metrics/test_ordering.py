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
"""Sequence scoring: how much was recovered, and in what order.

BEAM's `event_ordering` asks for events in the order the user raised them, so its rubric is an
ORDERED list. Averaging it like the other nine abilities would score a perfectly reversed answer
identically to a correct one -- which is the single thing this ability exists to detect, and the
first test below is the one that would catch losing it.

Kendall tau-b is implemented rather than imported (scipy is not a dependency), so it is pinned
against hand-computable values rather than trusted by inspection.
"""

import math

from memrank.metrics.ordering import kendall_tau_b, score_sequence

_GOLD = ["core functionality", "transaction errors", "security review"]


def test_tau_b_against_hand_computable_values():
    assert kendall_tau_b([1, 2, 3], [1, 2, 3]) == 1.0
    assert kendall_tau_b([1, 2, 3], [3, 2, 1]) == -1.0
    # 3 pairs: (1,2) concordant, (1,3) concordant, (2,3) discordant -> (2-1)/3
    assert math.isclose(kendall_tau_b([1, 2, 3], [1, 3, 2]), 1 / 3)


def test_tau_b_is_zero_when_a_vector_is_entirely_tied():
    """Undefined rather than zero-correlated, and there is no honest value to prefer. Matters here
    because unmatched events all share one rank, so ties are the common case, not the edge."""
    assert kendall_tau_b([1, 1, 1], [1, 2, 3]) == 0.0


def test_a_perfect_answer_scores_one():
    assert score_sequence(_GOLD, list(_GOLD)).score == 1.0


def test_a_reversed_answer_scores_near_zero():
    """The whole reason event_ordering is not nugget-averaged: every event is present, so a
    coverage-only metric would call this perfect."""
    reversed_ = score_sequence(_GOLD, list(reversed(_GOLD)))
    assert reversed_.f1 == 1.0, "all events recovered"
    assert reversed_.tau_norm == 0.0, "and in exactly the wrong order"
    assert reversed_.score == 0.0


def test_an_empty_answer_scores_zero_not_a_half():
    """Bare `tau_norm` floors an uncorrelated answer at 0.5, which is why BEAM's own harness
    reporting it alone overstates this ability. The product does not."""
    empty = score_sequence(_GOLD, [])
    assert empty.f1 == 0.0
    assert empty.score == 0.0


def test_a_missing_event_costs_coverage_but_keeps_order():
    partial = score_sequence(_GOLD, ["core functionality", "security review"])
    assert partial.recall < 1.0 and partial.precision == 1.0
    assert partial.tau_norm > 0.5, "what it did list is still in the right order"
    assert 0.0 < partial.score < 1.0


def test_extra_events_cost_precision():
    noisy = score_sequence(_GOLD, [*_GOLD, "unrelated thing"])
    assert noisy.recall == 1.0 and noisy.precision < 1.0
    assert noisy.score < 1.0


def test_duplicated_events_do_not_inflate_the_score():
    """Set membership means a repeated event cannot be counted twice, but it still costs
    precision -- listing the same thing three times is not a correct ordering."""
    duped = score_sequence(_GOLD, ["core functionality", "core functionality", "core functionality"])
    assert duped.recall < 1.0
    assert duped.score < 1.0
