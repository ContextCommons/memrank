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
"""The code that turns Jev's probabilities into verdicts and scores. No I/O, no model.

Every probability is thresholded at :data:`THRESHOLD`, one fixed number recorded with every
verdict. The probabilities themselves are kept beside it, so a different threshold later is a
re-scoring of the stored answers, not a rerun.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from memrank.judging.jev import questions as jq
from memrank.judging.judge import UnparseableVerdict
from memrank.metrics.ordering import (
    REVERSED,
    SAME,
    UNORDERED,
    OrderingScore,
    score_relations,
)

#: A condition holds when its probability, read in the direction that passes, is at least this.
THRESHOLD = 0.5

#: BEAM's scale, by Score level: not met, partly met, fully met.
RUBRIC_SCALE = (0.0, 0.5, 1.0)


def _answer(answers: dict[str, Any], key: str, kind: str) -> dict[str, Any]:
    answer = answers.get(key)
    if not isinstance(answer, dict) or answer.get("type") != kind:
        raise UnparseableVerdict(f"Jev's answers have no {kind} {key!r}: {answer!r}")
    return answer


def noul(answers: dict[str, Any], key: str) -> float:
    """P(yes) of Noul ``key``, refused when it is missing or not a probability."""
    value = _answer(answers, key, "noul").get("noul")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
        raise UnparseableVerdict(f"Jev's Noul {key!r} is not a probability: {value!r}")
    return float(value)


@dataclass(frozen=True)
class BinaryOutcome:
    """A binary verdict, and the probabilities it was decided from.

    ``passing`` is each condition's probability read in the direction that passes -- P(yes) for
    ``states_reference`` (and for ``declines`` on an abstention question), 1 - P(yes) for the
    others. ``passed`` is every one of them at least :data:`THRESHOLD`, and ``probability`` is
    the least of them, so ``passed`` is exactly ``probability >= THRESHOLD`` and a later
    threshold re-scores from this one number.
    """

    passed: bool
    probability: float
    passing: dict[str, float]
    rule: str


def binary(rules: jq.BinaryRules, answers: dict[str, Any],
           threshold: float = THRESHOLD) -> BinaryOutcome:
    """Pass when the answer states the reference, contradicts nothing, and does not decline; on
    an abstention question, pass when it declines."""
    if rules.abstention:
        passing = {jq.DECLINES: noul(answers, jq.DECLINES)}
        rule = jq.DECLINES
    else:
        passing = {jq.STATES: noul(answers, jq.STATES),
                   jq.CONTRADICTS: 1.0 - noul(answers, jq.CONTRADICTS),
                   jq.DECLINES: 1.0 - noul(answers, jq.DECLINES)}
        rule = f"{jq.STATES} and not {jq.CONTRADICTS} and not {jq.DECLINES}"
    least = min(passing.values())
    return BinaryOutcome(passed=least >= threshold, probability=least, passing=passing,
                         rule=f"{rule}, each at {threshold:g}")


@dataclass(frozen=True)
class CriterionOutcome:
    """One rubric criterion: its score on BEAM's scale and the distribution it came from."""

    criterion: str
    score: float
    expected_level: float
    probabilities: dict[str, float]


def _level(expected: float) -> int:
    """The expected level rounded half up, so 0.5 and 1.5 do not round to even."""
    return min(max(math.floor(expected + 0.5), 0), len(RUBRIC_SCALE) - 1)


def rubric(criteria: list[str], answers: dict[str, Any]) -> list[CriterionOutcome]:
    """Each criterion's expected Score level, rounded and mapped to 0 / 0.5 / 1."""
    outcomes = []
    for index, criterion in enumerate(criteria):
        answer = _answer(answers, jq.rubric_key(index), "score")
        expected = answer.get("score")
        if isinstance(expected, bool) or not isinstance(expected, (int, float)):
            raise UnparseableVerdict(f"Jev's Score for criterion {index} is not a number: "
                                     f"{expected!r}")
        outcomes.append(CriterionOutcome(
            criterion=criterion, score=RUBRIC_SCALE[_level(float(expected))],
            expected_level=float(expected),
            probabilities=dict(answer.get("probabilities") or {})))
    return outcomes


def _relation(answers: dict[str, Any], first: int, second: int) -> str:
    """How pair ``(first, second)`` is ordered in the answer, against the reference order."""
    picked = _answer(answers, jq.pair_key(first, second), "choice").get("choice")
    if picked not in (jq.A_FIRST, jq.B_FIRST, jq.UNCLEAR):
        raise UnparseableVerdict(f"Jev's order choice for {first},{second} is {picked!r}")
    if picked == jq.UNCLEAR:
        return UNORDERED
    a, _ = jq.pair_events(first, second)
    earlier_first = (picked == jq.A_FIRST) == (a == first)
    return SAME if earlier_first else REVERSED


@dataclass(frozen=True)
class OrderingOutcome:
    """An extraction-free ordering score, and every judgement it was counted from."""

    scored: OrderingScore
    present: list[int]
    relations: dict[str, str]
    extras: bool


def ordering(n_events: int, answers: dict[str, Any],
             threshold: float = THRESHOLD) -> OrderingOutcome:
    """Coverage, order and extras from the answers, scored by :func:`score_relations`.

    An order is read only for pairs of events the answer asserts: a pair with an absent event has
    no order in the answer to agree or disagree with.
    """
    present = [i for i in range(n_events) if noul(answers, jq.present_key(i)) >= threshold]
    relations = {(i, j): _relation(answers, i, j)
                 for i in present for j in present if i < j}
    extras = noul(answers, jq.EXTRAS) >= threshold
    scored = score_relations(n_events, present, relations, extras)
    return OrderingOutcome(scored=scored, present=present,
                           relations={f"{i},{j}": r for (i, j), r in relations.items()},
                           extras=extras)
