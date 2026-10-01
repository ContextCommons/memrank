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
"""The one rule that decides a cell's headline score.

Every surface that shows "the score" reads it from here: ``memrank runs ls``, ``runs show``, the
runs API (and therefore the web GUI), the runner's end-of-run line, the leaderboard and
``compare``. They disagreed before. Three of them carried their own copy of the legacy
rankability default and one of those copies was wrong, so a legacy BEAM record ranked in the GUI
while every other surface withheld it. Worse, none of them promoted the JUDGED score: a judged
BEAM run -- the only kind of BEAM run that measures anything -- displayed "—" everywhere while its
real number sat unread in ``judged_metrics``.

The rule, in precedence order:

1. **Judged**, when the run was judged and covered any questions at all. This is the benchmarks'
   own protocol (LoCoMo and LongMemEval grade a generated answer; BEAM grades rubric nuggets),
   so when it exists it is not one score among several -- it is the score. Rankable only at
   ``MIN_RANKABLE_COVERAGE`` or better, because a number computed over a third of the questions
   is a real measurement of an unstated subset, publishable but not comparable.
2. **The self-contained composite**, when the benchmark declares one and it is rankable. Today
   that means the internal ``demo`` fixture and ``relation_graph``'s structural score; the three
   external benchmarks no longer report a composite at all.
3. **Withheld.** No number is the honest answer, and it is a different statement from zero.

Deliberately artifact-driven: nothing here branches on a benchmark's name. An old artifact
carries the flags it was written with and is projected by what it says, so re-reading history
never depends on today's registry.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

#: Judged coverage below this is real but not comparable -- see rule 1.
MIN_RANKABLE_COVERAGE = 0.5

#: The kind reported when the judge produced the number. One name for every benchmark: the
#: runner writes rubric averages and binary verdicts alike into ``answer_correctness``, so the
#: shape that produced it is a property of the benchmark, not of this projection.
JUDGED = "judged_answer_correctness"

#: The kind reported when there is no number to show.
WITHHELD = "withheld"

#: What a cell that names no quality metric is measuring. Only legacy artifacts omit it.
DEFAULT_QUALITY_METRIC = "substring_recall"


@dataclass(frozen=True)
class Headline:
    """One cell's headline score, and what it is.

    Attributes:
        value: The number, or ``None`` when withheld.
        kind: :data:`JUDGED`, the benchmark's ``quality_metric``, or :data:`WITHHELD`.
        rankable: Whether this may be ranked against other runs, not merely displayed.
        coverage: The judged fraction, when the judge produced the value; otherwise ``None``.
    """

    value: float | None
    kind: str
    rankable: bool
    coverage: float | None


def rankable_composite(cell: Mapping[str, Any]) -> bool:
    """Whether ``cell``'s composite is a self-contained quality score to display and rank.

    The legacy default, unified. An artifact written before ``composite_rankable`` existed falls
    back to ``substring_recall_supported``, which keeps legacy BEAM withheld and legacy locomo
    shown -- the behaviour every surface but one already had. A key present but ``None`` (a
    leaderboard row read before validation) counts as absent rather than as a refusal, since
    ``None`` there means "not stated", not "not rankable".
    """
    declared = cell.get("composite_rankable")
    if declared is not None:
        return bool(declared)
    supported = cell.get("substring_recall_supported")
    return True if supported is None else bool(supported)


def composite_display(composite_rankable: bool, composite: float | None) -> str:
    """Render the recall/composite cell: the number when the composite is a
    self-contained rankable score, else withheld (judge required).

    ``None`` is withheld on the same footing as unrankable: a benchmark whose quality metric is
    the judge's reports no composite at all, and there is no number to print for it."""
    if not composite_rankable or composite is None:
        return "n/a (judge required)"
    return f"{composite:.3f}"


def cell_headline(cell: Mapping[str, Any]) -> Headline:
    """The headline score for one cell -- artifact, summary entry or synced record alike.

    Args:
        cell: Any mapping carrying the cell's scoring fields (``judged_metrics``, ``composite``,
            ``composite_rankable``, ``substring_recall_supported``, ``quality_metric``). Absent
            keys are legacy, never errors.

    Returns:
        The :class:`Headline`. Never raises: a listing that cannot project one row must still
        render the rest.
    """
    judged = cell.get("judged_metrics")
    if isinstance(judged, Mapping):
        coverage = judged.get("judged_coverage")
        if coverage:
            return Headline(value=judged.get("answer_correctness"), kind=JUDGED,
                            rankable=coverage >= MIN_RANKABLE_COVERAGE, coverage=coverage)
    composite = cell.get("composite")
    if composite is not None and rankable_composite(cell):
        return Headline(value=composite, kind=cell.get("quality_metric") or DEFAULT_QUALITY_METRIC,
                        rankable=True, coverage=None)
    return Headline(value=None, kind=WITHHELD, rankable=False, coverage=None)
