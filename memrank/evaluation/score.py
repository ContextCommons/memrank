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
"""The headline number with the declarations that make it readable.

``metrics.headline`` already owns the RULE for which number represents a cell, and nothing
here re-decides it: :func:`score_of` reads that rule's answer and attaches what the number
is of, what it is not, and who decided it -- the prose that lived in
``core.QUALITY_METRIC_DESCRIPTIONS``, out of a result's reach.

This is the read form. Nothing here is stored; ``EvalResult.to_dict()`` is untouched.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from memrank.evaluation.case import DECIDED_BY_JUDGE, DECIDED_BY_MEMRANK
from memrank.quality import WITHHELD_DECLARATION, declaration_for


@dataclass(frozen=True)
class Score:
    """The headline number, and the honest declarations that travel with it.

    The number alone is not readable: ``0.8`` under ``quality_metric:
    "substring_recall"`` has always needed prose that lived in a constant the result could
    not reach. :attr:`of` and :attr:`not_of` are that prose, carried here.

    ``decided_by`` rather than a judged boolean, because the question a reader has is who
    decided the number -- memrank's deterministic proxy or a model -- and a boolean answers
    it only for as long as those are the only two possibilities.
    """

    #: What the number measures. Always stated, including when there is no number.
    of: str
    #: What it does NOT establish. ``None`` when the metric has nothing to disclaim.
    not_of: str | None = None
    #: The number, or ``None`` when withheld. ``None`` is not zero.
    value: float | None = None
    #: :data:`DECIDED_BY_MEMRANK`, :data:`DECIDED_BY_JUDGE`, or ``None`` when nothing decided
    #: a number at all.
    decided_by: str | None = None
    #: Why there is no number, when there is none. ``None`` when there is one.
    withheld: str | None = None
    #: Whether this may be RANKED against other runs, not merely displayed.
    rankable: bool = False
    #: The metric's short human-facing label, e.g. ``"recall"``.
    label: str = ""
    #: The metric kind the result declared, e.g. ``"substring_recall"``.
    kind: str = ""
    #: The judged fraction, when a judge produced the number; ``None`` otherwise. A number
    #: over a third of the questions is a real measurement of an unstated subset.
    coverage: float | None = None


def score_of(cell: Mapping[str, Any]) -> Score:
    """The headline for one cell, with its declarations. Reads the cell; writes nothing."""
    from memrank.metrics.headline import DEFAULT_QUALITY_METRIC, JUDGED, WITHHELD, cell_headline

    headline = cell_headline(cell)
    declared = declaration_for(cell.get("quality_metric") or DEFAULT_QUALITY_METRIC)
    if headline.kind == WITHHELD:
        # The number is absent, so the declaration is about the number this evaluation WOULD
        # have produced -- which is what a reader needs to know to get one. `withheld` says
        # why there is none, from the metric itself where it has something to say.
        return Score(of=declared.of, not_of=declared.not_of, label=declared.label,
                     kind=WITHHELD,
                     withheld=declared.withheld or WITHHELD_DECLARATION.withheld)
    if headline.kind != JUDGED:
        return Score(of=declared.of, not_of=declared.not_of, value=headline.value,
                     decided_by=DECIDED_BY_MEMRANK, rankable=headline.rankable,
                     label=declared.label, kind=headline.kind)
    judged = declaration_for(JUDGED)
    return Score(of=judged.of, not_of=judged.not_of, value=headline.value,
                 decided_by=DECIDED_BY_JUDGE, rankable=headline.rankable,
                 label=judged.label, kind=JUDGED, coverage=headline.coverage)
