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
"""Scoring a sequence against a reference sequence: how much, and in what order.

BEAM's `event_ordering` asks a model to list events in the order the user raised them, so its
rubric is an ORDERED list rather than a set of independent criteria. Averaging it the way the
other nine abilities are averaged would score a perfectly reversed answer identically to a correct
one, which is the one thing this ability exists to detect.

Pure functions, no LLM and no I/O: the alignment that decides which predicted event corresponds to
which reference event is a judge call and lives in `judge_shape`. Everything here is arithmetic on
already-aligned sequences, which is what makes it testable against known values instead of against
a model's opinion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class OrderingScore:
    """Coverage and order, kept separate as well as combined."""
    precision: float
    recall: float
    f1: float
    tau_norm: float
    #: `tau_norm * f1`. Reported ALONE would be dishonest: `tau_norm` is `(tau_b + 1) / 2`, so an
    #: answer correlating with nothing scores ~0.5 and an empty one is not distinguishable from a
    #: half-right one. BEAM's own `report_results.py` prints the bare `tau_norm`; their paper
    #: describes the metric as "capturing both recall and ordering fidelity", which only the
    #: product does. We follow the paper -- decision-beam-targets-the-spec-not-the-harness.
    score: float


def kendall_tau_b(x: list[float], y: list[float]) -> float:
    """Kendall's tau-b: rank correlation with a tie correction, in [-1, 1].

    Implemented rather than imported because `scipy` is not a dependency and pulling it in for one
    coefficient is a poor trade. Tie handling is the whole reason this is tau-b and not tau-a:
    unmatched events all share one rank (see `score_sequence`), so ties are the common case here,
    and tau-a would divide by a pair count that includes pairs it cannot order.

    Returns 0.0 when either vector is entirely tied -- the correlation is undefined and there is no
    honest value to prefer.
    """
    if len(x) != len(y):
        raise ValueError(f"vectors must be the same length, got {len(x)} and {len(y)}")
    n = len(x)
    concordant = discordant = tied_x = tied_y = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx, dy = x[i] - x[j], y[i] - y[j]
            # A pair tied in BOTH counts toward both corrections, which is what makes the
            # denominator sqrt((n0 - n1)(n0 - n2)) rather than a single term.
            if dx == 0:
                tied_x += 1
            if dy == 0:
                tied_y += 1
            if dx != 0 and dy != 0:
                if (dx > 0) == (dy > 0):
                    concordant += 1
                else:
                    discordant += 1
    n0 = n * (n - 1) / 2
    denominator = math.sqrt((n0 - tied_x) * (n0 - tied_y))
    return (concordant - discordant) / denominator if denominator else 0.0


def score_sequence(reference: list[str], predicted: list[str]) -> OrderingScore:
    """Score an ALIGNED predicted sequence against the reference sequence.

    ``predicted`` must already be canonicalised: an event judged equivalent to a reference event
    carries that reference event's exact string, and anything unmatched keeps its own text. Set
    membership is then plain equality, and this function never has to decide what "the same event"
    means -- that judgement is a separate, auditable step.

    Ranking: both sequences are ranked over their union, and every item absent from a sequence
    takes one shared rank past the end. That is what turns "this event is missing" into a tie
    rather than into an arbitrary position, and it is why tau-b is the right coefficient.
    """
    ref_set, pred_set = set(reference), set(predicted)
    true_positives = len(ref_set & pred_set)
    precision = true_positives / len(predicted) if predicted else 0.0
    recall = true_positives / len(reference) if reference else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    union = list(dict.fromkeys(reference + predicted))
    absent_rank = len(union) + 1

    def ranks(sequence: list[str]) -> list[float]:
        position = {item: i + 1 for i, item in enumerate(sequence)}
        return [float(position.get(item, absent_rank)) for item in union]

    tau_norm = (kendall_tau_b(ranks(reference), ranks(predicted)) + 1) / 2
    return OrderingScore(precision=precision, recall=recall, f1=f1,
                         tau_norm=tau_norm, score=tau_norm * f1)
