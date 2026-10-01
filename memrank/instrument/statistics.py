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
#
# Used by the current path (service/scoring): it sits in a deprecated package but is not
# deprecated itself, and it must not warn.
"""The two readings a paired comparison needs, and nothing that draws a conclusion.

Both are the standard forms for paired data, and both are computed exactly rather than
approximated:

- **McNemar's exact test** over the discordant pairs of a binary measure -- the exact binomial
  two-sided tail, never the chi-square approximation, which is the approximation that misleads
  precisely where the discordant count is small (NIST/SEMATECH e-Handbook of Statistical
  Methods, section 7.3.5, McNemar's test).
- **A paired bootstrap percentile interval** for a continuous measure's mean difference,
  resampling task-pairs together and clustering by group where tasks share one, because tasks
  of one group are not independent draws (Miller, *Adding Error Bars to Evals*, 2024).

Deterministic: the bootstrap's seed and resample count are arguments, and the reading records
both, so the same two results always produce the same interval.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from math import comb

#: Below this many discordant pairs the split is not characterised, and the reading says so
#: instead of refusing -- refusing would hide the numbers that are there.
FEW_DISCORDANT = 10
#: Resamples for the bootstrap. Recorded in the reading beside the seed.
RESAMPLES = 2000
SEED = 20260921


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Two-sided exact McNemar p: the binomial tail of the smaller discordant count at p=0.5.

    The concordant pairs carry no information about the difference and do not enter, which is
    what makes this a test of the split rather than of the scores.
    """
    n = only_a + only_b
    if n == 0:
        return 1.0
    k = min(only_a, only_b)
    tail = sum(comb(n, i) for i in range(k + 1))
    return min(1.0, 2.0 * tail / (2 ** n))


def _clusters(pairs: Sequence[tuple[str, float, float]],
              groups: dict[str, str | None]) -> list[list[tuple[str, float, float]]]:
    """The pairs grouped by the cluster they are resampled as: their group, or themselves."""
    held: dict[str, list[tuple[str, float, float]]] = {}
    for pair in pairs:
        held.setdefault(groups.get(pair[0]) or pair[0], []).append(pair)
    return list(held.values())


def bootstrap_ci(pairs: Sequence[tuple[str, float, float]], groups: dict[str, str | None],
                 *, resamples: int = RESAMPLES, seed: int = SEED,
                 ) -> tuple[float | None, float | None]:
    """A 95% percentile interval for the mean of (b - a), resampling whole groups.

    Task-pairs move together: the same task's two values are one observation, because that is
    what pairing bought. Groups move together too, so a 200-question conversation counts as one
    draw rather than 200 independent ones.
    """
    clustered = _clusters(pairs, groups)
    if not clustered:
        return (None, None)
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(resamples):
        drawn = [pair
                 for _ in range(len(clustered))
                 for pair in clustered[rng.randrange(len(clustered))]]
        means.append(sum(b - a for _, a, b in drawn) / len(drawn))
    means.sort()
    low = means[int(0.025 * (len(means) - 1))]
    high = means[int(round(0.975 * (len(means) - 1)))]
    return (low, high)
