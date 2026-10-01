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
"""The paired reading: two results, laid side by side, task by task.

A lens above the run, not one of the seven. It introduces no member and no state: it reads two
results and returns something that is not a result.

It refuses unless both results are of the same evaluation at the same version -- two runs of
different measurements are not two measurements of two systems. Then, per measure, it pairs by
task id and reports what is there: the rates or the means, the gap, the flips, and how often
chance produces a split that size. It never says "better"; which system is preferable depends
on what the person is buying, and no number here knows that.

`print(paired)` is the whole of the read, and it ends by saying in a line that nothing above it
says "better" -- so the caveat travels with the numbers instead of living in whatever script
happened to format them.

The statistics are in `memrank.instrument.statistics`, with their sources: McNemar's exact test
(NIST/SEMATECH e-Handbook, 7.3.5) for a binary measure, and a cluster-resampled paired
bootstrap (Miller, *Adding Error Bars to Evals*, 2024) for a continuous one.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict

from memrank.errors import MemrankError
from memrank.instrument.measure import Value
from memrank.instrument.result import Result
from memrank.instrument.statistics import (
    FEW_DISCORDANT,
    RESAMPLES,
    SEED,
    bootstrap_ci,
    mcnemar_exact,
)

#: What is said instead of characterising a gap the data cannot characterise.
TOO_FEW = "too few discordant tasks to characterise the gap"

#: The last line of every paired reading, so the caveat is never left behind by the numbers.
NOT_BETTER = ("A gap is a gap. Nothing above says which system is better; that depends on what\n"
              "you are buying, and these numbers do not know what that is.")

#: How many flipped tasks are named before the list is left to `pairing.flips`.
FLIPS_SHOWN = 3


class PairingRefused(MemrankError):
    """Two results that are not two readings of the same thing cannot be paired."""


class Flip(BaseModel):
    """One task whose value changed between the two runs."""

    model_config = ConfigDict(frozen=True)

    task_id: str
    was: float | bool | None
    now: float | bool | None


class MeasurePairing(BaseModel):
    """One measure's reading over the tasks both runs have."""

    model_config = ConfigDict(frozen=True)

    measure: str
    kind: Literal["binary", "continuous"]
    #: Tasks present in both runs for this measure, with a value in both.
    tasks: int
    mean_a: float
    mean_b: float
    #: ``mean_b - mean_a``. A gap, never a verdict.
    gap: float
    #: The 2x2 of a binary measure: both right, both wrong, only A, only B.
    both: int | None = None
    neither: int | None = None
    only_a: int | None = None
    only_b: int | None = None
    #: Tasks whose value differs between the runs. The only pairs that carry information.
    discordant: int = 0
    #: Exact McNemar two-sided p over the discordant pairs. Binary measures only.
    p_value: float | None = None
    #: 95% paired bootstrap percentile interval of the mean difference. Continuous only.
    ci_low: float | None = None
    ci_high: float | None = None
    resamples: int | None = None
    seed: int | None = None
    flips: tuple[Flip, ...] = ()
    caution: str | None = None

    def __str__(self) -> str:
        """One measure's reading: the means, the gap, the counts, the flips, the caution."""
        lines = [f"{self.measure} ({self.kind}, {self.tasks} paired task(s))",
                 f"  mean A {self.mean_a:.3f}   mean B {self.mean_b:.3f}   "
                 f"gap {self.gap:+.3f}   {self.discordant} discordant"]
        if self.kind == "binary":
            lines.append(f"  both {self.both}  neither {self.neither}  only A {self.only_a}  "
                         f"only B {self.only_b}  McNemar exact p = {self.p_value}")
        else:
            lines.append(f"  95% paired bootstrap [{self.ci_low:.3f}, {self.ci_high:.3f}] "
                         f"over {self.resamples} resamples, seed {self.seed}")
        lines += [f"  flipped: {flip.task_id}  {flip.was} -> {flip.now}"
                  for flip in self.flips[:FLIPS_SHOWN]]
        if self.caution:
            lines.append(f"  caution: {self.caution}")
        return "\n".join(lines)


class Paired(BaseModel):
    """Two results, paired. Reports what differs; the person interprets."""

    model_config = ConfigDict(frozen=True)

    evaluation: str
    version: str
    system_a: str
    system_b: str
    only_in_a: tuple[str, ...] = ()
    only_in_b: tuple[str, ...] = ()
    measures: tuple[MeasurePairing, ...] = ()

    def __str__(self) -> str:
        """The reading as a person reads it, ending in the line that refuses to pick a winner."""
        lines = [f"{self.evaluation} at {self.version}",
                 f"  A = {self.system_a}    B = {self.system_b}"]
        if self.only_in_a or self.only_in_b:
            lines.append(f"  unpaired: only in A {self.only_in_a}, only in B {self.only_in_b}")
        for pairing in self.measures:
            lines += ["", str(pairing)]
        return "\n".join([*lines, "", NOT_BETTER])


def _refuse(a: Result, b: Result) -> None:
    for result in (a, b):
        if result.refused:
            raise PairingRefused(
                f"{result.system.name}'s run refused and has no traces: {result.refusal}")
    if a.evaluation.name != b.evaluation.name:
        raise PairingRefused(
            f"these are runs of different evaluations -- {a.evaluation.name!r} and "
            f"{b.evaluation.name!r} -- so pairing them would compare two measurements")
    if a.evaluation.version != b.evaluation.version:
        raise PairingRefused(
            f"both runs are of {a.evaluation.name!r} at different versions -- "
            f"{a.evaluation.version!r} and {b.evaluation.version!r} -- so the tasks are not "
            f"known to be the same tasks")


def _by_task(values: Sequence[Value], measure: str) -> dict[str, float]:
    """One measure's task-scope values, by task id, as floats. `None` values are not paired."""
    return {v.task_id: float(v.value) for v in values
            if v.measure == measure and v.task_id is not None and v.value is not None}


def _groups(result: Result) -> dict[str, str | None]:
    return {trace.task_id: trace.group for trace in result.traces}


def _is_binary(numbers: Sequence[float]) -> bool:
    return all(number in (0.0, 1.0) for number in numbers)


def _binary(measure: str, pairs: list[tuple[str, float, float]], flips: tuple[Flip, ...]
            ) -> MeasurePairing:
    both = sum(1 for _, x, y in pairs if x == 1.0 and y == 1.0)
    neither = sum(1 for _, x, y in pairs if x == 0.0 and y == 0.0)
    only_a = sum(1 for _, x, y in pairs if x == 1.0 and y == 0.0)
    only_b = sum(1 for _, x, y in pairs if x == 0.0 and y == 1.0)
    discordant = only_a + only_b
    mean_a = sum(x for _, x, _ in pairs) / len(pairs)
    mean_b = sum(y for _, _, y in pairs) / len(pairs)
    return MeasurePairing(
        measure=measure, kind="binary", tasks=len(pairs), mean_a=mean_a, mean_b=mean_b,
        gap=mean_b - mean_a, both=both, neither=neither, only_a=only_a, only_b=only_b,
        discordant=discordant, p_value=mcnemar_exact(only_a, only_b), flips=flips,
        caution=TOO_FEW if discordant < FEW_DISCORDANT else None)


def _continuous(measure: str, pairs: list[tuple[str, float, float]], flips: tuple[Flip, ...],
                groups: dict[str, str | None], *, resamples: int, seed: int) -> MeasurePairing:
    mean_a = sum(x for _, x, _ in pairs) / len(pairs)
    mean_b = sum(y for _, _, y in pairs) / len(pairs)
    low, high = bootstrap_ci(pairs, groups, resamples=resamples, seed=seed)
    discordant = len(flips)
    return MeasurePairing(
        measure=measure, kind="continuous", tasks=len(pairs), mean_a=mean_a, mean_b=mean_b,
        gap=mean_b - mean_a, discordant=discordant, ci_low=low, ci_high=high,
        resamples=resamples, seed=seed, flips=flips,
        caution=TOO_FEW if discordant < FEW_DISCORDANT else None)


def _one_measure(measure: str, a: Result, b: Result, groups: dict[str, str | None],
                 *, resamples: int, seed: int) -> MeasurePairing | None:
    left, right = _by_task(a.values, measure), _by_task(b.values, measure)
    shared = [task for task in left if task in right]
    if not shared:
        return None
    pairs = [(task, left[task], right[task]) for task in shared]
    flips = tuple(Flip(task_id=task, was=x, now=y) for task, x, y in pairs if x != y)
    if _is_binary([value for _, x, y in pairs for value in (x, y)]):
        return _binary(measure, pairs, flips)
    return _continuous(measure, pairs, flips, groups, resamples=resamples, seed=seed)


def paired(a: Result, b: Result, *, resamples: int = RESAMPLES, seed: int = SEED) -> Paired:
    """Read two results of the same evaluation side by side, task by task, per measure."""
    _refuse(a, b)
    tasks_a = {t.task_id for t in a.traces}
    tasks_b = {t.task_id for t in b.traces}
    names = [m for m in dict.fromkeys(v.measure for v in a.values)
             if m in {v.measure for v in b.values}]
    groups = {**_groups(a), **_groups(b)}
    readings = [_one_measure(name, a, b, groups, resamples=resamples, seed=seed)
                for name in names]
    return Paired(
        evaluation=a.evaluation.name, version=a.evaluation.version,
        system_a=a.system.name, system_b=b.system.name,
        only_in_a=tuple(sorted(tasks_a - tasks_b)), only_in_b=tuple(sorted(tasks_b - tasks_a)),
        measures=tuple(r for r in readings if r is not None))
