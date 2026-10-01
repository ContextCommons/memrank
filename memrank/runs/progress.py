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
"""How far along a run is -- counted in the work it actually does.

WHAT THIS REPLACES. Progress used to be ``queries_done / total_queries``, computed only at unit
boundaries. Two things were wrong with that, and both showed up on one real run
(hindsight x locomo, 272 documents and 4,620 retrievals):

- **Ingest counted for nothing.** At hindsight's measured 5.16 s/document against 0.293 s/retrieval,
  ingest is 51% of the wall clock -- and reported 0%.
- **It moved ten times in forty-six minutes**, once per unit, so it read 0% for the first five.

INGEST AND RETRIEVE INTERLEAVE; JUDGE DOES NOT. A run is ingest->retrieve per unit, ten times
over, not one ingest phase followed by one retrieve phase. So both counters are cumulative across
the whole run and advance alternately; neither ever resets. A bar that filled and emptied ten
times could not answer "how much longer", which is the question this exists for. Judging is the
exception: it runs once, after the last unit, over the queries the judge can grade.

A JUDGED RUN CANNOT BE PROJECTED UNTIL IT JUDGES. Judging is sized upfront like everything else,
but its cost per query -- five LLM calls -- is unknowable until the first one returns, so
:attr:`RunProgress.pct` withholds the overall percentage for the whole of ingest and retrieve
rather than quoting one it would have to revise downwards. The per-stage bars carry real counts,
rates and ETAs throughout, and the overall ETA renders as a lower bound; see :attr:`pct`.

TIME, NOT ITEMS. The overall percentage is work-seconds done over work-seconds projected, at the
rates the run is observing. Counting items instead would put 272 documents beside 4,620 retrievals
and call ingest 6% of the run when it is half of it. Deriving both the percentage and the ETA from
the same projection is also what stops them contradicting each other.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Literal

#: The stages a cell spends its time in, in the order a run reaches them. Ingest and retrieve
#: alternate per unit; judge follows the last of them, once.
STAGES = ("ingest", "retrieve", "judge")


#: Exponential smoothing for the observed rate, tqdm's default. Their docs describe the range as
#: "0 (average speed) to 1 (current/instantaneous speed)"; 0.3 leans recent without chasing noise.
#: It earns its place here rather than a cumulative mean because a run's rate genuinely moves --
#: per-document latency on the mem0 SDK arm is p50 11.3 s against p99 39 s, and workers ramp -- so a
#: cumulative average keeps quoting the first minute's speed an hour later.
_SMOOTHING = 0.3


@dataclass
class _Ema:
    """tqdm's bias-corrected exponential moving average (``tqdm.std.EMA``).

    The correction matters at the start, which is exactly when someone is asking how long this will
    take: without dividing by ``1 - beta**calls`` the first observations are dragged toward the
    zero the accumulator started at, and the first ETA reads far too long.
    """

    alpha: float = _SMOOTHING
    last: float = 0.0
    calls: int = 0

    def observe(self, value: float) -> None:
        beta = 1 - self.alpha
        self.last = self.alpha * value + beta * self.last
        self.calls += 1

    @property
    def value(self) -> float:
        if not self.calls:
            return self.last
        return self.last / (1 - (1 - self.alpha) ** self.calls)


@dataclass
class _Stage:
    """One stage's counters, its work-seconds, and the rate observed so far."""

    total: int = 0
    done: int = 0
    #: SUMMED item latency -- the work-seconds :attr:`RunProgress.pct` accounts in. Additive across
    #: interleaved stages and across concurrent workers, which is what that ratio needs. It is NOT
    #: elapsed time and must never be rendered as a duration (see `_ingest_throughput`).
    seconds: float = 0.0
    #: WALL-CLOCK deltas between this stage's updates, smoothed -- what the rate is derived from.
    _ema_items: _Ema = field(default_factory=_Ema)
    _ema_interval: _Ema = field(default_factory=_Ema)
    _last_at: float | None = None

    def observe(self, items: int, now: float) -> None:
        """Fold one update into the rate, tqdm-style: ``ema(dn) / ema(dt)``.

        Deltas between updates rather than total elapsed, which is what makes this correct under
        ``--workers N`` for free: N workers land N times as many updates in the same wall clock, so
        the intervals shrink and the rate rises to the real throughput. Dividing ``done`` by summed
        latency -- what this used to do -- measured ONE worker and quoted an ETA N times too long.

        It also survives the interleaving: ingest and retrieve alternate per unit, and the long gap
        while the other stage runs arrives as one large interval that the smoothing absorbs rather
        than a permanently depressed average.
        """
        if self._last_at is not None:
            self._ema_items.observe(items)
            self._ema_interval.observe(max(now - self._last_at, 1e-9))
        self._last_at = now

    def resume(self, now: float) -> None:
        """Start (or restart) this stage's clock, so its first item is timed from here.

        Re-seeded on every entry, not just the first: the gap since this stage last ran was spent
        in ANOTHER stage, and charging it here would depress the rate with time this stage never
        used.
        """
        self._last_at = now

    @property
    def rate_per_second(self) -> float | None:
        """Items per second of wall clock, or None until this stage has actually done something.

        None rather than 0.0: a stage that has not started has an UNKNOWN rate, and zero would
        make its ETA infinite rather than unknown.
        """
        interval = self._ema_interval.value
        if not self.done or self._ema_interval.calls == 0 or interval <= 0:
            return None
        return self._ema_items.value / interval

    @property
    def eta_seconds(self) -> float | None:
        """Seconds of work left at the observed rate, or None while the rate is unknown."""
        rate = self.rate_per_second
        if rate is None:
            return None
        return max(0, self.total - self.done) / rate

    @property
    def projected_seconds(self) -> float | None:
        """Total WORK-seconds this stage will cost, or None while unknowable.

        Deliberately not ``seconds + eta_seconds``. Those are different units: ``seconds`` is
        summed item latency and ``eta_seconds`` is wall clock, so under ``--workers N`` adding them
        mixes N workers' work into one worker's remaining time and :attr:`RunProgress.pct` -- a
        ratio of work-seconds -- stops meaning anything.

        Remaining work is priced at this stage's mean item cost instead, which keeps numerator and
        denominator in the same unit. The ratio is then concurrency-agnostic: N workers inflate
        both halves identically, which is exactly why ``pct`` never needed the wall clock.
        """
        if not self.done or self.seconds <= 0:
            return None
        mean_item_seconds = self.seconds / self.done
        return self.seconds + max(0, self.total - self.done) * mean_item_seconds

    def as_dict(self) -> dict[str, Any]:
        return {"done": self.done, "total": self.total,
                "rate_per_second": self.rate_per_second,
                "eta_seconds": self.eta_seconds}


@dataclass
class RunProgress:
    """The live work model for one cell: what is left, how fast it is going, how long that is.

    Fed by the runner as each document and each retrieval completes, and rendered by `watch`,
    `ps` and `runs show` from one stored record so they cannot disagree about what a percentage
    means.
    """

    units: int = 0
    unit: int = 0
    stage: str | None = None
    _stages: dict[str, _Stage] = field(default_factory=lambda: {s: _Stage() for s in STAGES})
    # ``--workers N`` runs whole units concurrently, each on its own adapter, all reporting into
    # one model. ``done += 1`` is not atomic; a lost increment would leave a counter permanently
    # short of a total it can then never reach, and an ETA that never converges.
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    #: Where "now" comes from, injectable exactly as tqdm's ``self._time`` is. The rate is measured
    #: against the wall clock, so a test that wants a rate has to be able to advance one --
    #: fabricating item latency in a tight loop describes a run that finished instantly.
    #: Monotonic rather than wall time: NTP stepping the clock mid-run must not invent a rate.
    _now: Any = field(default=time.monotonic, repr=False)

    def plan(self, *, units: int, documents: int, retrievals: int,
             judgements: int = 0) -> None:
        """Record the work the run is about to do. Known before the first document.

        ``judgements`` counts the queries the judge will actually grade -- not every query.
        A query without a gold answer, or whose category its judge shape does not grade, is
        never judged, and counting it in the denominator would leave the bar stopping short
        of 100% on a complete run. Zero (the default) is an unjudged run, which has no judge
        stage to show.
        """
        self.units = units
        self._stages["ingest"].total = documents
        self._stages["retrieve"].total = retrievals
        self._stages["judge"].total = judgements
        # Start every stage's clock here, the way tqdm stamps ``start_t`` at construction, so the
        # FIRST item already has an interval to be timed against. Without it a stage shows a count
        # but no rate until its second item, and `pct`'s promise -- "the per-stage bars have real
        # counts and a real rate from the first item" -- would be false for exactly the stretch
        # where someone is asking how long this will take. `enter_stage` re-seeds on resume, so a
        # stage that waits its turn is not charged for the wait.
        started = self._now()
        for tracked in self._stages.values():
            tracked.resume(started)

    def record(self, stage: str, *, items: int = 1, seconds: float = 0.0) -> None:
        """Note completed work in one stage, and make it the stage now moving."""
        if stage not in self._stages:
            raise ValueError(f"unknown stage {stage!r}; expected one of {STAGES}")
        with self._lock:
            tracked = self._stages[stage]
            tracked.done += items
            tracked.seconds += seconds
            tracked.observe(items, self._now())
            self.stage = stage

    def enter_unit(self, index: int) -> None:
        """Announce which unit is being worked, 1-based."""
        self.unit = index

    def enter_stage(self, stage: str) -> None:
        """Announce the stage now running, before any of its work has completed.

        Separate from :meth:`record` because a stage becomes current when it STARTS, not when its
        first item lands -- and between those two moments a reader that resolves
        ``record[record["stage"]]`` would otherwise find nothing and show 0/0.
        """
        if stage not in self._stages:
            raise ValueError(f"unknown stage {stage!r}; expected one of {STAGES}")
        with self._lock:
            # Start this stage's clock here, so its first item is timed from when it resumed and
            # not from whenever it last ran -- the interleaving would otherwise charge ingest for
            # the retrievals that happened in between.
            self._stages[stage].resume(self._now())
            self.stage = stage

    @property
    def pct(self) -> int | None:
        """Percent of the run's WORK-SECONDS done, or None until the whole run can be projected.

        None while ANY stage with work left has never run. Projecting over only the stages that
        have a rate looks tempting -- it would put a number on screen during unit 1's ingest -- but
        the number is an over-estimate of a denominator that is about to grow, so it FALLS the
        moment the missing stage starts. On hindsightxlocomo that is 10% dropping to 5%, and a
        progress bar that moves backwards is worse than one that waits.

        The wait is never blank: the per-stage bars have real counts and a real rate from the
        first item, so something is always moving while this holds off. On an unjudged run it is
        also brief -- it ends when unit 1's first retrieval lands. On a JUDGED run it lasts until
        judging starts, because five LLM calls per query cost an amount nothing before them can
        measure. That is the same trade taken deliberately over a wider gap: an overall
        percentage that ran to 90% and then fell to 60% when judging began would be worse than
        one that waited, and the ingest and retrieve bars answer "is it moving" throughout.
        """
        if any(s.rate_per_second is None and s.total > s.done for s in self._stages.values()):
            return None
        done, projected = 0.0, 0.0
        for tracked in self._stages.values():
            total = tracked.projected_seconds
            if total is None:  # a stage with no work at all -- contributes nothing either way
                continue
            done += tracked.seconds
            projected += total
        if projected <= 0:
            return None
        return min(100, int(done * 100 / projected))

    @property
    def eta_seconds(self) -> float | None:
        """Seconds left across both stages, or None while nothing has a rate."""
        etas = [s.eta_seconds for s in self._stages.values() if s.eta_seconds is not None]
        return sum(etas) if etas else None

    @property
    def is_lower_bound(self) -> bool:
        """Whether some stage with work left has never run, so the ETA understates the total.

        True through unit 1's ingest on every run: retrieval has not happened, and reporting a
        confident total there would understate a hindsightxlocomo run by roughly twenty minutes.
        """
        return any(s.rate_per_second is None and s.total > s.done
                   for s in self._stages.values())

    def as_dict(self) -> dict[str, Any]:
        """The record persisted in the heartbeat and shipped to S3."""
        return {
            "stage": self.stage, "unit": self.unit, "units": self.units,
            **{name: tracked.as_dict() for name, tracked in self._stages.items()},
            "pct": self.pct, "eta_seconds": self.eta_seconds,
            "eta_is_lower_bound": self.is_lower_bound,
        }


class StageTimer:
    """Time one item and record it against a stage -- the runner's write surface.

    A context manager so the elapsed time and the completed count can never be recorded
    separately, which is the shape that lets a rate drift away from the work it describes.
    """

    def __init__(self, progress: RunProgress | None, stage: str) -> None:
        self._progress, self._stage, self._start = progress, stage, 0.0

    def __enter__(self) -> StageTimer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> Literal[False]:
        if self._progress is not None:
            self._progress.record(self._stage, seconds=time.perf_counter() - self._start)
        return False
