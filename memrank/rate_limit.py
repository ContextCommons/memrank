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
"""One rate limit pauses EVERY worker, because the quantity over budget is aggregate demand.

A run died on a 429 with the per-worker retry already in place and working -- the log holds ~35
recovered limits. It died because backoff inside one worker's thread cannot reduce the load the
*other* workers are producing: while worker A sleeps, B-E keep calling the provider and hold the
account at its ceiling, so A wakes into the same saturated window it slept on. Ten independent
backoffs are not a throttle; they are ten threads taking turns to fail.

The account limit is shared, so the pause has to be shared too. `RateLimitGate` is that: any worker
that sees a 429 closes the gate, and every worker waits on it before touching the provider again.
Aggregate demand actually drops, the rolling window drains, and oversubscription becomes a slower
run instead of a failed one -- at any ``--workers`` value.

Two details carry the whole design:

  - **Pauses combine with `max()`, never by addition.** A rate limit is account-wide, so all ten
    workers report the same one within the same second. Ten additive pauses would park a healthy
    run for ten windows; one shared deadline parks it for one.
  - **Workers resume at jittered offsets.** Releasing every thread on the same instant re-saturates
    the window immediately, which is the failure the gate exists to prevent, reintroduced at the
    exit.

The clock and sleep are injectable so the tests drive a fake clock: CLAUDE.md forbids
timing-dependent tests, and a gate verified by wall-clock sleeps would be exactly that.
"""

from __future__ import annotations

import random
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

#: A pause longer than the rolling window it is draining serves no purpose, and a malformed header
#: must not park a run for an hour. Matches the OpenAI SDK's own sanity bound on `retry-after`
#: (`_base_client.py`: `if retry_after is not None and 0 < retry_after <= 60`).
MAX_PAUSE_S = 60.0

#: Spread over which workers released by one pause resume. Small -- its job is to decorrelate
#: threads, not to add meaningful delay.
REOPEN_JITTER_S = 0.5


def is_rate_limited(exc: BaseException) -> bool:
    """Whether ``exc`` is a provider rate limit, recognised WITHOUT importing any provider.

    memrank drives engines it does not depend on, so it cannot catch `openai.RateLimitError` by
    type. Duck-typing on the two things every provider agrees about -- a 429 status and a class name
    -- keeps this engine-agnostic.

    Deliberately narrow. A retry loop that widened into a catch-all would hide real defects behind
    silent repetition, which is worse than the failure it was meant to soften.

    Lives here rather than in the runner because three call sites now need it -- ingest, retrieve and
    the judge -- and they draw on ONE account. A predicate re-implemented per site is how one of them
    ends up disagreeing about what a rate limit looks like, which is the failure the gate exists to
    prevent (CLAUDE.md: cross-cutting controls at one chokepoint).
    """
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if status == 429 or str(status) == "429":
        return True
    return type(exc).__name__.endswith("RateLimitError")


def retry_after_seconds(exc: BaseException) -> float | None:
    """How long the provider asked us to wait, read from the response headers.

    The wait is structured data, not prose: providers send `retry-after-ms` (millisecond precision,
    OpenAI's non-standard header) and the standard `retry-after` in whole seconds. Both are on the
    response the exception carries, so this reads them by duck-typing rather than importing any
    provider SDK -- the same constraint that shapes `runner._is_rate_limited`. Parsing the human
    sentence ("Please try again in 1.084s") would be the fragile way to recover a number we are
    handed properly.

    Returns ``None`` whenever the headers are absent or unusable, and never raises: this is an
    improvement over guessing, so a provider that omits it must degrade to backoff rather than
    fail the run.
    """
    headers = getattr(getattr(exc, "response", None), "headers", None)
    get = getattr(headers, "get", None)
    if get is None:
        return None
    milliseconds = _positive_float(get("retry-after-ms"))
    if milliseconds is not None:
        return min(milliseconds / 1000.0, MAX_PAUSE_S)
    seconds = _positive_float(get("retry-after"))
    if seconds is not None:
        return min(seconds, MAX_PAUSE_S)
    return None


def _positive_float(value: Any) -> float | None:
    """Parse a header value, rejecting anything that is not a usable positive number."""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


class RateLimitGate:
    """A pause every worker observes, closed by whichever worker hits the limit first.

    One instance per run. Safe to share across threads; ``wait`` and ``pause`` are the whole
    interface.
    """

    def __init__(self, *, now: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep,
                 jitter: Callable[[], float] = random.random) -> None:
        self._now = now
        self._sleep = sleep
        self._jitter = jitter
        self._lock = threading.Lock()
        self._open_at = 0.0
        self._pauses = 0
        self._paused_seconds = 0.0

    def pause(self, seconds: float) -> float:
        """Close the gate for ``seconds``, and report how much that ADDED to the existing pause.

        Combining by `max()` rather than by addition is what makes this a throttle instead of a
        stall. Every worker sees the same account-wide limit within the same second; if each
        appended its own wait, a single limit would compound into N windows of silence and the run
        would take longer than the failure it replaced.
        """
        with self._lock:
            now = self._now()
            until = now + seconds
            if until <= self._open_at:
                return 0.0                     # already covered by a wider pause
            added = until - max(self._open_at, now)
            self._open_at = until
            self._pauses += 1
            self._paused_seconds += added
            return added

    def wait(self) -> None:
        """Block until the gate is open. Returns immediately when it already is.

        Re-checks after sleeping because another worker may have widened the pause in the meantime
        -- the common case, since a saturated limit fails several workers at once.
        """
        while True:
            with self._lock:
                remaining = self._open_at - self._now()
            if remaining <= 0:
                return
            self._sleep(remaining + self._jitter() * REOPEN_JITTER_S)

    def snapshot(self) -> dict[str, Any]:
        """What the run had to wait through, for the receipt.

        A run that survived by waiting six minutes must be distinguishable from one that sailed
        through: they are not comparable on latency, and nothing else in the artifact would say so.
        """
        with self._lock:
            return {"pauses": self._pauses, "paused_seconds": round(self._paused_seconds, 3)}


class GateScope:
    """The "current cell's gate", as an explicit object instead of the module global.

    A sweep builds its shared judge runtime ONCE (the call cap spans every target), but each
    cell's pauses must land in THAT cell's gate -- and so in that cell's receipt snapshot. The
    scope is the indirection that keeps both true: the judge runtime holds ``scope.get`` and
    the sweep enters ``scope.use(cell_gate)`` around each cell. Purely per-instance state, so
    two scopes never interact and tests need no teardown.
    """

    def __init__(self) -> None:
        #: Open by default: a completer used outside any cell still waits on a real gate,
        #: which is the same promise the module-level default made.
        self._gate = RateLimitGate()

    def get(self) -> RateLimitGate:
        return self._gate

    @contextmanager
    def use(self, gate: RateLimitGate) -> Iterator[None]:
        previous = self._gate
        self._gate = gate
        try:
            yield
        finally:
            self._gate = previous


# The module-global gate (`_GATE`/`set_gate`/`gate`) is gone: `run_cell` takes an explicit
# ``gate`` parameter, and a sweep's shared judge runtime resolves the current cell's gate
# through a `GateScope` instead of ambient state.
