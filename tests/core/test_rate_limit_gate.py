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
"""One 429 pauses every worker, for as long as the provider asked.

A run at `--workers 5` died on a rate limit with the per-worker retry already working -- the log
holds ~35 recovered limits. Backoff inside one thread cannot lower the demand the other four are
producing, so the sleeper woke into the same saturated window every time.

Every test here drives a FAKE clock. CLAUDE.md forbids timing-dependent tests, and a throttle
verified by real sleeps would be the clearest possible violation of that.
"""

from __future__ import annotations

import pytest

from memrank.rate_limit import MAX_PAUSE_S, RateLimitGate, retry_after_seconds


class _Clock:
    """A clock that only moves when something sleeps."""

    def __init__(self) -> None:
        self.t = 1000.0
        self.slept: list[float] = []

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.t += seconds


def _gate(clock: _Clock, *, jitter: float = 0.0) -> RateLimitGate:
    return RateLimitGate(now=clock.now, sleep=clock.sleep, jitter=lambda: jitter)


def test_a_worker_that_did_not_hit_the_limit_waits_anyway():
    """THE defect this module exists to fix. Worker A takes the 429; worker B never sees one and
    must still stop, because the quota B is spending is what keeps A rate limited."""
    clock = _Clock()
    gate = _gate(clock)

    gate.pause(30.0)          # worker A hit the limit
    gate.wait()               # worker B, which did not

    assert clock.slept == [30.0], "worker B sailed through and kept the window saturated"


def test_an_open_gate_costs_nothing():
    clock = _Clock()

    _gate(clock).wait()

    assert clock.slept == []


def test_simultaneous_pauses_combine_rather_than_stack():
    """A rate limit is account-wide, so all five workers report the SAME one within a second. Five
    additive pauses would park a healthy run for five windows -- slower than the failure it
    replaced."""
    clock = _Clock()
    gate = _gate(clock)

    for _ in range(5):
        gate.pause(30.0)
    gate.wait()

    assert clock.slept == [30.0], "the pauses stacked; five workers now wait 150s for one limit"
    assert gate.snapshot()["paused_seconds"] == pytest.approx(30.0)


def test_a_longer_pause_extends_a_shorter_one():
    """`max()`, not `first wins`: a worker told to wait a minute must not be released after the
    two seconds an earlier worker was told."""
    clock = _Clock()
    gate = _gate(clock)

    gate.pause(2.0)
    gate.pause(50.0)
    gate.wait()

    assert clock.t == pytest.approx(1050.0)


def test_a_pause_widened_while_waiting_is_observed():
    """Workers fail in a cluster, so the gate routinely widens after a thread is already asleep on
    it. Returning at the original deadline would release it into the newer limit."""
    clock = _Clock()
    running: dict[str, RateLimitGate] = {}
    widened = {"done": False}

    def sleep_then_widen(seconds: float) -> None:
        clock.sleep(seconds)
        if not widened["done"]:       # another worker takes a 429 during our sleep
            widened["done"] = True
            running["gate"].pause(20.0)

    gate = RateLimitGate(now=clock.now, sleep=sleep_then_widen, jitter=lambda: 0.0)
    running["gate"] = gate
    gate.pause(10.0)

    gate.wait()

    assert clock.t == pytest.approx(1030.0), "released at the stale deadline"


def test_workers_resume_at_jittered_offsets():
    """Releasing every thread on the same instant re-saturates the window immediately -- the exact
    failure the gate exists to prevent, reintroduced at the exit."""
    clock = _Clock()
    gate = RateLimitGate(now=clock.now, sleep=clock.sleep, jitter=lambda: 1.0)
    gate.pause(10.0)

    gate.wait()

    assert clock.slept == [10.5], "no jitter: all workers resume on the same instant"


class _Headers(dict):
    """httpx.Headers is a case-insensitive mapping; only `.get` is used here."""


def _limit(**headers) -> Exception:
    """A 429 shaped like a provider's: the wait is on the response, not in the message."""
    exc = Exception("rate limit reached")
    exc.response = type("_R", (), {"headers": _Headers(headers)})()
    return exc


def test_the_millisecond_header_is_preferred():
    """`retry-after-ms` is what OpenAI actually sends and it is precise; `retry-after` rounds
    241ms up to a whole second."""
    assert retry_after_seconds(_limit(**{"retry-after-ms": "241",
                                         "retry-after": "1"})) == pytest.approx(0.241)


def test_the_standard_header_is_the_fallback():
    assert retry_after_seconds(_limit(**{"retry-after": "3"})) == pytest.approx(3.0)


def test_an_absurd_header_is_clamped():
    """A malformed or hostile header must not park a run for an hour. Same bound the OpenAI SDK
    applies to its own retries."""
    assert retry_after_seconds(_limit(**{"retry-after": "86400"})) == MAX_PAUSE_S


@pytest.mark.parametrize("exc", [
    Exception("no response at all"),
    _limit(),
    _limit(**{"retry-after": "soon"}),
    _limit(**{"retry-after": "-5"}),
])
def test_an_unusable_header_falls_back_rather_than_raising(exc):
    """This is an improvement over guessing, not a dependency: a provider that omits the header
    must degrade to backoff, never fail the run."""
    assert retry_after_seconds(exc) is None


def test_the_receipt_can_tell_a_paused_run_from_a_clean_one():
    clock = _Clock()
    gate = _gate(clock)

    gate.pause(30.0)
    gate.pause(1.0)          # inside the first pause; adds nothing

    assert gate.snapshot() == {"pauses": 1, "paused_seconds": 30.0}
