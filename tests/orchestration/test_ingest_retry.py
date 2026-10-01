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
"""A rate limit is retried only when the engine can PROVE the failed write did nothing.

A 429 killed a mem0 run at document 286 of 1232 -- eleven minutes and about $0.60 of extraction
discarded, with nothing resumable. The obvious fix is backoff around the call, and it would be
wrong: mem0's `add()` extracts, reconciles, then writes in a loop, so a failure in the third phase
leaves the scope half-written. Retrying then re-extracts against a partially populated store, and
whether that duplicates depends on a reconcile LLM noticing -- not a foundation for a benchmark
number.

`state_fingerprint` turns that assumption into a question the engine answers.

That safety argument survived contact with a real run; the retry BUDGET did not. Five attempts at
1s base is ~20s of waiting against a rolling 60-second TPM window, so the budget expired before the
window could drain, and a run died at document 554/1232. The bound is now wall-clock, and the wait
itself moved to a shared gate -- see `test_rate_limit_gate.py` for why per-worker backoff could not
have worked regardless of how long it waited.
"""

from __future__ import annotations

import pytest

from memrank.core import Document
from memrank.errors import MemrankError
from memrank.rate_limit import RateLimitGate
from memrank.runner import _INGEST_RETRY_DEADLINE_S, RateLimitExhausted, _ingest_one


class _RateLimit(Exception):
    """Shaped like a provider's 429. memrank cannot import `openai` to catch the real one."""

    status_code = 429


class _Engine:
    """An adapter that fails a given number of times, and reports its own state honestly."""

    name = "fake"

    def __init__(self, *, failures: int, writes_on_failure: bool = False,
                 fingerprint: str | None = "empty"):
        self.failures = failures
        self.writes_on_failure = writes_on_failure
        self._fingerprint = fingerprint
        self.ingested: list[str] = []
        self.attempts = 0
        self.retry_after_ms: str | None = None

    def state_fingerprint(self, scope: str) -> str | None:
        return self._fingerprint

    def ingest(self, docs):
        self.attempts += 1
        if self.failures > 0:
            self.failures -= 1
            if self.writes_on_failure:      # a partial write: the store moved before it threw
                self._fingerprint = f"changed-{self.attempts}"
            exc = _RateLimit("rate limit reached")
            if self.retry_after_ms is not None:   # providers state the wait in a header
                exc.response = type("_R", (), {"headers": {"retry-after-ms": self.retry_after_ms}})()
            raise exc
        self.ingested.extend(d.id for d in docs)


class _FakeClock:
    """Time advances only when the gate sleeps, so the deadline is exercised without waiting."""

    def __init__(self) -> None:
        self.t = 0.0

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += seconds


@pytest.fixture
def clock(monkeypatch):
    """One clock behind both the gate's sleeps and the runner's deadline check."""
    fake = _FakeClock()
    monkeypatch.setattr("memrank.runner.time.monotonic", fake.now)
    return fake


@pytest.fixture
def gate(clock):
    return RateLimitGate(now=clock.now, sleep=clock.sleep, jitter=lambda: 0.0)


def _doc():
    return Document(id="d1", content="hello")


def test_a_transient_rate_limit_is_retried_when_the_store_is_unchanged(gate):
    """The case worth surviving: the 429 landed on the extraction LLM, before any write, so a
    retry is exactly equivalent to never having failed."""
    engine = _Engine(failures=2)

    _ingest_one(engine, _doc(), scope="unit-1", label="[fake]", position="49/50", gate=gate)

    assert engine.ingested == ["d1"]
    assert engine.attempts == 3


def test_a_partial_write_is_never_retried(gate):
    """The defect this exists to prevent. The store changed before the failure, so retrying would
    ingest the document twice and no later number could be trusted."""
    engine = _Engine(failures=1, writes_on_failure=True)

    with pytest.raises(RuntimeError) as excinfo:
        _ingest_one(engine, _doc(), scope="unit-1", label="[fake]", position="49/50", gate=gate)

    assert "PART-WAY THROUGH" in str(excinfo.value)
    assert engine.ingested == [], "the document must not have been ingested twice"


def test_an_engine_that_cannot_report_its_state_is_not_retried(gate):
    """`None` means "cannot be asked", which is CANNOT PROVE SAFE -- not permission."""
    engine = _Engine(failures=1, fingerprint=None)

    with pytest.raises(RuntimeError) as excinfo:
        _ingest_one(engine, _doc(), scope="unit-1", label="[fake]", position="1/50", gate=gate)

    assert "cannot report" in str(excinfo.value)


def test_a_non_rate_limit_error_is_raised_immediately(gate):
    """A retry loop that widened into a catch-all would hide real defects behind repetition."""

    class _Broken(_Engine):
        def ingest(self, docs):
            self.attempts += 1
            raise ValueError("a real bug")

    engine = _Broken(failures=0)

    with pytest.raises(ValueError):
        _ingest_one(engine, _doc(), scope="unit-1", label="[fake]", position="1/50", gate=gate)

    assert engine.attempts == 1, "it must not have been retried"


def test_the_run_that_died_at_five_attempts_now_survives(gate):
    """The regression, stated as the run that prompted it. Four units reached attempt 4/5 and one
    took a fifth 429; nothing was wrong with them that another minute would not have fixed."""
    engine = _Engine(failures=8)

    _ingest_one(engine, _doc(), scope="unit-1", label="[fake]", position="18/44", gate=gate)

    assert engine.ingested == ["d1"]
    assert engine.attempts == 9, "the old count-based budget stopped at 5"


def test_retries_are_bounded_by_the_clock(gate, clock):
    """A run that cannot proceed fails; it does not spin. Some conditions -- a revoked key, zero
    quota -- no amount of waiting fixes, and CLAUDE.md's "no degraded modes" means stopping rather
    than looking healthy forever."""
    engine = _Engine(failures=10_000)

    with pytest.raises(RateLimitExhausted) as excinfo:
        _ingest_one(engine, _doc(), scope="unit-1", label="[fake]", position="1/50", gate=gate)

    assert clock.t >= _INGEST_RETRY_DEADLINE_S
    assert "lower --workers" in str(excinfo.value)


def test_a_pause_is_not_counted_as_engine_latency(gate, clock):
    """Waiting out a quota is the RUNNER's choice, not the engine's response time. Wrapping the
    retry loop in the timer -- which is how both ingest and retrieve were measured -- put a
    30-second pause straight into `ingest p95`, so a throttled account read as a slow engine and
    every cross-engine latency comparison in that run was quietly wrong.
    """
    engine = _Engine(failures=3)

    elapsed = _ingest_one(engine, _doc(), scope="unit-1", label="[fake]", position="1/50",
                          gate=gate)

    assert clock.t > 0, "the gate did pause, so there is something to exclude"
    assert elapsed < 1.0, "the pause landed in the latency sample"


def test_exhaustion_is_not_reported_as_a_bug_in_memrank():
    """`_exit_with` prints "this is a bug in memrank" for anything that is not a `MemrankError`,
    and the run this fixes did exactly that -- sending the reader after a defect that did not
    exist. An exhausted provider quota is a condition of the environment."""
    assert issubclass(RateLimitExhausted, MemrankError)


def test_the_wait_comes_from_the_provider_not_from_a_guess(gate, clock):
    """`retry-after-ms` says 241ms; the old backoff slept 1.5s for it, and a whole minute's limit
    got the same 1.5s. Both directions are wrong and the header was there the entire time."""
    engine = _Engine(failures=1)
    engine.retry_after_ms = "250"

    _ingest_one(engine, _doc(), scope="unit-1", label="[fake]", position="1/50", gate=gate)

    assert clock.t == pytest.approx(0.25)
