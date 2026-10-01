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
"""Judging concurrently must return exactly what judging sequentially returned.

Judging is the dominant cost of a full run and was strictly sequential: 5.00 LLM calls per judged
query (measured 344/69 and 345/69 on two smoke runs), ~3,015 for a full LoCoMo, projecting 4-8
hours AFTER ingest and retrieval -- in the stage where a run has the most to lose.

The calls are independent, so this parallelises. What makes it safe rather than merely fast is the
shape: classify (pure) -> grade (concurrent) -> reduce (ordered, single-threaded). No accumulator is
shared, so the metrics are bit-identical at any worker count. That equality is the property here; a
faster judge that quietly returns a different number would be worse than a slow one.

See docs/2026-08-12-judge-stage-cost-and-resumability.md.
"""

from __future__ import annotations

import threading

import pytest

from memrank import rate_limit, runner
from memrank.judging import client
from memrank.judging.judge import JudgeConfig
from tests.fakes import JudgeFakeBenchmark, make_fake_completer

WORKER_COUNTS = (1, 2, 4, 8)


@pytest.fixture
def units():
    return JudgeFakeBenchmark().load()


def _rows(units):
    return [{"query_id": q["id"], "retrieved": [{"content": "favorite drink is green tea"}]}
            for unit in units for q in unit.queries]


def _judge(units, *, judge_workers, completer=None):
    return runner._apply_judge(units, _rows(units), JudgeConfig(no_context_control=False),
                               completer or make_fake_completer(),
                               judge_workers=judge_workers)


# ------------------------------------------------------------------ #
# The property: same answer, any width
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("judge_workers", WORKER_COUNTS)
def test_metrics_are_identical_at_every_worker_count(units, judge_workers):
    assert _judge(units, judge_workers=judge_workers) == _judge(units, judge_workers=1)


def test_a_pool_wider_than_the_work_is_harmless(units):
    """`max_workers` is clamped to the item count; 0 items must not construct a 0-width pool."""
    assert _judge(units, judge_workers=64) == _judge(units, judge_workers=1)
    assert runner._grade_all([], JudgeConfig(), make_fake_completer(), "matched",
                             runner.BinaryJudgeShape(runner.GENERIC_BINARY_CATEGORIES), 8) == []


def test_grading_actually_runs_on_several_threads(units):
    """A guard on the guard: if the pool silently collapsed to one thread, the equality tests
    above would still pass and prove nothing about concurrency.

    Two queries against an instant completer will not overlap by luck -- one thread finishes both
    before a second is scheduled. So overlap is REQUIRED here: each thread's first call waits on a
    two-party barrier, which only clears if two threads are genuinely in flight at once. The
    timeout bounds the failure (a single-threaded pool breaks the barrier and the assertion says
    so); it never decides a pass, so nothing here depends on how fast the machine is.
    """
    seen: set[int] = set()
    entered: set[int] = set()
    lock = threading.Lock()
    both_in_flight = threading.Barrier(2, timeout=10)
    inner = make_fake_completer()

    def watching(model, system, user):
        tid = threading.get_ident()
        with lock:
            seen.add(tid)
            first_call_on_this_thread = tid not in entered
            entered.add(tid)
        if first_call_on_this_thread:
            try:
                both_in_flight.wait()
            except threading.BrokenBarrierError:
                pass          # reported by the assertion below, not by an exception from a worker
        return inner(model, system, user)

    _judge(units, judge_workers=4, completer=watching)
    assert len(seen) > 1, "every judge call ran on one thread -- the pool did not fan out"


# ------------------------------------------------------------------ #
# The count: a spend figure that drops increments is worse than none
# ------------------------------------------------------------------ #

def test_the_count_is_exact_under_concurrency():
    """`counting_completer` increments under a lock. A plain read-modify-write loses increments
    when N threads interleave, and `judge_calls_made` would under-report what a run spent.

    This was a CAP test, asserting that a ceiling could not be passed. The ceiling is gone -- it
    bounded the measurement rather than the run -- but the counter outlived it and still has to be
    exact, because it is what the receipt reports.
    """
    calls: list[int] = []
    lock = threading.Lock()
    start = threading.Barrier(8)

    def base(model, system, user):
        start.wait(timeout=5)          # maximise overlap on the increment
        with lock:
            calls.append(1)
        return "ok"

    counted, counter = client.counting_completer(base)
    errors: list[BaseException] = []

    def attempt():
        try:
            counted("m", "s", "u")
        except BaseException as exc:   # noqa: BLE001 - reported, not swallowed
            errors.append(exc)

    threads = [threading.Thread(target=attempt) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert not errors
    assert len(calls) == 8, "every call goes out; nothing is refused"
    assert counter() == 8, "an increment was lost under concurrency"


# ------------------------------------------------------------------ #
# The cache: a partial read is returned as a verdict
# ------------------------------------------------------------------ #

def test_a_concurrent_cache_write_is_never_read_half_written(tmp_path):
    """`cached_completer` returns on `path.exists()`, so an in-place write means a reader can
    catch a truncated response AND GRADE ON IT. Staging then renaming makes the file appear whole
    or not at all -- the same defect class that discarded a 4h50m run on 2026-08-12.
    """
    big = "x" * 200_000                # large enough that a non-atomic write is observably partial
    ready = threading.Barrier(2)

    def base(model, system, user):
        return big

    cached = client.cached_completer(base, cache_dir=tmp_path, prompt_version="v1")
    seen: list[str] = []

    def writer():
        ready.wait(timeout=5)
        cached("m", "s", "u")

    def reader():
        ready.wait(timeout=5)
        for _ in range(400):
            for path in tmp_path.glob("*.txt"):     # only fully-renamed files carry this suffix
                seen.append(path.read_text(encoding="utf-8"))

    threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert all(text == big for text in seen), "a reader observed a partially written cache entry"


# ------------------------------------------------------------------ #
# The gate: judging was the one caller that never waited
# ------------------------------------------------------------------ #

class _Clock:
    """A clock that only moves when something sleeps (CLAUDE.md forbids timing-dependent tests)."""

    def __init__(self) -> None:
        self.t = 1000.0
        self.slept: list[float] = []

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.t += seconds


class _Throttled(Exception):
    """What a provider 429 duck-types as: a status code and a telling class name."""

    status_code = 429


def test_a_rate_limited_judge_call_waits_instead_of_killing_the_run(monkeypatch):
    """Judging never touched the gate. Sequentially that survived; concurrently it would not --
    `_judge_one_query` catches only `UnparseableVerdict`, so a 429 propagates and ends the run in
    its LAST stage, after every ingest and retrieve has already been paid for."""
    clock = _Clock()
    gate = rate_limit.RateLimitGate(now=clock.now, sleep=clock.sleep, jitter=lambda: 0.0)
    attempts: list[int] = []

    def base(model, system, user):
        attempts.append(1)
        if len(attempts) == 1:
            raise _Throttled("slow down")
        return "graded"

    assert client.gated_completer(base, gate=lambda: gate)("m", "s", "u") == "graded"
    assert len(attempts) == 2, "the call was not retried after the rate limit"
    assert clock.slept, "the gate was never paused, so every other worker sailed on"


def test_a_non_rate_limit_error_is_not_retried(monkeypatch):
    """The predicate is deliberately narrow. A retry loop that widened into a catch-all would
    hide real defects behind silent repetition."""
    clock = _Clock()
    gate = rate_limit.RateLimitGate(now=clock.now, sleep=clock.sleep, jitter=lambda: 0.0)
    attempts: list[int] = []

    def base(model, system, user):
        attempts.append(1)
        raise ValueError("malformed request")

    with pytest.raises(ValueError):
        client.gated_completer(base, gate=lambda: gate)("m", "s", "u")
    assert len(attempts) == 1


def test_the_gate_sits_inside_the_cap(monkeypatch):
    """Composition is cache(cap(gate(base))). A 429 is a rejected call that cost nothing, so its
    retries must not each consume a unit of a budget that exists to bound SPEND."""
    clock = _Clock()
    gate = rate_limit.RateLimitGate(now=clock.now, sleep=clock.sleep, jitter=lambda: 0.0)
    attempts: list[int] = []

    def base(model, system, user):
        attempts.append(1)
        if len(attempts) < 3:
            raise _Throttled("slow down")
        return "graded"

    complete, counter = client.build_completer(
        JudgeConfig(completer=base, cache=False), gate=lambda: gate)
    assert complete("m", "s", "u") == "graded"
    assert len(attempts) == 3, "the gate did not absorb the retries"
    assert counter() == 1, "429 retries were counted as spend"


def test_the_cache_still_returns_what_it_stored(tmp_path):
    """The staging rewrite must not change the cache's actual job."""
    calls = []

    def base(model, system, user):
        calls.append(1)
        return "graded"

    cached = client.cached_completer(base, cache_dir=tmp_path, prompt_version="v1")
    assert cached("m", "s", "u") == "graded"
    assert cached("m", "s", "u") == "graded"
    assert len(calls) == 1, "second call should have been served from the cache"
