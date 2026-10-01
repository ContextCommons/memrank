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
"""What an engine may say about itself, and what memrank measures for it (ATO-2136).

The engine contract asks for no measurement memrank takes: every ingest and retrieve is timed at
memrank's own call boundary, so an engine neither has to report latency nor can flatter it. Two
things only an engine can know remain, and both are DECLARATIONS rather than requirements -- what
a provider billed it, and its own spend inside the hop memrank timed around it. This module is
where those declarations are collected, pooled across the adapters a ``--workers`` run builds,
and refused where they would overwrite the measurement.
"""

from __future__ import annotations

from memrank.core import REQUIRED_LATENCY_KEYS, MemoryAdapter
from memrank.instrumentation import LatencyCollector, TokenCollector


def pooled_usage(adapter: MemoryAdapter) -> TokenCollector:
    """The per-unit adapter's own usage counter, so declarations pool across concurrent units.

    A run at ``--workers N`` builds N adapters, so the run's usage is the pooled population of
    what each was told it spent -- and a statistic of rendered statistics is not that, which is
    why the COLLECTORS merge rather than the rendered metrics. `TokenCollector` is the shape
    `token_metrics` is rendered from, so every adapter that counts usage already has one under
    ``tokens``.

    An adapter that declares usage through `token_metrics` while keeping no collector cannot have
    its declaration pooled, and is refused rather than quietly reported as unmeasured.

    Raises:
        TypeError: When the adapter overrides `token_metrics` but exposes no `TokenCollector`.
    """
    collector = getattr(adapter, "tokens", None)
    if isinstance(collector, TokenCollector):
        return collector
    if type(adapter).token_metrics is not MemoryAdapter.token_metrics:
        raise TypeError(
            f"{getattr(adapter, 'name', type(adapter).__name__)!r} declares token usage through "
            f"token_metrics() but exposes no TokenCollector as `self.tokens`, so its declaration "
            f"cannot be pooled across the adapters a --workers run builds. Record usage into a "
            f"memrank.instrumentation.TokenCollector, or run at --workers 1.")
    return TokenCollector()


def pooled_declared_latency(adapter: MemoryAdapter) -> LatencyCollector:
    """Whatever this engine says about time only it can see, as samples memrank can pool.

    Raises:
        ValueError: When a declared bucket would render one of memrank's own measured keys. An
            engine reporting the harness's measurement is the thing this contract stopped asking
            for, and resolving the collision by precedence would hide it.
    """
    declared = LatencyCollector()
    for bucket, samples in adapter.declared_latency().items():
        if set(_declared_bucket_keys(bucket)) & REQUIRED_LATENCY_KEYS:
            raise ValueError(
                f"{getattr(adapter, 'name', type(adapter).__name__)!r} declares latency bucket "
                f"{bucket!r}, which renders a key memrank measures itself at its own call "
                f"boundary. Declare only time nobody outside the engine can see, under its own "
                f"bucket name (see MemoryAdapter.declared_latency).")
        for sample in samples:
            declared.record(bucket, sample)
    return declared


def _declared_bucket_keys(bucket: str) -> tuple[str, str]:
    """The two keys a declared bucket contributes. p50 and p95 only -- p99 of a declared
    population memrank never sampled would read as a measurement it made."""
    return f"{bucket}_p50_ms", f"{bucket}_p95_ms"


def latency_report(measured: LatencyCollector,
                   declared: LatencyCollector) -> dict[str, float]:
    """memrank's six measured keys, plus the engine-side ones it was told about.

    Memrank's own measurement is written LAST and so is not overridable, and the collision check
    in `pooled_declared_latency` means there is nothing for it to override. Both together are why
    an engine cannot report the number the leaderboard reads.
    """
    engine_side: dict[str, float] = {}
    for bucket in declared.labels():
        summary = declared.summary(bucket)
        p50_key, p95_key = _declared_bucket_keys(bucket)
        engine_side[p50_key] = summary["p50_ms"]
        engine_side[p95_key] = summary["p95_ms"]
    return {**engine_side, **measured.as_metrics()}
