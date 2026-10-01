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
"""Latency instrumentation.

A ``LatencyCollector`` records wall-clock duration for each adapter call
bucketed by a label (typically ``"ingest"`` or ``"retrieve"``) and computes
percentile summaries on flush. The ``track`` context manager is the only
intended write surface -- direct ``record`` access is provided for adapters
whose internal timing differs from wall clock.
"""

from __future__ import annotations

import statistics
import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager


class LatencyCollector:
    """Collect per-bucket latency samples in milliseconds.

    Samples are kept in memory for the duration of a benchmark run. The
    runner calls :meth:`as_metrics` at the end to extract p50/p95/p99 for
    each bucket in the format ``MemoryAdapter.latency_metrics`` requires.
    """

    def __init__(self) -> None:
        self._samples: dict[str, list[float]] = defaultdict(list)

    def record(self, label: str, duration_ms: float) -> None:
        """Record a single latency sample for ``label`` (in milliseconds)."""
        if duration_ms < 0:
            raise ValueError(f"duration_ms must be >= 0, got {duration_ms}")
        self._samples[label].append(float(duration_ms))

    @contextmanager
    def track(self, label: str) -> Iterator[None]:
        """Context manager that records wall-clock time under ``label``."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self.record(label, elapsed_ms)

    def samples(self, label: str) -> list[float]:
        """Return the raw samples for ``label`` (copy)."""
        return list(self._samples.get(label, []))

    def merge(self, other: LatencyCollector) -> None:
        """Absorb another collector's samples (used to combine per-unit collectors
        after a concurrent run). Single-threaded; call once all workers have joined."""
        for label, samples in other._samples.items():
            self._samples[label].extend(samples)

    def labels(self) -> list[str]:
        """Return all labels that have at least one sample."""
        return list(self._samples.keys())

    def reset(self) -> None:
        """Clear all collected samples."""
        self._samples.clear()

    @staticmethod
    def _percentile(samples: list[float], pct: float) -> float:
        """Return the ``pct`` percentile of ``samples`` using linear interpolation.

        Returns 0.0 for empty input. ``pct`` is in [0, 100].
        """
        if not samples:
            return 0.0
        if len(samples) == 1:
            return samples[0]
        ordered = sorted(samples)
        # Linear interpolation between closest ranks.
        rank = (pct / 100.0) * (len(ordered) - 1)
        lo = int(rank)
        hi = min(lo + 1, len(ordered) - 1)
        frac = rank - lo
        return ordered[lo] + (ordered[hi] - ordered[lo]) * frac

    def summary(self, label: str) -> dict[str, float]:
        """Return a percentile summary for one bucket."""
        samples = self._samples.get(label, [])
        if not samples:
            return {
                "p25_ms": 0.0, "p50_ms": 0.0, "p75_ms": 0.0,
                "p95_ms": 0.0, "p99_ms": 0.0, "iqr_ms": 0.0,
                "count": 0, "mean_ms": 0.0,
            }
        p25 = self._percentile(samples, 25)
        p75 = self._percentile(samples, 75)
        return {
            "p25_ms": p25,
            "p50_ms": self._percentile(samples, 50),
            "p75_ms": p75,
            "p95_ms": self._percentile(samples, 95),
            "p99_ms": self._percentile(samples, 99),
            "iqr_ms": p75 - p25,
            "mean_ms": statistics.fmean(samples),
            "count": len(samples),
        }

    def as_metrics(self) -> dict[str, float]:
        """Render samples in the shape ``MemoryAdapter.latency_metrics`` returns.

        Always emits the six required keys; missing buckets report 0.0.
        """
        ingest = self.summary("ingest")
        retrieve = self.summary("retrieve")
        return {
            "ingest_p50_ms": ingest["p50_ms"],
            "ingest_p95_ms": ingest["p95_ms"],
            "ingest_p99_ms": ingest["p99_ms"],
            "retrieve_p50_ms": retrieve["p50_ms"],
            "retrieve_p95_ms": retrieve["p95_ms"],
            "retrieve_p99_ms": retrieve["p99_ms"],
        }
