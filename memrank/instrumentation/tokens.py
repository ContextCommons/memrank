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
"""Token-usage instrumentation.

Adapters call :meth:`TokenCollector.record` after each LLM round-trip with the
input + output token totals. The collector aggregates per-bucket statistics
for ``MemoryAdapter.token_metrics`` to consume.
"""

from __future__ import annotations

import statistics
from collections import defaultdict


class TokenCollector:
    """Aggregate per-call token counts bucketed by label.

    Standard buckets are ``"ingest"`` and ``"query"`` -- these line up with
    ``MemoryAdapter.token_metrics``'s required keys. Adapters may add
    additional buckets (e.g., ``"reflection"``) for richer reporting.
    """

    def __init__(self) -> None:
        self._samples: dict[str, list[int]] = defaultdict(list)

    def record(self, label: str, tokens: int) -> None:
        """Record ``tokens`` (input + output) under ``label``."""
        if tokens < 0:
            raise ValueError(f"tokens must be >= 0, got {tokens}")
        self._samples[label].append(int(tokens))

    def record_call(self, label: str, input_tokens: int, output_tokens: int) -> None:
        """Convenience: record the sum of input + output tokens."""
        self.record(label, int(input_tokens) + int(output_tokens))

    def samples(self, label: str) -> list[int]:
        """Return the raw samples for ``label`` (copy)."""
        return list(self._samples.get(label, []))

    def merge(self, other: TokenCollector) -> None:
        """Absorb another collector's samples (combine per-unit collectors after a
        concurrent run). Single-threaded; call once all workers have joined."""
        for label, samples in other._samples.items():
            self._samples[label].extend(samples)

    def labels(self) -> list[str]:
        """Return all labels that have at least one sample."""
        return list(self._samples.keys())

    def reset(self) -> None:
        """Clear all collected samples."""
        self._samples.clear()

    @staticmethod
    def _percentile(samples: list[int], pct: float) -> float:
        """Return the ``pct`` percentile (0-100) using linear interpolation."""
        if not samples:
            return 0.0
        if len(samples) == 1:
            return float(samples[0])
        ordered = sorted(samples)
        rank = (pct / 100.0) * (len(ordered) - 1)
        lo = int(rank)
        hi = min(lo + 1, len(ordered) - 1)
        frac = rank - lo
        return ordered[lo] + (ordered[hi] - ordered[lo]) * frac

    def summary(self, label: str) -> dict[str, float]:
        """Return mean + p95 for one bucket (always emits both keys)."""
        samples = self._samples.get(label, [])
        if not samples:
            return {"mean": 0.0, "p95": 0.0, "count": 0, "total": 0}
        return {
            "mean": statistics.fmean(samples),
            "p95": self._percentile(samples, 95),
            "count": len(samples),
            "total": sum(samples),
        }

    def as_metrics(self) -> dict[str, float | None]:
        """Render samples in the shape ``MemoryAdapter.token_metrics`` returns.

        Always emits the four required keys. A bucket with no samples reports ``None``, NOT 0.0 --
        most engines return no usage field at all (of five measured, only hindsight reports any),
        and a zero there is a claim that an engine used no tokens rather than an admission that
        nobody counted. Downstream already assumed this: ``compare._engine_tokens`` renders a
        falsy mean as "n/a (engine did not report usage)". The convention was real and unwritten;
        this writes it down at the source, so it does not depend on zero being falsy.
        """
        query = self.summary("query")
        ingest = self.summary("ingest")
        return {
            "tokens_per_query_mean": query["mean"] if query["count"] else None,
            "tokens_per_query_p95": query["p95"] if query["count"] else None,
            "tokens_per_ingest_mean": ingest["mean"] if ingest["count"] else None,
            "tokens_per_ingest_p95": ingest["p95"] if ingest["count"] else None,
        }
