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
"""Tests for latency and token instrumentation."""

from __future__ import annotations

import pytest

from memrank.instrumentation import LatencyCollector, TokenCollector


def test_latency_collector_records_explicit_samples():
    """Direct ``record`` calls accumulate into the right bucket."""
    coll = LatencyCollector()
    coll.record("ingest", 1.0)
    coll.record("ingest", 3.0)
    coll.record("retrieve", 5.0)
    assert coll.samples("ingest") == [1.0, 3.0]
    assert coll.samples("retrieve") == [5.0]


def test_latency_collector_track_context_manager_records_duration():
    """``track`` records a positive elapsed duration."""
    coll = LatencyCollector()
    with coll.track("retrieve"):
        sum(range(1000))  # any work; we only assert duration > 0
    samples = coll.samples("retrieve")
    assert len(samples) == 1
    assert samples[0] >= 0.0


def test_latency_collector_summary_computes_percentiles():
    """Percentiles are deterministic and monotonic across the sorted distribution."""
    coll = LatencyCollector()
    for value in range(1, 101):
        coll.record("ingest", float(value))
    summary = coll.summary("ingest")
    assert summary["count"] == 100
    assert summary["p50_ms"] == pytest.approx(50.5, rel=0, abs=0.01)
    assert summary["p95_ms"] >= summary["p50_ms"]
    assert summary["p99_ms"] >= summary["p95_ms"]


def test_latency_collector_as_metrics_emits_required_keys():
    """``as_metrics`` always exposes the six required adapter keys."""
    coll = LatencyCollector()
    metrics = coll.as_metrics()
    expected = {
        "ingest_p50_ms",
        "ingest_p95_ms",
        "ingest_p99_ms",
        "retrieve_p50_ms",
        "retrieve_p95_ms",
        "retrieve_p99_ms",
    }
    assert set(metrics) == expected
    assert all(value == 0.0 for value in metrics.values())


def test_latency_collector_rejects_negative_duration():
    """Recording a negative duration must fail loudly."""
    coll = LatencyCollector()
    with pytest.raises(ValueError):
        coll.record("ingest", -1.0)


def test_token_collector_record_call_sums_input_and_output():
    """``record_call`` totals the two halves before bucketing."""
    coll = TokenCollector()
    coll.record_call("query", input_tokens=100, output_tokens=50)
    coll.record_call("query", input_tokens=200, output_tokens=100)
    samples = coll.samples("query")
    assert samples == [150, 300]


def test_token_collector_summary_emits_mean_and_p95():
    """Summary always emits both mean and p95."""
    coll = TokenCollector()
    for value in range(1, 21):
        coll.record("query", value)
    summary = coll.summary("query")
    assert summary["count"] == 20
    assert summary["mean"] == pytest.approx(10.5, rel=0, abs=0.01)
    assert summary["p95"] >= summary["mean"]


def test_token_collector_as_metrics_emits_required_keys():
    """``as_metrics`` matches the four adapter contract keys exactly."""
    coll = TokenCollector()
    coll.record("query", 1000)
    coll.record("ingest", 4000)
    metrics = coll.as_metrics()
    assert set(metrics) == {
        "tokens_per_query_mean",
        "tokens_per_query_p95",
        "tokens_per_ingest_mean",
        "tokens_per_ingest_p95",
    }
    assert metrics["tokens_per_query_mean"] == pytest.approx(1000.0)
    assert metrics["tokens_per_ingest_mean"] == pytest.approx(4000.0)


def test_token_collector_rejects_negative_tokens():
    """Negative token counts must raise."""
    coll = TokenCollector()
    with pytest.raises(ValueError):
        coll.record("query", -1)


def test_collector_reset_clears_all_samples():
    """``reset`` flushes every bucket."""
    coll = LatencyCollector()
    coll.record("ingest", 1.0)
    coll.reset()
    assert coll.samples("ingest") == []


def test_latency_summary_includes_p25_p75_and_iqr():
    from memrank.instrumentation.latency import LatencyCollector

    c = LatencyCollector()
    for ms in [10.0, 20.0, 30.0, 40.0, 50.0]:
        c.record("retrieve", ms)
    s = c.summary("retrieve")
    assert s["p25_ms"] == 20.0
    assert s["p75_ms"] == 40.0
    assert s["iqr_ms"] == 20.0
