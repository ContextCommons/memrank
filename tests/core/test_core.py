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
"""Tests for the core ABCs and value types."""

from __future__ import annotations

import pytest

from memrank.core import (
    AdapterResponse,
    Benchmark,
    BenchmarkUnit,
    Document,
    MemoryAdapter,
)


def test_memory_adapter_cannot_instantiate_directly():
    """The MemoryAdapter ABC must reject direct instantiation."""
    with pytest.raises(TypeError):
        MemoryAdapter()  # type: ignore[abstract]


def test_benchmark_cannot_instantiate_directly():
    """The Benchmark ABC must reject direct instantiation."""
    with pytest.raises(TypeError):
        Benchmark()  # type: ignore[abstract]


def test_document_minimal_construction():
    """Document only requires an id and content."""
    doc = Document(id="d1", content="hello")
    assert doc.id == "d1"
    assert doc.content == "hello"
    assert doc.user_id is None
    assert doc.metadata == {}


def test_adapter_response_holds_documents_and_raw():
    """AdapterResponse should accept a documents list and an optional raw dict."""
    docs = [Document(id="d1", content="x")]
    resp = AdapterResponse(query_id="q1", documents=docs, raw={"results": []})
    assert resp.query_id == "q1"
    assert resp.documents == docs
    assert resp.raw == {"results": []}


def test_benchmark_unit_groups_documents_and_queries():
    """BenchmarkUnit binds an isolation id to a doc list and query list."""
    docs = [Document(id="d1", content="x")]
    queries = [{"id": "q1", "text": "?", "user_id": "u1"}]
    unit = BenchmarkUnit(unit_id="u1", isolation_id="u1", documents=docs, queries=queries)
    assert unit.documents == docs
    assert unit.queries == queries


def test_partial_adapter_subclass_must_implement_all_methods():
    """A subclass missing abstract methods cannot be instantiated."""

    class HalfAdapter(MemoryAdapter):
        name = "half"
        version = "0.0.1"

        def prepare(self, isolation_unit):  # noqa: D401
            return None

    with pytest.raises(TypeError):
        HalfAdapter()  # type: ignore[abstract]


def test_complete_adapter_subclass_can_instantiate():
    """A subclass implementing every abstract method should construct cleanly."""

    class CompleteAdapter(MemoryAdapter):
        name = "complete"
        version = "0.0.1"
        engine_version = "0.0.1"

        def prepare(self, isolation_unit):
            self.isolation = isolation_unit

        def ingest(self, documents):
            self.docs = documents

        def retrieve(self, query, k, user_id, query_timestamp=None):
            return [], {}

        def cleanup(self):
            self.docs = []

        def latency_metrics(self):
            return {
                "ingest_p50_ms": 0.0,
                "ingest_p95_ms": 0.0,
                "ingest_p99_ms": 0.0,
                "retrieve_p50_ms": 0.0,
                "retrieve_p95_ms": 0.0,
                "retrieve_p99_ms": 0.0,
            }

        def token_metrics(self):
            return {
                "tokens_per_query_mean": 0.0,
                "tokens_per_query_p95": 0.0,
                "tokens_per_ingest_mean": 0.0,
                "tokens_per_ingest_p95": 0.0,
            }

    adapter = CompleteAdapter()
    adapter.prepare("unit-1")
    adapter.ingest([Document(id="d1", content="x")])
    docs, raw = adapter.retrieve("q", 10, "u1")
    assert docs == [] and raw == {}
    adapter.cleanup()
    assert set(adapter.latency_metrics()) == {
        "ingest_p50_ms",
        "ingest_p95_ms",
        "ingest_p99_ms",
        "retrieve_p50_ms",
        "retrieve_p95_ms",
        "retrieve_p99_ms",
    }
    assert set(adapter.token_metrics()) == {
        "tokens_per_query_mean",
        "tokens_per_query_p95",
        "tokens_per_ingest_mean",
        "tokens_per_ingest_p95",
    }
