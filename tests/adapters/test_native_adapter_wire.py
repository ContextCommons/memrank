"""Wire-level tests for the native adapter: what it sends, and what it makes of what comes back.

The behaviours pinned here are the ones whose failure produces a plausible NUMBER rather than an
error -- an unmeasured token count reported as zero, a malformed result list quietly emptied, a
translator error swallowed into "nothing matched". Those are worse than a crash, because a crash
gets investigated.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from memrank.adapters.native import ContractError, Native
from memrank.core import Document
from tests.adapters.test_native_adapter import describe_body, wire


def routed(**bodies: dict[str, Any]) -> Any:
    """A handler answering describe conformingly and each operation with a supplied body."""
    def handler(request: httpx.Request) -> httpx.Response:
        operation = request.url.path.rsplit("/", 1)[-1]
        if operation == "describe":
            return httpx.Response(200, json=describe_body())
        return httpx.Response(200, json=bodies.get(operation, {}))
    return handler


def prepared(**bodies: dict[str, Any]) -> Native:
    """An adapter with one open isolation unit, ready to ingest or retrieve."""
    adapter = wire(Native(base_url="http://translator.test"), routed(**bodies))
    adapter.prepare("unit-1")
    return adapter


def document(doc_id: str = "d1") -> Document:
    return Document(id=doc_id, content="a note", user_id="u1")


# ----------------------------------------------------------------------- #
# Requests
# ----------------------------------------------------------------------- #

def test_documents_are_sent_verbatim():
    """Rendering is the translator's job; the adapter must not reshape on the way out."""
    sent: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json
        if request.url.path.endswith("/describe"):
            return httpx.Response(200, json=describe_body())
        sent.append(json.loads(request.content or b"{}"))
        return httpx.Response(200, json={})

    adapter = wire(Native(base_url="http://translator.test"), handler)
    adapter.prepare("unit-1")
    adapter.ingest([Document(id="d1", content="hi", user_id="u1", messages=[{"role": "user"}])])
    assert sent[-1]["documents"] == [{
        "id": "d1", "content": "hi", "user_id": "u1", "timestamp": None,
        "context": None, "messages": [{"role": "user"}], "metadata": {},
    }]


def test_empty_ingest_makes_no_call():
    adapter = prepared()
    before = adapter.latency.samples("ingest")
    adapter.ingest([])
    assert adapter.latency.samples("ingest") == before


def test_retrieve_sends_the_contract_fields():
    sent: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json
        if request.url.path.endswith("/describe"):
            return httpx.Response(200, json=describe_body())
        sent.append(json.loads(request.content or b"{}"))
        return httpx.Response(200, json={"documents": [], "raw": {}})

    adapter = wire(Native(base_url="http://translator.test"), handler)
    adapter.prepare("unit-1")
    adapter.retrieve("where", k=7, user_id="u1")
    assert sent[-1] == {"query": "where", "k": 7, "user_id": "u1", "query_timestamp": None}


# ----------------------------------------------------------------------- #
# Responses
# ----------------------------------------------------------------------- #

def test_retrieve_rebuilds_documents_in_order():
    body = {"documents": [
        {"id": "a", "content": "first", "metadata": {"score": 3}},
        {"id": "b", "content": "second"},
    ], "raw": {"matched": 2}}
    recall = prepared(retrieve=body).retrieve("q", k=5, user_id="u1")
    documents, raw = recall.documents, recall.declared
    assert [d.id for d in documents] == ["a", "b"]
    assert documents[0].metadata["score"] == 3
    assert documents[1].user_id == "u1"
    assert raw == {"matched": 2}


def test_retrieve_without_a_document_list_is_refused():
    """Not coerced to empty: an empty list is a valid answer meaning 'nothing matched'."""
    with pytest.raises(ContractError, match="ranked 'documents' list"):
        prepared(retrieve={"raw": {}}).retrieve("q", k=5, user_id="u1")


def test_non_object_document_is_refused():
    with pytest.raises(ContractError, match=r"documents\[1\]"):
        prepared(retrieve={"documents": [{"id": "a", "content": "x"}, "oops"]}).retrieve(
            "q", k=5, user_id="u1")


def test_translator_error_is_surfaced_not_swallowed():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/describe"):
            return httpx.Response(200, json=describe_body())
        return httpx.Response(500, json={"error": "engine rejected the batch"})

    adapter = wire(Native(base_url="http://translator.test"), handler)
    with pytest.raises(ContractError, match="engine rejected the batch"):
        adapter.prepare("unit-1")


# ----------------------------------------------------------------------- #
# Usage: absent is not zero
# ----------------------------------------------------------------------- #

def test_omitted_usage_is_unmeasured_not_zero():
    adapter = prepared(retrieve={"documents": [], "raw": {}})
    adapter.ingest([document()])
    adapter.retrieve("q", k=5, user_id="u1")
    assert adapter.token_metrics()["tokens_per_ingest_mean"] is None
    assert adapter.token_metrics()["tokens_per_query_mean"] is None


def test_reported_zero_is_recorded_as_zero():
    """A translator claiming zero is making a measurement; memrank must not conflate that with
    having no measurement at all."""
    adapter = prepared(ingest={"usage": {"total_tokens": 0}})
    adapter.ingest([document()])
    assert adapter.token_metrics()["tokens_per_ingest_mean"] == 0.0


def test_reported_usage_lands_in_the_right_buckets():
    adapter = prepared(ingest={"usage": {"total_tokens": 100}},
                       retrieve={"documents": [], "raw": {}, "usage": {"total_tokens": 20}})
    adapter.ingest([document()])
    adapter.retrieve("q", k=5, user_id="u1")
    assert adapter.token_metrics()["tokens_per_ingest_mean"] == 100.0
    assert adapter.token_metrics()["tokens_per_query_mean"] == 20.0


# ----------------------------------------------------------------------- #
# Latency
# ----------------------------------------------------------------------- #

def test_engine_time_is_reported_beside_wall_clock():
    """Wall-clock includes the translator's own hop; engine_ms is its claim about the engine."""
    adapter = prepared(retrieve={"documents": [], "raw": {}, "engine_ms": 12.5})
    adapter.retrieve("q", k=5, user_id="u1")
    metrics = adapter.latency_metrics()
    assert metrics["retrieve_engine_p50_ms"] == 12.5
    assert metrics["retrieve_p50_ms"] >= 0.0
    assert "ingest_engine_p50_ms" not in metrics


def test_required_latency_keys_survive_the_extra_ones():
    from memrank.core import REQUIRED_LATENCY_KEYS

    adapter = prepared(ingest={"engine_ms": 4.0})
    adapter.ingest([document()])
    assert REQUIRED_LATENCY_KEYS <= set(adapter.latency_metrics())
