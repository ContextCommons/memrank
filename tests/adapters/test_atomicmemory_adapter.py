"""Unit tests for the AtomicMemory adapter's request shapes.

These pin the adapter to the live core `/v1/memories` contract (verified against
a running core 2026-06-04): ingest `conversation` is a STRING (not a turn list),
no `metadata` on `/v1/memories/ingest` (core rejects it there), and search uses
`limit` (not `top_k`). No live backend -- a stub client captures payloads.
"""

from __future__ import annotations

from typing import Any

from memrank.adapters.atomicmemory import AtomicMemory
from memrank.core import Document


class _FakeResponse:
    # A real response carries a status, and the adapters now read it instead of calling a
    # no-op `raise_for_status`. Stating 200 here is what makes these doubles honest: the
    # old fakes could not have represented a refusal at all.
    status_code = 200

    def __init__(self, body: dict[str, Any]) -> None:
        self._body = body

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._body


class _StubClient:
    """Records POSTs; returns a canned body per path."""

    def __init__(self, bodies: dict[str, dict[str, Any]]) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.get_calls: list[tuple[str, dict[str, Any]]] = []
        self._bodies = bodies

    def post(self, url: str, json: dict[str, Any]) -> _FakeResponse:  # noqa: A002
        self.calls.append((url, json))
        return _FakeResponse(self._bodies.get(url, {}))

    def get(self, url: str, params: dict[str, Any]) -> _FakeResponse:
        self.get_calls.append((url, params))
        return _FakeResponse(self._bodies.get(url, {}))


def _adapter_with_stub(bodies: dict[str, dict[str, Any]]) -> tuple[AtomicMemory, _StubClient]:
    a = AtomicMemory(base_url="http://stub", api_key="k")
    bodies = {"/v1/memories/reset-source": {"success": True}, **bodies}
    stub = _StubClient(bodies)
    a._client = stub  # type: ignore[assignment]
    a.prepare("u1")
    stub.calls.clear()
    return a, stub


def test_prepare_resets_source_for_isolation_unit():
    a = AtomicMemory(base_url="http://stub", api_key="k")
    stub = _StubClient({"/v1/memories/reset-source": {"success": True}})
    a._client = stub  # type: ignore[assignment]

    a.prepare("u1")

    assert stub.calls == [
        ("/v1/memories/reset-source", {"user_id": "u1", "source_site": "memrank"})
    ]


def test_conversation_text_joins_messages():
    doc = Document(id="d1", content="raw",
                   messages=[{"role": "user", "content": "hi"},
                             {"role": "assistant", "content": "yo"}])
    text = AtomicMemory._document_to_conversation_text(doc)
    assert isinstance(text, str)
    assert "hi" in text and "yo" in text


def test_conversation_text_falls_back_to_content():
    doc = Document(id="d1", content="just content")
    assert AtomicMemory._document_to_conversation_text(doc) == "just content"


def test_ingest_sends_string_conversation_and_no_metadata():
    a, stub = _adapter_with_stub({"/v1/memories/ingest": {"facts_extracted": 1}})
    a.ingest([Document(id="sess_1", content="I am a marine biologist.",
                       metadata={"doc_id": "sess_1"})])
    url, payload = stub.calls[0]
    assert url == "/v1/memories/ingest"
    assert isinstance(payload["conversation"], str)
    assert payload["conversation"] == "I am a marine biologist."
    assert "metadata" not in payload  # core rejects metadata on /ingest
    assert payload["source_site"] and payload["user_id"]


def test_relation_graph_seed_uses_quick_verbatim_ingest_with_metadata():
    a, stub = _adapter_with_stub({"/v1/memories/ingest/quick": {"stored_memory_ids": ["m1"]}})
    a.ingest([Document(
        id="seed_1",
        content="Avery prefers blue dashboards for planning reviews.",
        metadata={"relation_graph_role": "seed"},
    )])

    url, payload = stub.calls[0]
    assert url == "/v1/memories/ingest/quick"
    assert payload["skip_extraction"] is True
    assert payload["conversation"] == "Avery prefers blue dashboards for planning reviews."
    assert payload["metadata"]["externalId"] == "seed_1"
    assert payload["metadata"]["relation_graph_role"] == "seed"


def test_retrieve_sends_limit_not_top_k_and_parses_memories():
    body = {"memories": [{"id": "m1", "content": "User is a marine biologist.", "score": 0.4}]}
    a, stub = _adapter_with_stub(
        {"/v1/memories/search": body, "/v1/memories/audit/recent": {"mutations": []}}
    )
    recall = a.retrieve("profession?", 7, "u1")
    docs, raw = recall.documents, recall.declared
    url, payload = stub.calls[0]
    assert url == "/v1/memories/search"
    assert payload["limit"] == 7
    assert "top_k" not in payload
    assert docs[0].content == "User is a marine biologist."
    assert docs[0].metadata.get("score") == 0.4
    assert raw["graph_snapshot"] == {"memories": []}
    assert stub.get_calls == [("/v1/memories/audit/recent", {"user_id": "u1", "limit": 200})]


def _claim_update_stub() -> dict[str, dict[str, Any]]:
    """Stub search + audit bodies for a claim-version update (v-old superseded by v-new)."""
    search_body = {
        "memories": [
            {
                "id": "m-new",
                "version_id": "v-new",
                "content": "Avery prefers green dashboards for planning reviews.",
                "score": 0.8,
            }
        ]
    }
    audit_body = {
        "mutations": [
            {
                "id": "v-new",
                "claim_id": "claim-1",
                "memory_id": "m-new",
                "content": "Avery prefers green dashboards for planning reviews.",
                "mutation_type": "update",
                "previous_version_id": "v-old",
                "valid_to": None,
                "superseded_by_version_id": None,
            },
            {
                "id": "v-old",
                "claim_id": "claim-1",
                "memory_id": None,
                "content": "Avery prefers blue dashboards for planning reviews.",
                "mutation_type": "add",
                "previous_version_id": None,
                "valid_to": "2026-06-18T00:00:00.000Z",
                "superseded_by_version_id": "v-new",
            },
        ]
    }
    return {
        "/v1/memories/search": search_body,
        "/v1/memories/audit/recent": audit_body,
    }


def test_retrieve_normalizes_claim_update_provenance_to_graph_context():
    a, _stub = _adapter_with_stub(_claim_update_stub())

    recall = a.retrieve("planning color?", 5, "u1")

    docs, raw = recall.documents, recall.declared

    assert docs[0].id == "v-new"
    assert docs[0].metadata["memory_id"] == "m-new"
    assert docs[0].metadata["relation_context"] == [
        {
            "provider_memory_id": "v-old",
            "text": "Avery prefers blue dashboards for planning reviews.",
            "relation": "updates",
            "context_group": "parents",
        }
    ]
    assert raw["graph_snapshot"]["memories"][0]["relations"] == [
        {"parent_provider_memory_id": "v-old", "relation": "updates"}
    ]


def test_graph_snapshot_flags_error_on_malformed_audit_body():
    a, _stub = _adapter_with_stub({"/v1/memories/audit/recent": {"unexpected": "shape"}})
    snap = a._graph_snapshot("u1")
    assert snap["memories"] == []
    assert snap.get("error")  # malformed -> error, so the benchmark fails loud


def test_timeout_env_override(monkeypatch):
    monkeypatch.delenv("ATOMICMEMORY_TIMEOUT_S", raising=False)
    assert AtomicMemory().timeout_s == 60.0
    monkeypatch.setenv("ATOMICMEMORY_TIMEOUT_S", "600")
    assert AtomicMemory().timeout_s == 600.0
    assert AtomicMemory(timeout_s=5).timeout_s == 5.0  # explicit wins
