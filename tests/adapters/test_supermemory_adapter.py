from __future__ import annotations

from typing import Any

import httpx

from memrank.adapters.supermemory import Supermemory
from memrank.core import Document

_RAISE = object()


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
    def __init__(self, bodies: dict[tuple[str, str], Any]) -> None:
        self.calls: list[tuple[str, str, dict[str, Any]]] = []
        self._bodies = bodies

    def _respond(self, method: str, url: str) -> _FakeResponse:
        body = self._bodies.get((method, url), {})
        if body is _RAISE:
            raise httpx.HTTPError("stub graph failure")
        return _FakeResponse(body)

    def post(self, url: str, json: dict[str, Any]) -> _FakeResponse:  # noqa: A002
        self.calls.append(("POST", url, json))
        return self._respond("POST", url)

    def request(self, method: str, url: str, json: dict[str, Any]) -> _FakeResponse:  # noqa: A002
        self.calls.append((method, url, json))
        return self._respond(method, url)


def _adapter_with_stub(
    bodies: dict[tuple[str, str], Any] | None = None,
) -> tuple[Supermemory, _StubClient]:
    adapter = Supermemory(base_url="http://stub")
    stub = _StubClient(bodies or {})
    adapter._client = stub  # type: ignore[assignment]
    return adapter, stub


def test_normalize_memory_maps_relations_history_and_flags():
    raw = {
        "id": "m-new",
        "memory": "Avery prefers green dashboards for planning reviews.",
        "isLatest": True,
        "isForgotten": False,
        "isInference": True,
        "version": 2,
        "memoryRelations": {"m-old": "updates"},
        "history": [
            {
                "id": "m-old",
                "memory": "Avery prefers blue dashboards for planning reviews.",
                "isLatest": False,
                "isForgotten": True,
                "version": 1,
            }
        ],
        "metadata": {"fixture_id": "relation/ambiguous-update/001"},
    }

    normalized = Supermemory._normalize_memory(raw)

    assert normalized["provider_memory_id"] == "m-new"
    assert normalized["text"] == "Avery prefers green dashboards for planning reviews."
    assert normalized["is_inference"] is True
    assert normalized["relations"] == [{"parent_provider_memory_id": "m-old", "relation": "updates"}]
    assert normalized["history"][0]["provider_memory_id"] == "m-old"
    assert normalized["metadata"]["fixture_id"] == "relation/ambiguous-update/001"


def test_ingest_routes_seeds_and_documents_through_v4_memories():
    # Regular documents go through /v4/memories (not /v3/documents), whose content sniffer rejects
    # transcript content -- see the adapter's module note. No readiness poll (v4 is synchronous).
    adapter, stub = _adapter_with_stub({
        ("DELETE", "/v3/documents/bulk"): {"ok": True},
        ("POST", "/v4/memories"): {"ok": True},
    })
    adapter.prepare("iso-1")

    adapter.ingest([
        Document(
            id="seed_1",
            content="Avery prefers blue dashboards for planning reviews.",
            metadata={"relation_graph_role": "seed"},
        ),
        Document(
            id="doc_1",
            content="Avery now prefers green dashboards for planning reviews.",
            context="Dashboard preference memo.",
            metadata={"relation_graph_role": "document", "task_type": "memory"},
        ),
    ])

    calls = stub.calls
    assert calls[0] == ("DELETE", "/v3/documents/bulk", {"containerTags": ["iso-1"]})
    assert calls[1][0:2] == ("POST", "/v4/memories")  # seeds
    assert calls[1][2]["memories"][0]["metadata"]["doc_id"] == "seed_1"
    assert calls[2][0:2] == ("POST", "/v4/memories")  # documents
    doc_mem = calls[2][2]["memories"][0]
    assert doc_mem["metadata"]["doc_id"] == "doc_1"
    assert doc_mem["metadata"]["entity_context"] == "Dashboard preference memo."
    assert doc_mem["content"] == "Avery now prefers green dashboards for planning reviews."
    # only two /v4/memories posts + the prepare cleanup -- no /v3/documents, no readiness poll
    assert [c[1] for c in calls] == ["/v3/documents/bulk", "/v4/memories", "/v4/memories"]


def _relation_search_stub() -> dict[tuple[str, str], dict[str, Any]]:
    """Stub /v4/search + /v4/memories/list for an updates-relation retrieval case."""
    return {
        ("POST", "/v4/search"): {
            "results": [
                {
                    "id": "m-green",
                    "memory": "Avery prefers green dashboards for planning reviews.",
                    "similarity": 0.93,
                    "context": {
                        "parents": [
                            {
                                "id": "m-blue",
                                "memory": "Avery prefers blue dashboards for planning reviews.",
                                "relation": "updates",
                            }
                        ]
                    },
                }
            ]
        },
        ("POST", "/v4/memories/list"): {
            "memoryEntries": [
                {
                    "id": "m-green",
                    "memory": "Avery prefers green dashboards for planning reviews.",
                    "memoryRelations": {"m-blue": "updates"},
                }
            ]
        },
    }


def test_retrieve_returns_relation_context_and_graph_snapshot():
    adapter, _stub = _adapter_with_stub(_relation_search_stub())

    recall = adapter.retrieve("planning color?", k=3, user_id="iso-1")

    docs, raw = recall.documents, recall.declared

    assert docs[0].id == "m-green"
    assert docs[0].metadata["score"] == 0.93
    assert docs[0].metadata["relation_context"] == [
        {
            "provider_memory_id": "m-blue",
            "text": "Avery prefers blue dashboards for planning reviews.",
            "relation": "updates",
            "context_group": "parents",
        }
    ]
    assert raw["graph_snapshot"]["memories"][0]["relations"] == [
        {"parent_provider_memory_id": "m-blue", "relation": "updates"}
    ]


def test_graph_snapshot_failure_is_captured_not_raised():
    adapter, _stub = _adapter_with_stub({("POST", "/v4/memories/list"): _RAISE})
    snap = adapter._graph_snapshot("iso-1")
    assert snap["memories"] == []
    assert snap.get("error")


def test_graph_snapshot_flags_error_on_malformed_list_body():
    adapter, _stub = _adapter_with_stub({("POST", "/v4/memories/list"): {"unexpected": "shape"}})
    snap = adapter._graph_snapshot("iso-1")
    assert snap["memories"] == []
    assert snap.get("error")  # 200 with no memory-list key -> error, not silent empty


def test_graph_snapshot_empty_list_is_valid_not_error():
    adapter, _stub = _adapter_with_stub({("POST", "/v4/memories/list"): {"memoryEntries": []}})
    snap = adapter._graph_snapshot("iso-1")
    assert snap["memories"] == []
    assert "error" not in snap  # present-but-empty key is a legit empty namespace


def test_tokens_recorded_from_search_usage():
    adapter, _stub = _adapter_with_stub({
        ("POST", "/v4/search"): {"results": [], "usage": {"total_tokens": 42}},
        ("POST", "/v4/memories/list"): {"memoryEntries": []},
    })
    adapter._isolation = "iso-1"
    adapter.retrieve("q", k=3, user_id="iso-1")
    metrics = adapter.token_metrics()
    assert metrics["tokens_per_query_mean"] == 42  # the query-tokens-total surfaces 42
