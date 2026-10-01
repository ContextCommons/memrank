"""Unit test for the Mem0 HTTP adapter's search payload.

Pins the retrieve payload to what the Mem0 OSS server actually honors (verified
against a running server 2026-06-04): `top_k` (not `limit`) and `min_similarity=0`
so the server's default distance threshold doesn't filter out the top-k results.
"""

from __future__ import annotations

from typing import Any

from memrank.adapters.mem0 import Mem0


class _FakeResponse:
    # A real response carries a status, and the adapters now read it instead of calling a
    # no-op `raise_for_status`. Stating 200 here is what makes these doubles honest: the
    # old fakes could not have represented a refusal at all.
    status_code = 200

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return {"results": [{"id": "m1", "memory": "User is a marine biologist", "score": 0.3}]}


class _StubClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def post(self, url: str, json: dict[str, Any]) -> _FakeResponse:  # noqa: A002
        self.calls.append((url, json))
        return _FakeResponse()


def test_http_search_sends_top_k_and_no_similarity_floor():
    """`min_similarity` must NOT be sent: this mem0 build returns nothing for every value of it.

    The previous version of this test asserted the parameter WAS sent, which locked the bug in --
    it mirrored the adapter's intent and never its effect. Verified against a live engine:

        omit -> 1 result    0 -> 0    0.0001 -> 0    0.5 -> 0    1 -> 0
    """
    a = Mem0(mode="http")
    a._client = _StubClient()  # type: ignore[assignment]
    a.prepare("u1")
    docs = a.retrieve("profession?", 9, "u1").documents
    url, payload = a._client.calls[0]  # type: ignore[attr-defined]
    assert url == "/search"
    assert payload["top_k"] == 9
    assert "min_similarity" not in payload
    assert "limit" not in payload
    assert docs[0].content == "User is a marine biologist"


def test_sdk_search_scopes_user_top_k_and_permissive_threshold():
    """SDK mode must use the real mem0 search API: filters={user_id}, top_k,
    threshold (keyword-only). The old call passed user_id/limit, which the SDK
    silently swallows via **kwargs -> no user scoping, default top_k/threshold."""
    a = Mem0(mode="sdk")
    calls: dict[str, Any] = {}

    class _FakeMem:  # mirrors mem0.Memory.search's real keyword-only signature
        def search(self, query, *, top_k=20, filters=None, threshold=0.1, **kw):
            calls.update(query=query, top_k=top_k, filters=filters, threshold=threshold)
            return {"results": [{"id": "m1", "memory": "green tea", "score": 0.9}]}

    a._memory = _FakeMem()  # type: ignore[assignment]
    a._isolation = "u1"
    docs = a.retrieve("drink?", 7, "u1").documents
    assert calls["filters"] == {"user_id": "u1"}  # scoped to the user
    assert calls["top_k"] == 7
    assert calls["threshold"] == 0.0  # permissive -> return the top-k
    assert docs[0].content == "green tea"


def test_transport_reflects_mode_not_hardcoded():
    """Reports must be honest about the integration surface: SDK mode is in-process
    (no HTTP overhead), so it must not mislabel itself as 'http' in the table."""
    assert Mem0(mode="http").transport == "http"
    assert Mem0(mode="sdk").transport == "sdk"
