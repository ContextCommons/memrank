"""Each engine retrieves in its own shape; memrank caps tokens, not result counts.

Two shapes were being misstated. Hindsight has no top-k -- it packs recall to a token
budget, and its vendor says so explicitly -- yet the adapter sliced results to ``k``, cutting
the engine off before the shared --token-budget could act. Supermemory inherited an API
default search mode its own vendor does not recommend, so a run measured a configuration
nobody chose (audit F4/F5, F9).
"""

from __future__ import annotations

from typing import Any

from memrank.adapters.hindsight import Hindsight
from memrank.adapters.supermemory import Supermemory


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
    """Records requests; answers every POST with one canned body."""

    def __init__(self, body: dict[str, Any]) -> None:
        self.calls: list[tuple[str, str, dict[str, Any]]] = []
        self._body = body

    def post(self, url: str, json: dict[str, Any]) -> _FakeResponse:  # noqa: A002
        self.calls.append(("POST", url, json))
        return _FakeResponse(self._body)

    def put(self, url: str, json: dict[str, Any]) -> _FakeResponse:  # noqa: A002
        self.calls.append(("PUT", url, json))
        return _FakeResponse({})

    def delete(self, url: str) -> _FakeResponse:
        self.calls.append(("DELETE", url, {}))
        return _FakeResponse({})

    def request(self, method: str, url: str, json: dict[str, Any]) -> _FakeResponse:  # noqa: A002
        self.calls.append((method, url, json))
        return _FakeResponse(self._body)


def _hindsight_returning(n: int) -> tuple[Hindsight, _StubClient]:
    body = {"results": [{"id": f"r{i}", "text": f"memory {i}"} for i in range(n)]}
    a = Hindsight(base_url="http://stub", api_key="k")
    stub = _StubClient(body)
    a._client = stub  # type: ignore[assignment]
    a.prepare("u1")
    return a, stub


def test_hindsight_keeps_every_result_the_engine_returned():
    """The engine already decided how much fits its token budget; k is not its shape."""
    adapter, _ = _hindsight_returning(25)
    docs = adapter.retrieve("when did we meet?", 10, "u1").documents
    assert len(docs) == 25          # not truncated to k=10


def test_hindsight_asks_for_a_token_budget_not_a_result_count():
    adapter, stub = _hindsight_returning(3)
    adapter.retrieve("q", 10, "u1")
    recall = [c for c in stub.calls if c[1].endswith("/memories/recall")][-1][2]
    assert "max_tokens" in recall
    assert "limit" not in recall and "top_k" not in recall and "k" not in recall


def test_supermemory_states_its_search_mode_rather_than_inheriting_one():
    """The API default is "memories"; the vendor recommends and benchmarks "hybrid"."""
    a = Supermemory(base_url="http://stub")
    stub = _StubClient({"results": []})
    a._client = stub  # type: ignore[assignment]
    a._isolation = "iso-1"

    a.retrieve("q", 10, "iso-1")

    search = [c for c in stub.calls if c[1] == "/v4/search"][-1][2]
    assert search["searchMode"] == "hybrid"
    assert search["threshold"] == 0        # let `limit` decide, not a score cutoff
    assert search["limit"] == 10
