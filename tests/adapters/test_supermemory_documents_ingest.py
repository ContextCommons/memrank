"""supermemory's selectable ingest mode: /v4/memories as shipped, or /v3/documents with a wait."""

from __future__ import annotations

from typing import Any

import pytest

from memrank.adapters import supermemory as sm
from memrank.adapters.supermemory import DocumentIngestFailed, Supermemory
from memrank.core import Document


class _Response:
    status_code = 200

    def __init__(self, body: dict[str, Any]) -> None:
        self._body = body

    def json(self) -> dict[str, Any]:
        return self._body


class _Engine:
    """Answers the calls the documents path makes; ``statuses`` is what each poll reports."""

    def __init__(self, statuses: list[str], memories: int = 3) -> None:
        self.statuses = list(statuses)
        self.memories = memories
        self.calls: list[tuple[str, str]] = []

    def post(self, url: str, json: dict[str, Any]) -> _Response:  # noqa: A002
        self.calls.append(("POST", url))
        if url == "/v3/documents":
            return _Response({"id": "engine-1", "status": "queued"})
        if url == "/v4/memories/list":
            return _Response({"memoryEntries": [], "pagination": {"totalItems": self.memories}})
        return _Response({})

    def get(self, url: str) -> _Response:
        self.calls.append(("GET", url))
        return _Response({"id": "engine-1", "status": self.statuses.pop(0)})

    def request(self, method: str, url: str, json: dict[str, Any]) -> _Response:  # noqa: A002
        self.calls.append((method, url))
        return _Response({})


def _adapter(engine: _Engine, mode: str | None) -> tuple[Supermemory, list[float]]:
    adapter = Supermemory(base_url="http://stub",
                          ingest=None if mode is None else {"mode": mode})
    adapter._client = engine  # type: ignore[assignment]
    pauses: list[float] = []
    adapter.pause = pauses.append
    adapter.prepare("ns")
    return adapter, pauses


_DOC = Document(id="s1", content="[Conversation (s1)]\nAnn: I adopted a beagle.")


def test_default_mode_is_the_shipped_memories_path() -> None:
    engine = _Engine(statuses=[])
    adapter, _ = _adapter(engine, mode=None)
    adapter.ingest([_DOC])
    assert adapter.ingest_mode == sm.INGEST_MEMORIES
    assert ("POST", "/v4/memories") in engine.calls
    assert ("POST", "/v3/documents") not in engine.calls


def test_documents_mode_posts_to_v3_and_waits_until_done() -> None:
    engine = _Engine(statuses=["queued", "extracting", "embedding", "done"])
    adapter, pauses = _adapter(engine, mode="documents")
    adapter.ingest([_DOC])
    assert ("POST", "/v3/documents") in engine.calls
    assert ("POST", "/v4/memories") not in engine.calls
    assert [c for c in engine.calls if c[0] == "GET"] == [("GET", "/v3/documents/engine-1")] * 4
    # One pause between polls, none after the terminal status.
    assert len(pauses) == 3


def test_a_failed_document_fails_the_ingest() -> None:
    adapter, _ = _adapter(_Engine(statuses=["queued", "failed"]), mode="documents")
    with pytest.raises(DocumentIngestFailed, match="status failed"):
        adapter.ingest([_DOC])


def test_done_with_no_memories_fails_the_ingest() -> None:
    adapter, _ = _adapter(_Engine(statuses=["done"], memories=0), mode="documents")
    with pytest.raises(DocumentIngestFailed, match="no memories"):
        adapter.ingest([_DOC])


def test_a_stalled_document_fails_after_the_poll_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sm, "_MAX_POLLS", 3)
    adapter, _ = _adapter(_Engine(statuses=["queued"] * 3), mode="documents")
    with pytest.raises(DocumentIngestFailed, match="still 'queued' after 3 polls"):
        adapter.ingest([_DOC])


@pytest.mark.parametrize("ingest", [{"mode": "vectors"}, {"mode": "documents", "extra": 1}])
def test_an_ingest_block_it_cannot_honour_is_refused(ingest: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="supermemory"):
        Supermemory(base_url="http://stub", ingest=ingest)

