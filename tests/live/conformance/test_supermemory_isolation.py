"""Live isolation smoke for Supermemory cleanup (skips without a server).

Seeds a namespace (a /v4/memories seed + a /v3/documents doc), runs cleanup(), and
asserts BOTH resource lists for the container are empty afterward. If the
/v3/documents/bulk delete does not also clear /v4/memories seeds, this FAILS loudly
(see tech-debt.md) rather than leaking across runs. Requires a local Supermemory
server (SUPERMEMORY_BASE_URL / http://localhost:6767)."""
from __future__ import annotations

import pytest

from memrank.core import Document
from tests.live.conformance.test_adapter_contract import _backend_reachable, _backend_url


def test_supermemory_cleanup_empties_namespace():
    url = _backend_url("supermemory")
    if not _backend_reachable(url):
        pytest.skip(f"Supermemory not reachable at {url!r}")
    from memrank.adapters.supermemory import Supermemory
    ns = "memrank-isolation-smoke"
    adapter = Supermemory()
    try:
        adapter.prepare(ns)
        adapter.ingest([
            Document(id="s1", content="Avery prefers blue dashboards.", user_id=ns,
                     metadata={"relation_graph_role": "seed"}),
            Document(id="d1", content="Avery now prefers green dashboards.", user_id=ns,
                     metadata={"task_type": "memory"}),
        ])
        adapter.cleanup()
        client = adapter._http()
        mem = client.post("/v4/memories/list", json={"containerTags": [ns], "limit": 50}).json()
        docs = client.post("/v3/documents/list", json={"containerTags": [ns], "limit": 50}).json()
        assert not (mem.get("memoryEntries") or mem.get("memories") or mem.get("results") or [])
        assert not (docs.get("documents") or docs.get("items") or docs.get("results") or [])
    finally:
        adapter.close()
