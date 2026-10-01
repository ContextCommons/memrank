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
"""AtomicMemory adapter.

Wraps the AtomicMemory HTTP API (``/v1/memories``) so Memrank can benchmark
it like any other engine. Latency and token statistics are recorded by
Memrank's instrumentation collectors -- never measured inside the adapter.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any

import httpx

from memrank.adapters import errors as adapter_errors
from memrank.adapters import transcript
from memrank.adapters.effective import embedder_from_env, llm_from_env
from memrank.core import Document, Memory, Recall
from memrank.instrumentation import LatencyCollector, TokenCollector

_DEFAULT_BASE_URL = "http://localhost:3070"
_DEFAULT_TIMEOUT_S = 60.0


class AtomicMemory(Memory):
    """HTTP adapter for an AtomicMemory backend.

    The base URL defaults to ``http://localhost:3070`` and can be overridden
    via the ``base_url`` constructor argument or the
    ``ATOMICMEMORY_API_URL`` env var.
    """

    name = "atomicmemory"
    version = "0.1.0"
    engine_version = os.environ.get("ATOMICMEMORY_ENGINE_VERSION", "unknown")
    transport = "http"
    cleanup_is_destructive = True
    graph_capable = True
    # Every env var this adapter reads hangs off this prefix, and is read through
    # `type(self)`, so a wire-compatible engine can subclass this adapter -- including from
    # outside the tree -- without inheriting this engine's credentials.
    env_prefix = "ATOMICMEMORY_"
    default_base_url = _DEFAULT_BASE_URL

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout_s: float | None = None,
        source_site: str = "memrank",
    ) -> None:
        prefix = type(self).env_prefix
        self.base_url = (base_url or os.environ.get(f"{prefix}API_URL", type(self).default_base_url)).rstrip("/")
        # Built from the prefix rather than written out, so a wire-compatible subclass that sets
        # its own `env_prefix` has a failure message naming ITS variable and not this engine's.
        self.base_url_env = f"{prefix}API_URL"
        # Via the credential chokepoint (environment -> wallet), so `memrank secrets set
        # ATOMICMEMORY_API_KEY` actually reaches the adapter.
        from memrank.config import secret

        self.api_key = api_key or secret(f"{prefix}API_KEY")
        # AM ingest is extraction-heavy; large conversations can exceed 60s.
        # Allow overriding the per-request timeout via ATOMICMEMORY_TIMEOUT_S.
        self.timeout_s = (
            timeout_s if timeout_s is not None
            else float(os.environ.get(f"{prefix}TIMEOUT_S", _DEFAULT_TIMEOUT_S))
        )
        self.source_site = source_site
        self._isolation: str | None = None
        self._client: httpx.Client | None = None
        self.latency = LatencyCollector()
        self.tokens = TokenCollector()

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def _http(self) -> httpx.Client:
        """Lazily construct a shared httpx.Client (one per adapter instance)."""
        if self._client is None:
            headers: dict[str, str] = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            self._client = adapter_errors.engine_client(
                engine=type(self).name, timeout_s=self.timeout_s,
                env_var=f"{type(self).env_prefix}TIMEOUT_S",
                base_url_env=self.base_url_env,
                base_url=self.base_url, headers=headers)
        return self._client

    def prepare(self, isolation_unit: str) -> None:
        """Tag this isolation unit so all subsequent calls scope to it."""
        self._isolation = isolation_unit
        self._reset_source(isolation_unit)

    def cleanup(self) -> None:
        """Forget the current isolation unit; the HTTP client is reused across units."""
        if self._isolation is not None:
            self._reset_source(self._isolation)
        self._isolation = None

    def close(self) -> None:
        """Close the shared httpx client."""
        if self._client is not None:
            self._client.close()
            self._client = None

    # ------------------------------------------------------------------ #
    # Ingest / retrieve
    # ------------------------------------------------------------------ #

    @staticmethod
    def _document_to_conversation_text(doc: Document) -> str:
        """Convert a Memrank ``Document`` into AM's expected conversation string.

        The core ``/v1/memories/ingest`` endpoint requires ``conversation`` as a
        single string (<=100k chars) and runs fact extraction over it. Rendering lives
        in :mod:`memrank.adapters.transcript` because the AtomicMemory wire contract
        has NO ingest-timestamp field, so the session date can only reach the engine
        inside this string.
        """
        return transcript.render(doc)

    def _user_id_for(self, doc: Document) -> str:
        if doc.user_id:
            return doc.user_id
        if self._isolation:
            return self._isolation
        return "memrank-default"

    def ingest(self, documents: list[Document]) -> None:
        """POST each document to ``/v1/memories/ingest`` under its user_id."""
        client = self._http()
        for doc in documents:
            if (doc.metadata or {}).get("relation_graph_role") == "seed":
                self._ingest_relation_graph_seed(client, doc)
                continue
            payload = {
                "user_id": self._user_id_for(doc),
                "conversation": self._document_to_conversation_text(doc),
                "source_site": self.source_site,
            }
            with self.latency.track("ingest"):
                response = client.post("/v1/memories/ingest", json=payload)
                adapter_errors.raise_for_status(response)
            body = adapter_errors.safe_json(response)
            usage = body.get("usage") or {}
            tokens = int(usage.get("total_tokens", 0))
            if tokens:
                self.tokens.record("ingest", tokens)

    def retrieve(
        self,
        query: str,
        k: int,
        user_id: str,
        query_timestamp: datetime | str | None = None,
    ) -> Recall:
        """POST a search request and shape the response into Documents."""
        client = self._http()
        payload: dict[str, Any] = {"user_id": user_id, "query": query, "limit": k}
        if query_timestamp is not None:
            payload["query_timestamp"] = (
                query_timestamp.isoformat() if isinstance(query_timestamp, datetime) else str(query_timestamp)
            )
        with self.latency.track("retrieve"):
            response = client.post("/v1/memories/search", json=payload)
            adapter_errors.raise_for_status(response)
        body = adapter_errors.safe_json(response)
        usage = body.get("usage") or {}
        tokens = int(usage.get("total_tokens", 0))
        if tokens:
            self.tokens.record("query", tokens)
        results = body.get("results") or body.get("memories") or []
        graph_snapshot = self._graph_snapshot(user_id)
        relation_context = self._relation_context_by_version(graph_snapshot)
        docs = self._shape_search_documents(results, user_id, relation_context)
        return Recall(documents=docs,
                      declared={**body, "graph_snapshot": graph_snapshot})

    @staticmethod
    def _shape_search_documents(
        results: list[dict[str, Any]],
        user_id: str,
        relation_context_by_version: dict[str, list[dict[str, Any]]],
    ) -> list[Document]:
        """Shape raw search results into Documents, attaching relation context by version."""
        docs: list[Document] = []
        for entry in results:
            version_id = entry.get("version_id") or entry.get("current_version_id")
            memory_id = entry.get("id") or entry.get("memory_id")
            doc_id = str(version_id or memory_id or len(docs))
            content = str(entry.get("content") or entry.get("text") or entry.get("memory") or "")
            metadata: dict[str, Any] = {}
            if entry.get("score") is not None:
                metadata["score"] = entry["score"]
            if memory_id is not None:
                metadata["memory_id"] = str(memory_id)
            if version_id is not None:
                metadata["version_id"] = str(version_id)
                metadata["relation_context"] = relation_context_by_version.get(str(version_id), [])
            if entry.get("metadata"):
                metadata.update(entry["metadata"])
            docs.append(Document(id=doc_id, content=content, user_id=user_id, metadata=metadata))
        return docs

    def _ingest_relation_graph_seed(self, client: httpx.Client, doc: Document) -> None:
        """Store seed memories verbatim so fixture setup does not test extraction."""
        payload = {
            "user_id": self._user_id_for(doc),
            "conversation": self._document_to_conversation_text(doc),
            "source_site": self.source_site,
            "skip_extraction": True,
            "metadata": {
                "externalId": doc.id,
                **(doc.metadata or {}),
            },
        }
        with self.latency.track("ingest"):
            response = client.post("/v1/memories/ingest/quick", json=payload)
            adapter_errors.raise_for_status(response)
        body = adapter_errors.safe_json(response)
        usage = body.get("usage") or {}
        tokens = int(usage.get("total_tokens", 0))
        if tokens:
            self.tokens.record("ingest", tokens)

    def _reset_source(self, user_id: str) -> None:
        response = self._http().post(
            "/v1/memories/reset-source",
            json={"user_id": user_id, "source_site": self.source_site},
        )
        adapter_errors.raise_for_status(response)

    def _graph_snapshot(self, user_id: str) -> dict[str, Any]:
        """Normalize AM claim-version provenance into Memrank graph shape.

        AM's public graph-like signal today is claim lineage: version rows carry
        mutation type plus previous_version_id. That is enough to score update
        provenance; extension/derivation provenance remains absent until core
        exposes a richer public relation contract.
        """
        try:
            response = self._http().get(
                "/v1/memories/audit/recent",
                params={"user_id": user_id, "limit": 200},
            )
            adapter_errors.raise_for_status(response)
        # `EngineRejected` is a MemrankError, not an httpx one, so it must be named here or this
        # best-effort snapshot would start failing runs that are otherwise fine. What it buys is
        # the reason: the recorded `error` now carries the engine's own words, not just a status.
        except (httpx.HTTPError, adapter_errors.EngineRejected) as exc:
            return {"memories": [], "error": str(exc)}
        body = adapter_errors.safe_json(response)
        mutations = body.get("mutations")
        if not isinstance(mutations, list):
            return {"memories": [], "error": f"audit body missing a 'mutations' list: got keys {sorted(body)}"}
        return {"memories": [self._normalize_claim_version(m) for m in mutations if isinstance(m, dict)]}

    @staticmethod
    def _normalize_claim_version(entry: dict[str, Any]) -> dict[str, Any]:
        version_id = entry.get("id") or entry.get("version_id")
        mutation_type = entry.get("mutation_type") or entry.get("mutationType")
        previous_version_id = entry.get("previous_version_id") or entry.get("previousVersionId")
        relations = []
        relation = AtomicMemory._mutation_relation(mutation_type)
        if previous_version_id and relation:
            relations.append({
                "parent_provider_memory_id": str(previous_version_id),
                "relation": relation,
            })
        return {
            "provider_memory_id": str(version_id) if version_id is not None else None,
            "memory_id": entry.get("memory_id") or entry.get("memoryId"),
            "claim_id": entry.get("claim_id") or entry.get("claimId"),
            "text": entry.get("content") or entry.get("memory") or entry.get("text") or "",
            "is_latest": entry.get("valid_to") is None and entry.get("superseded_by_version_id") is None,
            "is_forgotten": mutation_type == "delete",
            "is_inference": False,
            "version": version_id,
            "relations": relations,
            "history": [],
            "metadata": {
                "mutation_type": mutation_type,
                "mutation_reason": entry.get("mutation_reason") or entry.get("mutationReason"),
                "previous_version_id": previous_version_id,
                "actor_model": entry.get("actor_model") or entry.get("actorModel"),
                "contradiction_confidence": (
                    entry.get("contradiction_confidence") or entry.get("contradictionConfidence")
                ),
            },
        }

    @staticmethod
    def _mutation_relation(mutation_type: Any) -> str | None:
        if mutation_type in {"update", "supersede", "clarify"}:
            return "updates"
        return None

    @staticmethod
    def _relation_context_by_version(graph_snapshot: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
        memories = graph_snapshot.get("memories") or []
        by_id = {
            str(m.get("provider_memory_id")): m
            for m in memories
            if isinstance(m, dict) and m.get("provider_memory_id") is not None
        }
        out: dict[str, list[dict[str, Any]]] = {}
        for memory in memories:
            if not isinstance(memory, dict) or memory.get("provider_memory_id") is None:
                continue
            contexts = []
            for relation in memory.get("relations") or []:
                parent_id = str(relation.get("parent_provider_memory_id"))
                parent = by_id.get(parent_id, {})
                contexts.append({
                    "provider_memory_id": parent_id,
                    "text": parent.get("text", ""),
                    "relation": relation.get("relation"),
                    "context_group": "parents",
                })
            out[str(memory["provider_memory_id"])] = contexts
        return out

    # ------------------------------------------------------------------ #
    # Metrics
    # ------------------------------------------------------------------ #

    def latency_metrics(self) -> dict[str, float]:
        return self.latency.as_metrics()

    def token_metrics(self) -> dict[str, float | None]:
        return self.tokens.as_metrics()

    def _effective_llm(self) -> dict:
        return llm_from_env(type(self).env_prefix)

    def _effective_embedder(self) -> dict:
        return embedder_from_env(type(self).env_prefix)


#: Deprecated alias of the class above -- the same class object, so an out-of-tree import and
#: every ``isinstance`` against the older spelling keep holding. The suffix went because a reader
#: copies the class name out of a first result, and ``Adapter`` is memrank's word for the wrapper
#: rather than the reader's word for the system. Removing it is plan step 22 (ATO-2151).
AtomicMemoryAdapter = AtomicMemory
