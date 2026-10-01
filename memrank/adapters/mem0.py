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
"""Mem0 OSS adapter.

Two execution modes (chosen at runtime, no fallback chains):

1. ``sdk`` -- uses the ``mem0`` Python SDK (``mem0ai`` on PyPI). Requires
   ``pip install memrank[mem0]`` and any LLM/vector-store env vars Mem0
   itself reads.
2. ``http`` -- talks to a local Mem0 OSS server over HTTP at
   ``MEM0_HTTP_URL`` (default ``http://localhost:8888``).

Selection: pass ``mode="sdk"`` or ``mode="http"`` to the constructor. When
omitted, the adapter defaults to ``http`` (the validated path with consistent
search semantics); ``sdk`` is opt-in and not auto-selected, so behavior never
silently changes based on whether ``mem0ai`` happens to be installed.
"""

from __future__ import annotations

import copy
import hashlib
import logging
import os
import uuid
from datetime import datetime
from typing import Any

import httpx

from memrank.adapters import errors as adapter_errors
from memrank.adapters import transcript
from memrank.adapters.effective import embedder_from_env, llm_from_env
from memrank.config import ConfigError
from memrank.core import Document, Memory, Recall
from memrank.instrumentation import LatencyCollector, TokenCollector

_DEFAULT_HTTP_URL = "http://localhost:8888"
_DEFAULT_TIMEOUT_S = 60.0


def _try_import_mem0():
    """Return the ``Memory`` class from the mem0 SDK, or ``None`` if absent."""
    try:
        from mem0 import Memory  # type: ignore[import-not-found]
    except ImportError:
        return None
    return Memory


def _mem0_module():
    """The imported ``mem0`` package, for its ``__version__``. ``None`` when absent."""
    try:
        import mem0  # type: ignore[import-not-found]
    except ImportError:
        return None
    return mem0


class Mem0(Memory):
    """Adapter for Mem0 OSS -- SDK in-process or local HTTP server."""

    name = "mem0"
    version = "0.1.0"
    engine_version = os.environ.get("MEM0_ENGINE_VERSION", "unknown")
    transport = "mode-dependent"  # honest class-level sentinel; instances set the real surface
    cleanup_is_destructive = False

    def __init__(
        self,
        mode: str | None = None,
        base_url: str | None = None,
        timeout_s: float | None = None,
        config: dict[str, Any] | None = None,
        partitioning: dict[str, Any] | None = None,
    ) -> None:
        self.mode = mode or self._select_mode()
        if self.mode not in ("sdk", "http"):
            raise ValueError(f"Mem0 mode must be 'sdk' or 'http', got {self.mode!r}")
        # Report the real integration surface: SDK mode is in-process (no HTTP
        # overhead), so latency is not comparable to http-transport engines.
        self.transport = "sdk" if self.mode == "sdk" else "http"
        # Who destroys the engine's state, which differs entirely by mode.
        #
        # Over HTTP the placement owns it: `compose down -v` takes the datastore volume with the
        # run, so every run starts empty and `cleanup()` need only drop local bookkeeping. An
        # in-process SDK has no container and no placement teardown
        # (`placement/inprocess.py` returns None), so the adapter is the only thing that can --
        # exactly as `word-overlap`/`controls` do for their own in-process stores. Left False, a second
        # run would retrieve the first run's memories and read as an accuracy change.
        self.cleanup_is_destructive = self.mode == "sdk"
        # WHICH mem0 is running, resolved at CONSTRUCTION and not later. The receipt is built in
        # `runner.run_cell` before the first unit is prepared, so a value set when the SDK handle is
        # built -- as this was -- is always correct and always too late, and the run records
        # "unknown". That defeats an arm whose entire premise is 0.1.114 against 2.0.1.
        #
        # SDK only: over HTTP the engine is a container whose tag arrives through
        # MEM0_ENGINE_VERSION, which the class attribute already reads, and importing mem0 locally
        # would say nothing about what the container runs.
        if self.mode == "sdk":
            module = _mem0_module()
            self.engine_version = getattr(module, "__version__", None) or "unknown"
        self.base_url = (base_url or os.environ.get("MEM0_HTTP_URL", _DEFAULT_HTTP_URL)).rstrip("/")
        # Only meaningful over HTTP: an in-process SDK has no address, and naming a variable that
        # moves nothing is worse than saying there is none.
        self.base_url_env = "MEM0_HTTP_URL" if self.mode == "http" else None
        # Per-request HTTP timeout, overridable via MEM0_TIMEOUT_S. Whole-document ingestion of a
        # large benchmark doc (e.g. BEAM) is one synchronous call whose LLM extraction can outrun a
        # small timeout, so the cloud sets this generously.
        self.timeout_s = (
            timeout_s if timeout_s is not None
            else float(os.environ.get("MEM0_TIMEOUT_S", _DEFAULT_TIMEOUT_S))
        )
        self.config = config or {}
        # How memory is scoped. Empty is memrank's matched mode: one partition per benchmark unit,
        # the same scope every other engine gets, which is what makes a cross-engine row mean
        # anything. `{"by": "speaker"}` reproduces mem0's own published eval, which keeps a
        # separate memory per speaker and searches each (audit F13).
        self.partitioning = dict(partitioning or {})
        # Partitions actually written during ingest, so retrieve searches what exists rather than
        # what was predicted. Reset per unit in prepare(); a stale entry here would search another
        # conversation's memories.
        self._partitions: list[str] = []
        self._isolation: str | None = None
        self._memory: Any = None  # SDK handle
        self._client: httpx.Client | None = None
        self._default_user_id = f"memrank_{uuid.uuid4().hex[:8]}"
        self.latency = LatencyCollector()
        self.tokens = TokenCollector()

    @staticmethod
    def _select_mode() -> str:
        # Default to the HTTP server -- the validated path. SDK mode is opt-in via
        # mode="sdk"; its search call now matches the real mem0 API (filters /
        # top_k / threshold), but the SDK's embedder + vector store are its own
        # defaults, not the server's config, so SDK and HTTP numbers are not
        # directly comparable (see tech-debt.md). Defaulting to http also means
        # the mode never silently flips based on whether mem0ai is installed.
        return "http"

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def prepare(self, isolation_unit: str) -> None:
        """Stash the isolation unit and lazily build the underlying client."""
        self._isolation = isolation_unit
        self._partitions = []
        if self.mode == "sdk":
            if self._memory is None:
                self._memory = self._build_sdk_memory()
        else:
            self._http()  # construct early so connection errors surface here

    def cleanup(self) -> None:
        """Drop this unit's state. Destructive in SDK mode, where nothing else would do it.

        Over HTTP this is bookkeeping only -- the placement's `compose down -v` destroys the store.
        In SDK mode there is no placement teardown, so the memories live in a Chroma directory that
        outlives the process unless this removes them.

        Best effort by design: a `cleanup()` that raised would fail a run whose measurements are
        already complete. The per-run collection name is the real isolation guarantee (see
        `_resolved_config`); this keeps the disk from growing.
        """
        if self.mode == "sdk" and self._memory is not None and self._isolation is not None:
            for partition in (self._partitions or [self._isolation]):
                try:
                    self._memory.delete_all(user_id=partition)
                except Exception as exc:  # noqa: BLE001 - reported, never swallowed
                    logging.getLogger(__name__).warning(
                        f"mem0: could not drop memories for {partition!r} after the unit "
                               f"finished ({type(exc).__name__}: {exc}). The run's numbers are "
                               f"unaffected; the store may hold stale memories for a later run.")
        self._isolation = None
        self._partitions = []

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    # ------------------------------------------------------------------ #
    # SDK + HTTP construction
    # ------------------------------------------------------------------ #

    def _build_sdk_memory(self) -> Any:
        Memory = _try_import_mem0()
        if Memory is None:
            from memrank.provenance.install import dependency_instruction

            raise RuntimeError(
                "Mem0 SDK requested but 'mem0ai' is not installed. "
                f"Install with: {dependency_instruction('mem0ai', 'mem0')}"
            )
        if self.config:
            return Memory.from_config(self._with_credentials(self._resolved_config()))
        return Memory()

    def _search_kwargs(self, k: int, uid: str) -> dict[str, Any]:
        """Keyword arguments for ``Memory.search``, spelled the way the installed mem0 expects.

        Two mem0 generations are both reachable through this adapter and they disagree on the
        names, not just the defaults:

        =============  ==============  ==================================
        version        depth           scoping
        =============  ==============  ==================================
        0.1.114        ``limit``       ``user_id=`` (a direct keyword)
        2.0.1          ``top_k``       ``filters={"user_id": ...}``
        =============  ==============  ==================================

        Hardcoding either spelling breaks the other, and it breaks it at the first RETRIEVE --
        after a unit's documents have already been extracted and paid for. Reading the signature
        costs nothing and cannot go stale.

        ``threshold=0.0`` is permissive in both: 2.x defaults to 0.1, which filters real matches
        out, and 0.1.114 defaults to None.
        """
        import inspect

        params = inspect.signature(self._memory.search).parameters
        kwargs: dict[str, Any] = {"threshold": 0.0}
        kwargs["limit" if "limit" in params else "top_k"] = k
        if "user_id" in params:
            kwargs["user_id"] = uid
        else:
            kwargs["filters"] = {"user_id": uid}
        return kwargs

    def _resolved_config(self) -> dict[str, Any]:
        """``self.config`` with ``{run_id}`` substituted, so each run gets its own store.

        Chroma writes to a directory that outlives the process, and nothing in memrank deletes it
        between runs -- over HTTP that isolation came free from `compose down -v`. Without a
        per-run collection, run 2 retrieves run 1's memories and the contamination reads as an
        accuracy change rather than a bug.

        memory-arena solves it the same way, naming their collection
        ``f"{prefix}_{run_id}"``. The placeholder follows the convention `launch.command` already
        uses for ``{port}`` (`manifest._WORKSPACE_PLACEHOLDERS`): a manifest is static, so a value
        only the run knows has to arrive as a substitution.

        Belt and braces with `cleanup()`: a run killed mid-way never reaches teardown, and a fresh
        collection name means the next one is unaffected anyway.
        """
        run_id = self._isolation or self._default_user_id

        def substitute(node: Any) -> Any:
            if isinstance(node, str):
                return node.replace("{run_id}", run_id)
            if isinstance(node, dict):
                return {key: substitute(value) for key, value in node.items()}
            if isinstance(node, list):
                return [substitute(value) for value in node]
            return node

        return substitute(copy.deepcopy(self.config))

    def _with_credentials(self, cfg: dict[str, Any]) -> dict[str, Any]:
        """``cfg`` with each component's ``api_key`` resolved through memrank's chokepoint.

        A manifest states credential NAMES and never values, so an `sdk_config:` block arrives
        without keys. Filling them here rather than letting mem0 read the environment is the
        judge's rule (`judge_client.py:116`) and for its reason: `config.secret` resolves
        environment -> org -> wallet, and a key held only in the WALLET -- which is where
        `memrank secrets set` puts it -- would never reach a library that reads `os.environ`.

        It is also what memory-arena's harness does, passing `api_key` inside the config dict.

        Knowing that mem0 nests credentials at ``<role>.config.api_key`` is this adapter's business:
        the manifest and factory layers pass the block through without reading it, because the
        shape belongs to the library. An adapter for another SDK would know its own.
        """
        from memrank import config as memrank_config
        from memrank.secrets.requirements import PROVIDER_KEY_ENV

        # Deep copy: filling in place would write a live credential into `self.config`, which is
        # the same object a receipt serialises.
        filled = copy.deepcopy(cfg)
        for role in ("llm", "embedder"):
            block = filled.get(role)
            if not isinstance(block, dict):
                continue
            # Keyless providers (transformers, ollama, regex) map to "" and need nothing.
            secret_name = PROVIDER_KEY_ENV.get(block.get("provider") or "")
            if not secret_name:
                continue
            inner = block.setdefault("config", {})
            if inner.get("api_key"):
                continue  # stated explicitly; the manifest wins over anything resolved here
            value = memrank_config.secret(secret_name)
            if value is None:
                raise ConfigError(
                    f"{secret_name} is needed for this target's {role} provider "
                    f"{block.get('provider')!r} but was not found in the environment, the org, or "
                    f"the wallet. Set it with `memrank secrets set {secret_name}`.")
            inner["api_key"] = value
        return filled

    def _http(self) -> httpx.Client:
        if self._client is None:
            self._client = adapter_errors.engine_client(
                engine=self.name, timeout_s=self.timeout_s, env_var="MEM0_TIMEOUT_S",
                base_url_env=self.base_url_env,
                base_url=self.base_url)
        return self._client

    def _user_id_for(self, doc_user_id: str | None) -> str:
        return doc_user_id or self._isolation or self._default_user_id

    #: Ceiling on memories read for a fingerprint. `get_all` defaults to 100 and a LongMemEval unit
    #: is ~50 sessions of extracted facts, so the default would silently compare prefixes and call
    #: two different stores identical. High enough to cover a unit; truncation is REPORTED rather
    #: than ignored, because a fingerprint that might be partial is not evidence of anything.
    _FINGERPRINT_LIMIT = 10_000

    def state_fingerprint(self, scope: str) -> str | None:
        """A digest of everything mem0 holds for ``scope``, so a failed ingest can be classified.

        This is what makes a retry provable rather than hopeful: taken after the last good document
        and again after a failure, an unchanged digest means the failed `add()` wrote nothing, and
        re-running it is exactly equivalent to never having failed.

        Content, not counts. mem0's reconciliation emits UPDATE as well as ADD, and an UPDATE
        rewrites a memory in place -- a count would call that no change at all.

        Sorted before hashing, because result order belongs to the vector store and varies between
        reads of a store nothing has touched.

        Cheap where it is used: in SDK mode this is a local Chroma read -- no API call, no tokens,
        no LLM. Over HTTP it is one request to an endpoint that lists rather than searches.
        """
        entries = self._all_memories(scope)
        if entries is None:
            return None
        items = sorted(
            (str(e.get("id") or e.get("memory_id") or ""),
             str(e.get("memory") or e.get("content") or e.get("text") or ""))
            for e in entries
        )
        if len(items) >= self._FINGERPRINT_LIMIT:
            # Refusing beats guessing: a truncated read would make two different stores hash the
            # same, and the whole point of this value is that a match proves something.
            logging.getLogger(__name__).warning(
                f"mem0: {scope!r} holds at least {self._FINGERPRINT_LIMIT} memories, more "
                       f"than a fingerprint reads -- cannot prove the store is unchanged, so a "
                       f"failed ingest here will not be retried.")
            return None
        digest = hashlib.sha256()
        for memory_id, text in items:
            digest.update(memory_id.encode("utf-8"))
            digest.update(b"\x00")
            digest.update(text.encode("utf-8"))
            digest.update(b"\x00")
        return digest.hexdigest()

    def _all_memories(self, scope: str) -> list[dict[str, Any]] | None:
        """Every memory stored under ``scope``, or None when the engine cannot be asked."""
        if self.mode == "sdk":
            if self._memory is None:
                return None
            result = self._memory.get_all(user_id=scope, limit=self._FINGERPRINT_LIMIT)
        else:
            response = self._http().get("/memories", params={"user_id": scope})
            adapter_errors.raise_for_status(response)
            result = adapter_errors.safe_json(response)
        if isinstance(result, dict):
            entries = result.get("results", result.get("memories", []))
        else:
            entries = result
        return [e for e in (entries or []) if isinstance(e, dict)]

    def describe_engine(self) -> dict[str, Any] | None:
        """Report the components the mem0 server says it is running.

        mem0 is the only benchmarked engine that exposes its live configuration, via
        ``GET /configure``. Reading it is what lets a run record state what actually ran instead of
        what an operator declared. SDK mode has no such endpoint, so it returns ``None``.

        The server nests each component's settings under ``config``; this flattens them into the
        same shape :meth:`effective_config` uses so the two can be compared field by field.
        """
        if self.mode != "http":
            return None
        response = self._http().get("/configure")
        adapter_errors.raise_for_status(response)
        body = adapter_errors.safe_json(response) or {}
        embedder = body.get("embedder") or {}
        embedder_cfg = embedder.get("config") or {}
        llm = body.get("llm") or {}
        llm_cfg = llm.get("config") or {}
        return {
            "llm": {"provider": llm.get("provider"), "model": llm_cfg.get("model")},
            "embedder": {"provider": embedder.get("provider"),
                         "model": embedder_cfg.get("model"),
                         "dims": embedder_cfg.get("embedding_dims")},
        }

    # ------------------------------------------------------------------ #
    # Ingest / retrieve
    # ------------------------------------------------------------------ #

    @staticmethod
    def _document_to_messages(doc: Document) -> list[dict[str, Any]]:
        """Role-tagged turns, with the session date folded into the first one.

        mem0 takes a message list rather than free text, so it cannot use the shared
        transcript renderer directly -- but it still needs to be told WHEN the conversation
        happened, or it dates every memory to ingestion time (audit F15). The header rides
        on the first turn's content rather than as an extra message, so the transcript
        keeps its real turn count and no fabricated speaker appears.
        """
        return Mem0._messages_from(doc, doc.messages)

    @staticmethod
    def _messages_from(doc: Document, turns: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
        """The role/content shape mem0 takes, for ``turns`` of ``doc``.

        Split out so a partitioned ingest can render ONE SPEAKER'S turns and still get the session
        header, which rides on whatever turn comes first in that partition. A partition that lost
        the header would date half the corpus to ingestion time -- F15, reopened for one speaker.
        """
        if not turns:
            return [{"role": "user", "content": transcript.render(doc)}]
        messages = [
            {"role": str(m.get("role", "user")), "content": str(m.get("content", ""))}
            for m in turns
        ]
        head = transcript.header(doc)
        if head:
            messages[0]["content"] = f"{head}\n{messages[0]['content']}"
        return messages

    @staticmethod
    def _by_speaker(doc: Document) -> dict[str, list[dict[str, Any]]]:
        """``{speaker: their turns}``, in first-appearance order.

        A document whose turns carry no ``speaker`` -- LongMemEval, or any benchmark that renders
        prose rather than a dialogue -- yields a single unnamed group, which is the unpartitioned
        path. No speaker is invented: partitioning by a fabricated key would scope memory by
        something the source never said.
        """
        groups: dict[str, list[dict[str, Any]]] = {}
        for message in doc.messages or []:
            groups.setdefault(str(message.get("speaker") or ""), []).append(message)
        return groups

    def _partition_ids(self, doc: Document, uid: str) -> list[tuple[str, list[dict[str, Any]]]]:
        """The ``(user_id, turns)`` pairs this document should be ingested as.

        Matched mode is one pair -- the whole document under ``uid`` -- and is what every engine
        gets. ``partitioning: {by: speaker}`` reproduces mem0's own eval, which keeps a separate
        memory per speaker (audit F13).
        """
        if self.partitioning.get("by") != "speaker":
            return [(uid, doc.messages or [])]
        groups = self._by_speaker(doc)
        if not groups or set(groups) == {""}:
            return [(uid, doc.messages or [])]
        return [(f"{uid}-{speaker}", turns) for speaker, turns in groups.items() if speaker]

    def ingest(self, documents: list[Document]) -> None:
        for doc in documents:
            base_uid = self._user_id_for(doc.user_id)
            for uid, turns in self._partition_ids(doc, base_uid):
                if uid not in self._partitions:
                    self._partitions.append(uid)
                self._ingest_one(doc, uid, turns)

    def _ingest_one(self, doc: Document, uid: str, turns: list[dict[str, Any]]) -> None:
        """One ``add()`` under one partition id."""
        messages = self._messages_from(doc, turns)
        metadata = {"doc_id": doc.id, **(doc.metadata or {})}
        with self.latency.track("ingest"):
            if self.mode == "sdk":
                self._memory.add(messages=messages, user_id=uid, metadata=metadata)
            else:
                payload = {"messages": messages, "user_id": uid, "metadata": metadata}
                response = self._http().post("/memories", json=payload)
                adapter_errors.raise_for_status(response)
                body = adapter_errors.safe_json(response)
                self._record_usage(body, "ingest")

    def retrieve(
        self,
        query: str,
        k: int,
        user_id: str,
        query_timestamp: datetime | str | None = None,
    ) -> Recall:
        """Search every partition this unit wrote, and return everything they gave back.

        Unpartitioned (matched mode) that is one search under one id -- unchanged.

        Partitioned, it is one search per partition at ``top_k=k`` each, concatenated. The merged
        list is deliberately NOT cut to ``k``: mem0's published eval retrieves ~10 per speaker and
        answers from ~20, so trimming here would impose a result count their configuration never
        had -- the same defect as the ``[:k]`` slice memrank used to apply to hindsight (audit F4).
        The run's ``--token-budget`` still bounds what reaches the answerer, which is where a
        budget belongs.
        """
        # Every partition ingest actually WROTE, not "more than one": a conversation with a single
        # speaker writes `run-1-Caroline`, and searching `run-1` because there happened to be only
        # one of them would query a scope nothing was ever written to and score a clean zero.
        partitions = self._partitions if self.partitioning.get("by") == "speaker" else []
        if partitions:
            merged: list[Document] = []
            raws: list[dict[str, Any]] = []
            for partition in partitions:
                one = self._retrieve_one(query, k, partition)
                merged.extend(one.documents)
                raws.append(one.declared)
            return Recall(
                documents=merged,
                declared={"results": [r.get("results", []) for r in raws],
                          "partitions": partitions})
        return self._retrieve_one(query, k, user_id or self._user_id_for(None))

    def _retrieve_one(
        self,
        query: str,
        k: int,
        uid: str,
    ) -> Recall:
        if self.mode == "sdk":
            with self.latency.track("retrieve"):
                results = self._memory.search(query, **self._search_kwargs(k, uid))
            entries = results.get("results", results) if isinstance(results, dict) else results
            raw = results if isinstance(results, dict) else {"results": entries}
        else:
            # The Mem0 OSS server expects `top_k` (not `limit`).
            #
            # `min_similarity` is NOT sent. It used to be, as `0`, to defeat a default distance
            # threshold that filtered out valid matches. Against this build that inverts: EVERY
            # value returns nothing.
            #
            #   omit -> 1 result      0 -> 0      0.0001 -> 0      0.5 -> 0      1 -> 0
            #
            # So the engine ingested correctly and retrieved nothing, on every run memrank has ever
            # made -- scoring exactly like the no-memory control arm, which is what finally exposed
            # it. A parameter this server does not honour is worse than no parameter: it produces a
            # plausible number instead of an error.
            #
            # KNOWN COST: the server's own distance threshold now applies, so a borderline match may
            # be filtered where recall@k wants the top-k regardless. That needs the mem0 search API
            # investigated properly rather than another guessed parameter (tech-debt.md).
            payload = {"query": query, "user_id": uid, "top_k": k}
            with self.latency.track("retrieve"):
                response = self._http().post("/search", json=payload)
                adapter_errors.raise_for_status(response)
            raw = adapter_errors.safe_json(response)
            entries = raw.get("results", []) if isinstance(raw, dict) else raw
            self._record_usage(raw, "query")

        docs: list[Document] = []
        for entry in entries or []:
            entry_id = str(entry.get("id") or entry.get("memory_id") or len(docs))
            content = str(entry.get("memory") or entry.get("content") or entry.get("text") or "")
            metadata: dict[str, Any] = {}
            if entry.get("score") is not None:
                metadata["score"] = entry["score"]
            if entry.get("metadata"):
                metadata.update(entry["metadata"])
            docs.append(Document(id=entry_id, content=content, user_id=uid, metadata=metadata))
        return Recall(documents=docs,
                      declared=raw if isinstance(raw, dict) else {"results": entries})

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _record_usage(self, body: dict[str, Any], bucket: str) -> None:
        usage = (body or {}).get("usage") or {}
        total = int(usage.get("total_tokens", 0))
        if total:
            self.tokens.record(bucket, total)

    # ------------------------------------------------------------------ #
    # Metrics
    # ------------------------------------------------------------------ #

    def latency_metrics(self) -> dict[str, float]:
        return self.latency.as_metrics()

    def token_metrics(self) -> dict[str, float | None]:
        return self.tokens.as_metrics()

    def _effective_llm(self) -> dict:
        return llm_from_env("MEM0_")

    def _effective_embedder(self) -> dict:
        return embedder_from_env("MEM0_")


#: Deprecated alias of the class above -- the same class object, so an out-of-tree import and
#: every ``isinstance`` against the older spelling keep holding. The suffix went because a reader
#: copies the class name out of a first result, and ``Adapter`` is memrank's word for the wrapper
#: rather than the reader's word for the system. Removing it is plan step 22 (ATO-2151).
Mem0Adapter = Mem0
