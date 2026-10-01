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
"""Client for memrank's own translator contract (``docs/system-contract.md``).

Every other adapter in this package wraps ONE engine's API. This one wraps no engine at all: it
speaks a contract memrank publishes, and whoever is on the other end forwards to their own system
however they like. That is what lets memrank evaluate an engine it has never seen without a fork --
the vendor-specific part lives in their process, in their language, and memrank never imports it.

Two things are deliberately inverted relative to the built-in adapters:

- **Components are reported, not injected.** memrank configures a built-in engine through env vars
  it knows by name (``memrank/targets/engine_env.py``) and the manifest asserts what was set. It
  cannot do that for a stranger's engine, so the translator states its own configuration in
  ``describe`` and memrank records that -- which is why a native run is ``verified: "engine"``
  rather than ``"declared"``.
- **Transport is ``translator``.** A translator is an extra process and an extra hop, so its
  wall-clock latency is not comparable with an engine driven directly. Declaring a distinct
  transport class is what keeps ``docs/methodology.md``'s comparability rule doing that work
  instead of a reader having to know.
"""

from __future__ import annotations

import os
from dataclasses import asdict
from datetime import datetime
from typing import Any

import httpx

from memrank.adapters import errors as adapter_errors
from memrank.core import Document, Memory, Recall
from memrank.docs import doc_url
from memrank.errors import MemrankError
from memrank.instrumentation import LatencyCollector, TokenCollector

_DEFAULT_BASE_URL = "http://localhost:8099"
_DEFAULT_TIMEOUT_S = 900.0
_PREFIX = "/memrank/v1"

#: The contract revision this client implements. A translator announcing anything else is refused
#: rather than probed for compatibility: a partially-understood contract produces numbers whose
#: meaning nobody can state.
CONTRACT_VERSION = "v1"

_SECTIONS = ("adapter", "engine", "components", "capabilities")
_ROLES = ("llm", "embedder")
_CAPABILITIES = ("graph_snapshot", "context_budget")
_CONTRACT_DOC = doc_url("system-contract.md")


class ContractError(MemrankError):
    """A translator violated the system contract."""


def _error_message(response: httpx.Response) -> str:
    """The translator's own error text, per contract section 8, or the raw body.

    Defers to ``adapter_errors.body_excerpt``, which reads the same section-8 ``error`` field and
    applies the one redaction-and-cap policy every adapter answers to. The hand-rolled copy this
    replaces left the ``error`` field entirely uncapped and unredacted -- a translator is closer to
    us than a vendor engine, but it is still a separate process quoting a request we sent it.
    """
    return adapter_errors.body_excerpt(response)


def _json_object(response: httpx.Response, operation: str) -> dict[str, Any]:
    """Parse a contract response, which is always a JSON object."""
    try:
        body = response.json()
    except ValueError as exc:
        raise ContractError(
            f"{operation} did not return JSON: {response.text.strip()[:200]!r}. "
            f"Every contract response is a JSON object; see {_CONTRACT_DOC}.") from exc
    if not isinstance(body, dict):
        raise ContractError(
            f"{operation} returned {type(body).__name__}, not a JSON object; see {_CONTRACT_DOC}.")
    return body


def _iso(value: datetime | str | None) -> str | None:
    """Render a query timestamp for the wire, preserving None."""
    if value is None:
        return None
    return value.isoformat() if isinstance(value, datetime) else str(value)


def _to_document(entry: Any, index: int, user_id: str) -> Document:
    """Rebuild one returned Document, refusing a shape the scorer could not read."""
    if not isinstance(entry, dict):
        raise ContractError(
            f"retrieve returned {type(entry).__name__} at documents[{index}]; every entry must be "
            f"a Document object; see {_CONTRACT_DOC} section 3.")
    entry_id = entry.get("id")
    return Document(
        id=str(entry_id) if entry_id is not None else str(index),
        content=str(entry.get("content") or ""),
        user_id=entry.get("user_id") or user_id,
        timestamp=entry.get("timestamp"),
        context=entry.get("context"),
        messages=entry.get("messages"),
        metadata=dict(entry.get("metadata") or {}),
    )


def _require_sections(body: Any) -> dict[str, Any]:
    """Check describe's top-level shape and contract version."""
    if not isinstance(body, dict):
        raise ContractError(
            f"describe returned {type(body).__name__}, not an object; see {_CONTRACT_DOC}.")
    version = body.get("contract_version")
    if version != CONTRACT_VERSION:
        raise ContractError(
            f"translator announces contract_version {version!r}; this memrank implements "
            f"{CONTRACT_VERSION!r}. Upgrade one side rather than running a contract neither "
            f"fully implements.")
    for section in _SECTIONS:
        if not isinstance(body.get(section), dict):
            raise ContractError(
                f"describe is missing the {section!r} object; see {_CONTRACT_DOC} section 4.")
    return body


def _require_components(components: dict[str, Any]) -> None:
    """Every role must be stated -- as an object, or as an explicit null."""
    for role in _ROLES:
        if role not in components:
            raise ContractError(
                f"describe does not state components.{role}. Report the configuration, or null if "
                f"the engine has no {role} at all -- silence is not the same claim, and a blank "
                f"knob is what lets an engine measure a configuration nobody chose.")
        value = components[role]
        if value is not None and not isinstance(value, dict):
            raise ContractError(
                f"components.{role} must be an object or null, got {type(value).__name__}.")


def _require_capabilities(capabilities: dict[str, Any]) -> None:
    """Capabilities are announced, but the fairness control is not negotiable."""
    for capability in _CAPABILITIES:
        if capability not in capabilities:
            raise ContractError(
                f"describe does not state capabilities.{capability}; see {_CONTRACT_DOC} "
                f"section 4.")
    budget = capabilities["context_budget"]
    if budget != "matched":
        raise ContractError(
            f"capabilities.context_budget is {budget!r}; a translator must be 'matched'. The shared "
            f"--token-budget is the fairness control every target is held to, so a row cannot win "
            f"by returning more text. 'uncapped' and 'none' exist only for memrank's own baseline "
            f"arms.")


class Native(Memory):
    """Drives any translator implementing the memrank system contract.

    The base URL comes from the ``base_url`` argument or ``NATIVE_API_URL``, which the placement
    sets to the address it launched the translator on.
    """

    name = "native"
    base_url_env = "NATIVE_API_URL"
    version = "0.1.0"
    engine_version = "unknown"
    transport = "translator"
    # A translator owns its engine's state and is told to release it per unit; cleanup is a real
    # teardown rather than a local forget.
    cleanup_is_destructive = True
    # Replaced per instance from describe's capability block -- a translator announces this rather
    # than memrank assuming it.
    graph_capable = False

    def __init__(self, base_url: str | None = None, timeout_s: float | None = None) -> None:
        self.base_url = (
            base_url or os.environ.get("NATIVE_API_URL", _DEFAULT_BASE_URL)).rstrip("/")
        self.timeout_s = (
            timeout_s if timeout_s is not None
            else float(os.environ.get("NATIVE_TIMEOUT_S", _DEFAULT_TIMEOUT_S))
        )
        self._isolation: str | None = None
        self._client: httpx.Client | None = None
        self._description: dict[str, Any] | None = None
        self.latency = LatencyCollector()
        self.tokens = TokenCollector()

    # ------------------------------------------------------------------ #
    # Transport
    # ------------------------------------------------------------------ #

    def _http(self) -> httpx.Client:
        if self._client is None:
            self._client = adapter_errors.engine_client(
                engine=self.name, timeout_s=self.timeout_s, env_var="NATIVE_TIMEOUT_S",
                base_url_env=self.base_url_env,
                base_url=self.base_url,
                headers={"Content-Type": "application/json"},
            )
        return self._client

    def _call(self, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        """POST one contract operation, surfacing the translator's own error text."""
        response = self._http().post(f"{_PREFIX}/{operation}", json=payload)
        if response.status_code >= 400:
            raise ContractError(
                f"{operation} failed ({response.status_code}): {_error_message(response)}")
        return _json_object(response, operation)

    def close(self) -> None:
        """Close the shared httpx client."""
        if self._client is not None:
            self._client.close()
            self._client = None

    # ------------------------------------------------------------------ #
    # Description
    # ------------------------------------------------------------------ #

    def _describe(self) -> dict[str, Any]:
        """Fetch, validate and cache the translator's self-description.

        Cached because it is identity, not state: it cannot change mid-run without invalidating
        every measurement taken before it did.
        """
        if self._description is not None:
            return self._description
        response = self._http().get(f"{_PREFIX}/describe")
        if response.status_code >= 400:
            raise ContractError(
                f"describe failed ({response.status_code}): {_error_message(response)}")
        body = _require_sections(_json_object(response, "describe"))
        _require_components(body["components"])
        _require_capabilities(body["capabilities"])
        self._description = body
        self.engine_version = str(body["engine"].get("version") or "unknown")
        self.graph_capable = bool(body["capabilities"]["graph_snapshot"])
        return body

    def describe_engine(self) -> dict[str, Any] | None:
        """The translator's report of its engine's components, for the declaration cross-check."""
        components = self._describe()["components"]
        return {role: components[role] or {} for role in _ROLES}

    def _reported(self, role: str) -> dict[str, Any]:
        """One component from the CACHED description, without triggering a call.

        ``effective_config`` is documented never to raise, and it runs while a run record is being
        written -- long after the moment a network failure could be reported usefully.
        """
        description = self._description or {}
        return (description.get("components") or {}).get(role) or {}

    def _effective_llm(self) -> dict[str, Any]:
        llm = self._reported("llm")
        return {"provider": llm.get("provider"), "model": llm.get("model")}

    def _effective_embedder(self) -> dict[str, Any]:
        embedder = self._reported("embedder")
        return {"model": embedder.get("model"), "dims": embedder.get("dims")}

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def prepare(self, isolation_unit: str) -> None:
        """Validate the contract, then open a fresh isolation unit."""
        self._describe()
        self._call("prepare", {"isolation_unit": isolation_unit})
        self._isolation = isolation_unit

    def cleanup(self) -> None:
        """Release the current unit's state; safe before prepare and safe twice."""
        if self._isolation is None:
            return
        self._call("cleanup", {})
        self._isolation = None

    # ------------------------------------------------------------------ #
    # Ingest / retrieve
    # ------------------------------------------------------------------ #

    def ingest(self, documents: list[Document]) -> None:
        """Hand the unit's documents to the translator verbatim.

        No rendering happens here. Turning a Document into whatever shape an engine wants is the
        translator's job, and is precisely the part of an adapter that cannot be configuration.
        """
        if not documents:
            return
        payload = {"documents": [asdict(doc) for doc in documents]}
        with self.latency.track("ingest"):
            body = self._call("ingest", payload)
        self._record(body, token_bucket="ingest", latency_bucket="ingest")

    def retrieve(
        self,
        query: str,
        k: int,
        user_id: str,
        query_timestamp: datetime | str | None = None,
    ) -> Recall:
        """Ask one query and rebuild the ranked documents the scorer reads."""
        payload = {
            "query": query,
            "k": k,
            "user_id": user_id,
            "query_timestamp": _iso(query_timestamp),
        }
        with self.latency.track("retrieve"):
            body = self._call("retrieve", payload)
        self._record(body, token_bucket="query", latency_bucket="retrieve")
        entries = body.get("documents")
        if not isinstance(entries, list):
            raise ContractError(
                f"retrieve must return a ranked 'documents' list, got "
                f"{type(entries).__name__}; see {_CONTRACT_DOC} section 4.")
        documents = [_to_document(entry, index, user_id) for index, entry in enumerate(entries)]
        raw = body.get("raw")
        return Recall(documents=documents,
                      declared=raw if isinstance(raw, dict) else {"raw": raw})

    def _record(self, body: dict[str, Any], *, token_bucket: str, latency_bucket: str) -> None:
        """Record the optional usage and engine-side timing a response may carry.

        Both are absent-or-real by contract. An omitted ``usage`` leaves the bucket empty, which
        ``TokenCollector.as_metrics`` renders as ``None`` -- "nobody counted" -- rather than as a
        zero, which would claim the engine spent nothing.
        """
        usage = body.get("usage")
        if isinstance(usage, dict):
            total = usage.get("total_tokens")
            if isinstance(total, (int, float)) and not isinstance(total, bool):
                self.tokens.record(token_bucket, int(total))
        engine_ms = body.get("engine_ms")
        if isinstance(engine_ms, (int, float)) and not isinstance(engine_ms, bool):
            self.latency.record(f"{latency_bucket}_engine", float(engine_ms))

    # ------------------------------------------------------------------ #
    # Metrics
    # ------------------------------------------------------------------ #

    def declared_latency(self) -> dict[str, list[float]]:
        """The translator's claim about its engine alone -- time memrank's hop cannot see.

        Samples rather than percentiles, so memrank pools them across the adapters a ``--workers``
        run builds and renders one statistic of one population. The wall-clock keys are memrank's
        own measurement and are not declared here: they include this adapter's hop, because that
        is what memrank actually measured.
        """
        return {bucket: samples
                for bucket in ("ingest_engine", "retrieve_engine")
                if (samples := self.latency.samples(bucket))}

    def latency_metrics(self) -> dict[str, float]:
        """The six wall-clock keys this adapter timed internally, plus engine-side time.

        No longer read by the run loop -- memrank times its own calls and asks
        :meth:`declared_latency` for the rest (ATO-2136). Kept because callers outside the loop
        still ask an adapter what it saw.
        """
        metrics = self.latency.as_metrics()
        for bucket in ("ingest", "retrieve"):
            summary = self.latency.summary(f"{bucket}_engine")
            if summary["count"]:
                metrics[f"{bucket}_engine_p50_ms"] = summary["p50_ms"]
                metrics[f"{bucket}_engine_p95_ms"] = summary["p95_ms"]
        return metrics

    def token_metrics(self) -> dict[str, float | None]:
        return self.tokens.as_metrics()


#: Deprecated alias of the class above -- the same class object, so an out-of-tree import and
#: every ``isinstance`` against the older spelling keep holding. The suffix went because a reader
#: copies the class name out of a first result, and ``Adapter`` is memrank's word for the wrapper
#: rather than the reader's word for the system. Removing it is plan step 22 (ATO-2151).
NativeAdapter = Native
