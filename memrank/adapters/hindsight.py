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
"""Hindsight HTTP adapter.

Talks to a self-hosted Hindsight API (``hindsight-api``) over HTTP. Each
isolation unit gets its own bank, optionally reset on prepare so prior unit
state cannot leak in.

Targets the current Hindsight API (``ghcr.io/vectorize-io/hindsight``, port
8888): banks are created idempotently via ``PUT /v1/default/banks/{bank_id}``,
memories are retained under ``.../{bank_id}/memories`` and recalled under
``.../{bank_id}/memories/recall``.
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from typing import Any

import httpx

from memrank.adapters import errors as adapter_errors
from memrank.adapters import transcript
from memrank.adapters.effective import embedder_from_env, llm_from_env
from memrank.core import Document, Memory, Recall
from memrank.instrumentation import LatencyCollector, TokenCollector

_DEFAULT_BASE_URL = "http://localhost:8888"
_DEFAULT_TIMEOUT_S = 120.0
_BANK_ID_SAFE = re.compile(r"[^a-zA-Z0-9_-]")

#: Bank-creation settings this adapter knows how to send. The manifest deliberately does not police
#: an `ingest:` block's keys -- they are the engine's vocabulary, not memrank's -- so the adapter is
#: where an unsupported one has to stop. It must stop: a key this adapter does not forward is
#: silently dropped, and a target declaring `enable_obserrvations: false` would then run with
#: observations ON while its receipt recorded the setting it asked for.
_SUPPORTED_INGEST_SETTINGS: frozenset[str] = frozenset({"enable_observations"})


def _safe_bank_id(raw: str) -> str:
    """Strip characters that the hindsight-api bank_id parser rejects."""
    cleaned = _BANK_ID_SAFE.sub("-", raw or "default")
    return cleaned[:64] or "default"


class Hindsight(Memory):
    """HTTP adapter for a self-hosted Hindsight backend."""

    name = "hindsight"
    base_url_env = "HINDSIGHT_API_URL"
    version = "0.1.0"
    engine_version = os.environ.get("HINDSIGHT_ENGINE_VERSION", "unknown")

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        bank_prefix: str = "memrank",
        timeout_s: float | None = None,
        reset_per_unit: bool = True,
        retrieval: dict[str, Any] | None = None,
        ingest: dict[str, Any] | None = None,
    ) -> None:
        self.base_url = (base_url or os.environ.get("HINDSIGHT_API_URL", _DEFAULT_BASE_URL)).rstrip("/")
        # Via the credential chokepoint (environment -> wallet), so `memrank secrets set
        # HINDSIGHT_API_KEY` actually reaches the adapter. Defaults to "" because a locally-run
        # hindsight with auth disabled needs no key.
        from memrank.config import secret

        self.api_key = api_key or secret("HINDSIGHT_API_KEY") or ""
        self.bank_prefix = bank_prefix
        # Per-request HTTP timeout, overridable via HINDSIGHT_TIMEOUT_S. A large benchmark doc's
        # retain (LLM extraction, amplified by retain retries) can run for minutes in one call, so
        # the cloud sets this generously.
        self.timeout_s = (
            timeout_s if timeout_s is not None
            else float(os.environ.get("HINDSIGHT_TIMEOUT_S", _DEFAULT_TIMEOUT_S))
        )
        self.reset_per_unit = reset_per_unit
        # Retrieval depth, from the target's `retrieval:` block. The defaults are the matched-mode
        # settings: 8192 fact tokens and 8192 chunk tokens, both already above the 5000-token
        # budget the runner enforces, and `budget` left to the engine's own `mid`. A faithful
        # variant overrides them with the vendor's published figures (budget=high, 32768 / 16384)
        # -- and since 2026-08-19 that variant is the BARE `hindsight`, with `hindsight:matched`
        # the one that clears back to these defaults.
        self.retrieval = dict(retrieval or {})
        # Bank-creation settings, from the target's `ingest:` block. Empty by default, which is the
        # positive statement "the engine's own defaults" -- hindsight ships observations ON. Only
        # The bare, faithful `hindsight` states otherwise, because AMB disabled them to produce
        # the published numbers. `hindsight:matched` clears
        # the block back to empty, which is why "empty" has to mean the engine's default here.
        self.ingest_settings = dict(ingest or {})
        unsupported = set(self.ingest_settings) - _SUPPORTED_INGEST_SETTINGS
        if unsupported:
            # At construction, before a run starts and long before a number exists. Silently
            # dropping the key is the failure worth preventing: it produces a complete, plausible
            # result for a configuration the target never had.
            raise ValueError(
                f"hindsight cannot send ingest settings {sorted(unsupported)}; it supports "
                f"{sorted(_SUPPORTED_INGEST_SETTINGS)}. An unforwarded setting would leave the "
                f"engine on its default while the receipt records the value that was asked for.")
        self._isolation: str | None = None
        self._bank_id: str | None = None
        self._client: httpx.Client | None = None
        self.latency = LatencyCollector()
        self.tokens = TokenCollector()

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def _http(self) -> httpx.Client:
        if self._client is None:
            headers: dict[str, str] = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            self._client = adapter_errors.engine_client(
                engine=self.name, timeout_s=self.timeout_s, env_var="HINDSIGHT_TIMEOUT_S",
                base_url_env=self.base_url_env,
                base_url=self.base_url, headers=headers)
        return self._client

    def prepare(self, isolation_unit: str) -> None:
        """Create / reset a per-unit bank."""
        self._isolation = isolation_unit
        self._bank_id = _safe_bank_id(f"{self.bank_prefix}-{isolation_unit}")
        client = self._http()
        if self.reset_per_unit:
            try:
                client.delete(f"/v1/default/banks/{self._bank_id}")
            except httpx.HTTPError:
                # Bank may not exist yet -- recreate proceeds either way.
                pass
        # PUT is idempotent create-or-update; bank_id lives in the path, not the body.
        payload: dict[str, Any] = {"name": f"Memrank Bank ({self._bank_id})"}
        # Only sent when a target states it, exactly as `retrieval["budget"]` is in `retrieve`:
        # unset means the engine's own default and sending a value we invented would misreport the
        # configuration a number came from.
        #
        # This is the ONLY moment it can be said. `enable_observations` decides which of
        # hindsight's four memory networks are written, and the bank is created here -- by ingest
        # time the answer is already fixed, which is why no `retrieval:` block could ever reach it
        # (audit F6).
        if "enable_observations" in self.ingest_settings:
            payload["enable_observations"] = bool(self.ingest_settings["enable_observations"])
        response = client.put(f"/v1/default/banks/{self._bank_id}", json=payload)
        adapter_errors.raise_for_status(response)
        self._assert_bank_configured(client, payload)

    def _assert_bank_configured(self, client: httpx.Client, requested: dict[str, Any]) -> None:
        """Read the bank's config back and confirm the engine took what we asked for.

        A 200 from the PUT is not evidence. The bank endpoint accepts unknown fields silently and
        its response body echoes only name and disposition, so a misspelled or removed setting
        looks exactly like an honoured one -- and the run would then report a faithful
        configuration it never had. `GET .../config` reports the effective value and lists which
        keys are overrides, so the claim is checkable at the only moment it is still cheap.
        """
        stated = {key: value for key, value in requested.items() if key != "name"}
        if not stated:
            return
        response = client.get(f"/v1/default/banks/{self._bank_id}/config")
        adapter_errors.raise_for_status(response)
        effective = (adapter_errors.safe_json(response).get("config") or {})
        wrong = {key: (value, effective.get(key)) for key, value in stated.items()
                 if effective.get(key) != value}
        if wrong:
            raise adapter_errors.ConfigurationNotTaken(
                f"hindsight bank {self._bank_id!r} did not take the configuration this target "
                f"declares: {wrong} (requested, effective). Refusing to ingest -- the run would "
                f"record a configuration it does not have.")

    def cleanup(self) -> None:
        """Best-effort delete of the per-unit bank."""
        if self._bank_id is None:
            return
        client = self._http()
        try:
            client.delete(f"/v1/default/banks/{self._bank_id}")
        except httpx.HTTPError:
            pass
        self._bank_id = None
        self._isolation = None

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    # ------------------------------------------------------------------ #
    # Ingest / retrieve
    # ------------------------------------------------------------------ #

    @staticmethod
    def _doc_to_item(doc: Document) -> dict[str, Any]:
        # Rendered, not raw: the vendor asks for text conveying "who said what, and when"
        # (their ingest guidance), and raw content is a JSON dump for conversational
        # benchmarks. The native `timestamp` below still goes too -- it drives temporal
        # filtering, while the text is what extraction reads.
        item: dict[str, Any] = {
            "content": transcript.render(doc).replace("\x00", ""),
            "document_id": doc.id,
            "metadata": {"doc_id": doc.id, **(doc.metadata or {})},
        }
        if doc.timestamp:
            item["timestamp"] = doc.timestamp
        if doc.context:
            item["context"] = doc.context
        if doc.user_id:
            item.setdefault("tags", []).append(f"user:{doc.user_id}")
        return item

    def ingest(self, documents: list[Document]) -> None:
        if self._bank_id is None:
            raise adapter_errors.ConfigurationNotTaken(
                "Hindsight.ingest called before prepare() -- no bank exists to ingest into")
        client = self._http()
        items = [self._doc_to_item(doc) for doc in documents]
        if not items:
            return
        payload = {"items": items, "async": False}
        with self.latency.track("ingest"):
            response = client.post(f"/v1/default/banks/{self._bank_id}/memories", json=payload)
            adapter_errors.raise_for_status(response)
        body = adapter_errors.safe_json(response)
        self._assert_retain_settled(body, expected_items=len(items))
        usage = body.get("usage") or {}
        total = int(usage.get("total_tokens", 0))
        if total:
            self.tokens.record("ingest", total)

    @staticmethod
    def _assert_retain_settled(body: dict[str, Any], *, expected_items: int) -> None:
        """Confirm retain finished before the caller is allowed to recall (audit F7).

        We post ``"async": False``, which SHOULD make retain settled by return. Nothing checked
        it, and a regression there does not raise -- it recalls from a bank still filling and
        reports the shortfall as poor recall. A wrong number, not an error, which is the failure
        class this project exists to catch.

        Asserted on content, never on elapsed time (CLAUDE.md). The engine's own response
        distinguishes the two modes: a settled retain reports ``async: false`` and carries
        ``usage``; a queued one reports ``async: true`` and carries an ``operation_id`` instead.
        Observed directly against ghcr.io/vectorize-io/hindsight, both ways.
        """
        if not body.get("success", False):
            raise RuntimeError(f"hindsight retain did not report success: {body}")
        if body.get("async") or body.get("operation_id"):
            raise RuntimeError(
                f"hindsight queued retain instead of settling it (async={body.get('async')!r}, "
                f"operation_id={body.get('operation_id')!r}). Recall would run against a bank "
                f"still filling and score as poor recall.")
        counted = body.get("items_count")
        if counted is not None and int(counted) != expected_items:
            raise RuntimeError(
                f"hindsight retained {counted} of {expected_items} items. A partial ingest scores "
                f"as poor recall rather than failing.")

    def retrieve(
        self,
        query: str,
        k: int,
        user_id: str,
        query_timestamp: datetime | str | None = None,
    ) -> Recall:
        if self._bank_id is None:
            raise RuntimeError("Hindsight.retrieve called before prepare()")
        client = self._http()
        # ``include`` sub-fields are per-type option objects ({"max_tokens": N}), not booleans;
        # omitting a key (entities / source_facts) excludes that type.
        payload: dict[str, Any] = {
            "query": query[:1900],
            "max_tokens": int(self.retrieval.get("max_tokens", 8192)),
            "include": {"chunks": {"max_tokens": int(self.retrieval.get("chunk_max_tokens", 8192))}},
        }
        # Only sent when a target states it: unset means the engine's own default tier (`mid`), and
        # sending a value we invented would misreport which fusion depth produced the row.
        if self.retrieval.get("budget"):
            payload["budget"] = str(self.retrieval["budget"])
        if query_timestamp is not None:
            payload["query_timestamp"] = (
                query_timestamp.isoformat() if isinstance(query_timestamp, datetime) else str(query_timestamp)
            )
        if user_id:
            payload["tags"] = [f"user:{user_id}"]
            payload["tags_match"] = "any_strict"
        with self.latency.track("retrieve"):
            response = client.post(f"/v1/default/banks/{self._bank_id}/memories/recall", json=payload)
            adapter_errors.raise_for_status(response)
        body = adapter_errors.safe_json(response)
        usage = body.get("usage") or {}
        total = int(usage.get("total_tokens", 0))
        if total:
            self.tokens.record("query", total)
        # NOT sliced to ``k``. Hindsight has no top-k: it packs recall to a TOKEN budget, and its
        # vendor is explicit that "agents don't think in terms of result counts -- they think in
        # tokens". Cutting to ten items imposed a shape the engine does not model, and did it
        # BEFORE the fairness control could act -- memrank caps every target at --token-budget in
        # runner._context_text, so the budget is enforced either way. The slice only decided how
        # much of that budget hindsight was allowed to fill (audit F4/F5).
        #
        # ``k`` stays in the signature because the Memory contract defines it and other
        # engines need it; for this one it is not a meaningful request.
        results = body.get("results") or []
        docs = [self._result_to_doc(result, idx, user_id) for idx, result in enumerate(results)]
        return Recall(documents=docs, declared=body)

    @staticmethod
    def _result_to_doc(result: dict[str, Any], idx: int, user_id: str) -> Document:
        """Map a RecallResult to a Document (score lives under ``scores.final``)."""
        doc_id = str(result.get("id") or result.get("chunk_id") or idx)
        text = str(result.get("text") or result.get("content") or "")
        metadata: dict[str, Any] = {}
        if result.get("type"):
            metadata["type"] = result["type"]
        score = (result.get("scores") or {}).get("final")
        if score is not None:
            metadata["score"] = score
        return Document(id=doc_id, content=text, user_id=user_id, metadata=metadata)

    # ------------------------------------------------------------------ #
    # Metrics
    # ------------------------------------------------------------------ #

    def latency_metrics(self) -> dict[str, float]:
        return self.latency.as_metrics()

    def token_metrics(self) -> dict[str, float | None]:
        return self.tokens.as_metrics()

    def _effective_llm(self) -> dict:
        return llm_from_env("HINDSIGHT_")

    def _effective_embedder(self) -> dict:
        return embedder_from_env("HINDSIGHT_")


#: Deprecated alias of the class above -- the same class object, so an out-of-tree import and
#: every ``isinstance`` against the older spelling keep holding. The suffix went because a reader
#: copies the class name out of a first result, and ``Adapter`` is memrank's word for the wrapper
#: rather than the reader's word for the system. Removing it is plan step 22 (ATO-2151).
HindsightAdapter = Hindsight
