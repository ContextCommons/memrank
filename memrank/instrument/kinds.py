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
#
# Used by the current path (memrank.core): it sits in a deprecated package but is not
# deprecated itself, and it must not warn.
"""`Memory`, the memory kind: you tell it things, and later you ask it for what is relevant.

The one memory contract is DEFINED here, under the kind's own name. `memrank.core` imports it
and keeps `MemoryAdapter` and `MemoryEngine` as deprecated aliases of this same class object,
so `isinstance(x, MemoryAdapter)`, every registered adapter and every out-of-tree subclass keep
holding; the CLI keeps its own nouns.

It lives here rather than in `system.py` because it needs the value types in `memrank.contract`
and `system.py` is what `memrank.core` imports; one module per kind that carries a contract
keeps the dependency pointing one way -- `core` -> `kinds` -> `system`.
"""

from __future__ import annotations

from abc import abstractmethod
from datetime import datetime
from typing import Any

from memrank.contract import REQUIRED_TOKEN_KEYS, Document, Recall
from memrank.instrument.system import System


class Memory(System):
    """The contract every memory engine implements to be benchmarkable.

    Subclasses declare ``name``, ``version`` (the adapter's own version) and
    ``engine_version`` (the wrapped engine's version) as class attributes.
    The runner exercises adapters in this lifecycle:

        for unit in benchmark.load():
            adapter.prepare(unit.isolation_id)
            adapter.ingest(unit.documents)
            for query in unit.queries:
                recall = adapter.retrieve(query["text"], k, query["user_id"])
            adapter.cleanup()

    Four lifecycle methods, and nothing that reports a measurement Memrank takes
    itself: latency is timed at Memrank's own call boundary, so an engine neither
    has to report it nor can flatter it (ATO-2136). What only an engine can know
    it may DECLARE, and both declarations are optional: :meth:`token_metrics` for
    what a provider billed it, and :meth:`declared_latency` for its own spend
    inside the hop Memrank timed around it.

    Adapters MUST be deterministic given the same seed (or document
    non-determinism explicitly).
    """

    name: str = "abstract"
    version: str = "0.0.0"
    engine_version: str = "unknown"
    # Whether this adapter emits a graph snapshot (``raw["graph_snapshot"]``) and
    # can therefore satisfy a ``requires_graph`` benchmark. False by default; the
    # runner SKIPS (adapter, benchmark) cells where the benchmark needs a graph
    # but the adapter is not graph-capable, rather than crashing in the scorer.
    graph_capable: bool = False
    # How much retrieved context this adapter may hand the reader. "matched" -- the default for every
    # real engine -- caps it at the shared ``--token-budget``, which is the project's central fairness
    # control: a row must not win by dumping more text. Only the full-context control arm sets
    # "uncapped"; only the no-memory arm sets "none".
    context_budget: str = "matched"
    # Whether this adapter returns a RANKED list. False for the in-context controls, which hand
    # back the whole store in ingest order -- rank-cut metrics (recall_all@k, ndcg_any@k) are
    # undefined for them, and reporting 0.0 reads as total retrieval failure when the arm in fact
    # returned everything. LongMemEval's own harness draws the same line: retrieval metrics come
    # from `run_retrieval.py`'s ranked output, and its `full-history-session` / `no-retrieval`
    # generation modes are never scored on them at all.
    ranks_results: bool = True
    # The environment variable that moves this adapter's engine address, named so a failure can
    # quote it. Every adapter that talks to an engine over the network declares one; the in-process
    # adapters leave it None, which is what makes "there is no address to change" sayable rather
    # than an omission. Read by `memrank.adapters.preflight`, which is the only place a user is
    # told where memrank looked -- a message that describes the setting instead of naming it sends
    # the reader to the source to find out what it is called.
    base_url_env: str | None = None

    @abstractmethod
    def prepare(self, isolation_unit: str) -> None:
        """Set up state for a fresh benchmark unit.

        The ``isolation_unit`` is the per-conversation, per-question, or
        per-user identifier from the benchmark. Adapters MUST guarantee that
        nothing ingested in a previous unit leaks into this one.
        """

    @abstractmethod
    def ingest(self, documents: list[Document]) -> None:
        """Load documents into the engine for the current isolation unit."""

    @abstractmethod
    def retrieve(
        self,
        query: str,
        k: int,
        user_id: str,
        query_timestamp: datetime | str | None = None,
    ) -> Recall:
        """Return what this engine recalled for one query.

        A :class:`~memrank.contract.Recall`: at most ``k`` documents in the order the
        engine ranked them -- the order IS the measurement, so never pad the list -- and
        ``declared``, the provider response untouched, which the runner stores in the
        reproducibility receipt for forensic debugging.

        An engine with nothing to declare leaves ``declared`` empty. Failures raise; an
        empty ``documents`` means "searched, found none".
        """

    @abstractmethod
    def cleanup(self) -> None:
        """Tear down state for the current isolation unit."""

    def declared_latency(self) -> dict[str, list[float]]:
        """Declare timings only this engine can see. Optional, and never the wall clock.

        memrank times every ingest and retrieve at its own call boundary and reports that as the
        run's latency, so there is nothing here for an engine to report twice or to flatter. What
        an engine CAN add is time nobody outside it can see -- a translator's ``engine_ms``, the
        engine's own spend inside the hop memrank measured around it.

        SAMPLES per bucket, in milliseconds, not percentiles: memrank pools them across the
        adapters a ``--workers`` run builds and renders ``<bucket>_p50_ms`` / ``<bucket>_p95_ms``
        itself, so a run at any width reports the same statistic of the same population. An engine
        rendering its own percentiles cannot be pooled, only picked between.

        The conventional buckets are ``ingest_engine`` and ``retrieve_engine``, which is what
        `docs/system-contract.md` section 7 already calls the figure to quote when discussing the
        engine rather than the harness. Declared, not verified: memrank cannot check it, and it is
        never the headline.
        """
        return {}

    def token_metrics(self) -> dict[str, float | None]:
        """Declare per-call token usage, where this engine is told what it spent. Optional.

        Keys: ``tokens_per_query_mean``, ``tokens_per_query_p95``, ``tokens_per_ingest_mean``,
        ``tokens_per_ingest_p95``. :class:`memrank.instrumentation.TokenCollector` renders them.

        A key whose value is ``None`` means the engine reported no usage for that bucket, which is
        the common case -- an engine that genuinely spent no tokens reports 0.0. The two are
        different claims and must not share a value. The default declares ``None`` throughout,
        which is the honest answer for an engine that cannot be asked, and is the same shape
        :meth:`describe_engine` returns ``None`` for.

        Not abstract, because an engine is not improved by writing a stub: it was required of
        everyone and never checked, and a run whose engine returned ``{}`` from both metric
        methods completed and produced a composite (ATO-2136). What memrank REPORTS for latency
        is its own measurement; what it reports here is this declaration.
        """
        return dict.fromkeys(REQUIRED_TOKEN_KEYS, None)

    def declare_components(
        self,
        *,
        llm: dict[str, Any] | None = None,
        embedder: dict[str, Any] | None = None,
        transport: str | None = None,
        verified: str | None = None,
    ) -> None:
        """Record the components a resolved target manifest declares for this adapter.

        Set by the target factory after construction, so a manifest -- not ambient environment --
        describes what the run record reports. Anything left ``None`` keeps the adapter's own
        env-derived answer, so an adapter that is never declared behaves exactly as before.

        This is a precedence rule, not a fallback: the factory cross-checks the manifest against the
        environment and raises on any disagreement *before* calling this, so the two sources can
        never silently differ by the time precedence applies.
        """
        self._declared: dict[str, Any] = {
            key: value for key, value in
            (("llm", llm), ("embedder", embedder), ("transport", transport),
             ("verified", verified))
            if value is not None
        }

    def effective_config(self) -> dict[str, Any]:
        """Return the components this adapter actually used, for the run record.

        Captures the memory-engine identity plus the (operator-declared) LLM and
        embedder the backend is configured with. Recording the LLM/embedder here --
        even before they are tunable knobs -- keeps runs comparable when they are
        later promoted to explicit parameters. Values default to ``None`` and this
        never raises. Subclasses override ``_effective_llm`` / ``_effective_embedder``;
        :meth:`declare_components` overrides both.
        """
        declared: dict[str, Any] = getattr(self, "_declared", {})
        return {
            "engine": {
                "name": self.name,
                "version": self.version,
                "engine_version": self.engine_version,
                "transport": declared.get("transport", getattr(self, "transport", "unknown")),
            },
            "llm": declared.get("llm", self._effective_llm()),
            "embedder": declared.get("embedder", self._effective_embedder()),
            # How much this record can be trusted: "engine" when the engine confirmed it,
            # "declared" when the engine cannot be asked. Deliberately not a boolean -- "not asked"
            # and "asked and failed" are different, and a mismatch never reaches here (it raises).
            "verified": declared.get("verified", "declared"),
        }

    def describe_engine(self) -> dict[str, Any] | None:
        """The engine's OWN report of the components it is running, or ``None``.

        This is the only path by which memrank learns what an engine is actually configured with
        rather than what an operator claims. Returning ``None`` means "this engine cannot be asked"
        -- which is the honest answer for most engines, and is recorded as such in the run record
        instead of being presented as a passed check.

        Shape mirrors :meth:`effective_config`: ``{"llm": {provider, model},
        "embedder": {provider, model, dims}}``. Any key may be absent when the engine does not
        report it; only the keys present are compared.

        Raises:
            Exception: Transport failures propagate. An unreachable engine is a real failure, not a
                reason to skip the check -- the run would fail moments later at ingest anyway.
        """
        return None

    def state_fingerprint(self, scope: str) -> str | None:
        """A digest of everything stored under ``scope``, or ``None`` if the engine cannot say.

        Exists to answer one question that cannot be answered any other way: after a failed
        ingest, **did anything get written?** Only the engine knows, and the answer decides whether
        retrying is safe.

        mem0's ``add()`` is three steps -- extract, reconcile, then a loop that writes. A failure in
        the first two wrote nothing and a retry is exactly equivalent to never having failed. A
        failure inside the loop leaves the scope half-written, and re-running extraction against a
        partially populated store makes correctness depend on the reconcile LLM noticing, which is
        not something a benchmark number should rest on. Comparing this digest before and after
        distinguishes the two cases instead of assuming one.

        ``None`` -- the default -- means "this engine cannot be asked", the honest answer for most,
        and the runner treats it as CANNOT PROVE SAFE rather than as permission. An engine without
        it simply does not get retried.

        Must digest CONTENT, not just a count: reconciliation can UPDATE a memory in place, which
        rewrites text while leaving the number of memories unchanged.

        Must be stable across repeated reads of an unchanged store -- sort before hashing, since
        result order is a vector store's business and not a change in state.

        Args:
            scope: The isolation unit whose state to digest -- the same value ``prepare`` was given.

        Returns:
            A digest, or ``None`` when the engine exposes no way to enumerate its own state.
        """
        return None

    def _effective_llm(self) -> dict[str, Any]:
        """Provider/model of the engine's extraction LLM (override to populate)."""
        return {"provider": None, "model": None}

    def _effective_embedder(self) -> dict[str, Any]:
        """Model/dims of the engine's embedder (override to populate)."""
        return {"model": None, "dims": None}


__all__ = ["Memory"]
