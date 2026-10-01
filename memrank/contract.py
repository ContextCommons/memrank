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
"""The value types the engine and evaluation contracts are written in.

Split out of :mod:`memrank.core` so the two ABCs and the data they exchange can be read
apart: a person implementing an engine needs the types here and two methods there. Every
name is re-exported from ``memrank.core``, which is where they have always been imported
from, and most from ``memrank`` itself.

Nothing here imports anything of memrank's -- these are the leaves of the contract, and a
value type that needed a module of ours would not be one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: The keys a run's ``latency_metrics`` and ``token_metrics`` carry. Named here because several
#: places assert on this shape -- the instrumentation collectors that render it, the conformance
#: suite, `memrank targets verify`, and the readers downstream -- and a list restated per site is
#: one edit away from as many different contracts.
#:
#: Latency is MEMRANK'S measurement, taken at its own call boundary and assembled in
#: `memrank.evaluation.measurement`; it is no longer asked of the engine (ATO-2136). Token usage
#: is the engine's own DECLARATION, because only the engine can be told what a provider billed --
#: optional, and absent means ``None`` per key rather than zero.
REQUIRED_LATENCY_KEYS: frozenset[str] = frozenset({
    "ingest_p50_ms", "ingest_p95_ms", "ingest_p99_ms",
    "retrieve_p50_ms", "retrieve_p95_ms", "retrieve_p99_ms",
})
REQUIRED_TOKEN_KEYS: frozenset[str] = frozenset({
    "tokens_per_query_mean", "tokens_per_query_p95",
    "tokens_per_ingest_mean", "tokens_per_ingest_p95",
})


@dataclass
class Document:
    """A single piece of content the adapter ingests or returns.

    The shape mirrors what every benchmark loader produces: a stable ``id``,
    free-form ``content``, an optional ``user_id`` for isolation scoping, an
    optional ``timestamp`` (ISO-8601), and an optional structured ``messages``
    list when the source has multi-turn structure.

    Adapters should treat ``content`` as the canonical text payload.
    """

    id: str
    content: str
    user_id: str | None = None
    timestamp: str | None = None
    context: str | None = None
    messages: list[dict[str, Any]] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        # Truncated by hand rather than dataclass-generated: a corpus document is
        # kilobytes of text, and a notebook cell showing a list of these must read as an
        # inventory, not a transcript. The size says what the ellipsis hides.
        size = len(self.content.encode("utf-8"))
        head = self.content[:40].replace("\n", " ")
        ellipsis = "…" if len(self.content) > 40 else ""
        scope = f", user_id={self.user_id!r}" if self.user_id is not None else ""
        return f"Document({self.id!r}{scope}, {size:,}B: {head!r}{ellipsis})"


@dataclass(frozen=True)
class Recall:
    """What one :meth:`~memrank.core.MemoryAdapter.retrieve` call gave back.

    Two things, and the type says which is which: the passages the engine recalled, in
    the order it ranked them, and whatever the engine declared about the call. Both are
    part of the trace -- what was recalled, and what the engine said -- so a person
    reading a low score can see the ranked list and the provider's own answer side by
    side rather than an unlabelled pair.

    Replaces the ``tuple[list[Document], dict[str, Any]]`` retrieve returned since the
    first commit: a pair whose halves only the source said apart.

    ``declared`` is the provider payload untouched, so it has no shape memrank can name --
    it is the engine's words, recorded for forensics and never interpreted as a
    measurement. ``{}`` is the honest value for an engine with nothing to add.
    """

    documents: list[Document]
    declared: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        # Counts and keys, never the passages: a Recall printed in a notebook is an
        # inventory of a call, and its documents each already truncate themselves.
        keys = ",".join(sorted(self.declared)) or "-"
        return f"Recall(documents={len(self.documents)}, declared={{{keys}}})"


@dataclass
class AdapterResponse:
    """Output from a single adapter ``retrieve`` call against one query.

    ``documents`` is the ranked list returned by the adapter; ``raw`` is the
    untouched provider payload (kept for forensic debugging in the receipt);
    ``latency_ms`` is wall-clock time as measured by the runner.
    """

    query_id: str
    documents: list[Document]
    raw: dict[str, Any] | None = None
    latency_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkUnit:
    """A scoring unit for a benchmark.

    A unit is the smallest piece of work a benchmark scores independently.
    For LoCoMo it's a conversation; for BEAM it's a conversation x ability;
    for LongMemEval it's a single QA item. Each unit carries its own
    ``isolation_id`` so the adapter can scope memory state correctly.
    """

    unit_id: str
    isolation_id: str
    documents: list[Document]
    queries: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        # Counts, not contents: the generated repr inlined every document's full text,
        # which made `bench.load()` in a notebook print the whole corpus.
        return (f"BenchmarkUnit({self.unit_id!r}, documents={len(self.documents)}, "
                f"queries={len(self.queries)})")




@dataclass(frozen=True)
class EvalInfo:
    """Static, declared description of an eval for ``memrank evals show``.

    DECLARED, never loaded: rendering this must not construct units or touch the
    dataset cache -- ``evals show`` is a catalog view, not a download trigger. Which
    is also why sizes are prose claims (``units_declared``) rather than counts: a
    real count would require ``load()``.
    """

    unit: str
    """What one scoring unit is, e.g. ``"conversation"``."""
    units_declared: str
    """Prose size claim from the dataset's own description (declared, not counted)."""
    slices: tuple[str, ...]
    """Named slices besides full, e.g. ``("smoke", "mini")``."""
    tiers: tuple[str, ...] = ()
    """Named tiers, leading with the default; empty means no tier dimension."""

