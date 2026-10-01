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
"""Evidence-based retrieval scoring.

Engine-id-agnostic: a query is scored by whether the gold answer's required
spans appear in the *content* of a single retrieved document, optionally gated
by true evidence doc ids (matched against ``metadata['doc_id']``, never the
engine's opaque ``id``). Negative queries invert: correct means the wrong
memory was NOT surfaced. There is deliberately no "non-empty = hit" fallback.
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from memrank.composition import Scorer
from memrank.core import AdapterResponse, BenchmarkUnit, Document

_PUNCT_WS = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Lowercase, strip accents, drop punctuation, collapse whitespace."""
    decomposed = unicodedata.normalize("NFKD", text)
    no_accents = "".join(c for c in decomposed if not unicodedata.combining(c))
    no_punct = _PUNCT_WS.sub(" ", no_accents.lower())
    return _WS.sub(" ", no_punct).strip()


@dataclass
class EvidenceSpec:
    """Per-query gold evidence.

    ``kind='positive'``: hit if a single doc contains all ``required_spans``
    (and no ``forbidden_spans``), optionally within ``evidence_doc_ids``.
    ``kind='negative'``: hit if NO retrieved doc contains any ``forbidden_spans``.
    """

    required_spans: list[str] = field(default_factory=list)
    forbidden_spans: list[str] = field(default_factory=list)
    evidence_doc_ids: list[str] = field(default_factory=list)
    kind: str = "positive"


@dataclass
class MatchResult:
    """Outcome of scoring one query against retrieved docs."""

    hit: bool
    matched_span: str | None = None
    matched_doc_id: str | None = None


def _source_id(doc: Document) -> str | None:
    return (doc.metadata or {}).get("doc_id")


def _doc_satisfies(spec: EvidenceSpec, doc: Document) -> str | None:
    """Return the matched span set as a string if ``doc`` satisfies a positive spec."""
    content = normalize(doc.content or "")
    source = _source_id(doc)
    # Lenient gate: enforce evidence_doc_ids only when the doc actually carries a
    # doc_id. Fact-extracting engines (AM, Mem0) and some benchmarks (BEAM) don't
    # preserve source ids; fall back to content matching rather than voiding all hits.
    if spec.evidence_doc_ids and source is not None and source not in spec.evidence_doc_ids:
        return None
    if any(normalize(f) in content for f in spec.forbidden_spans):
        return None
    required = [normalize(s) for s in spec.required_spans]
    if required and all(span in content for span in required):
        return " + ".join(required)
    return None


def score_query(spec: EvidenceSpec, retrieved: list[Document]) -> MatchResult:
    """Score one query's retrieved documents against its evidence spec."""
    if spec.kind == "negative":
        for doc in retrieved:
            if any(normalize(f) in normalize(doc.content or "") for f in spec.forbidden_spans):
                return MatchResult(hit=False, matched_doc_id=doc.id)
        return MatchResult(hit=True)
    for doc in retrieved:
        span = _doc_satisfies(spec, doc)
        if span is not None:
            return MatchResult(hit=True, matched_span=span, matched_doc_id=doc.id)
    return MatchResult(hit=False)


def spec_from_query(query: dict) -> EvidenceSpec:
    """Build an EvidenceSpec from a benchmark query dict."""
    return EvidenceSpec(
        required_spans=list(query.get("required_spans") or []),
        forbidden_spans=list(query.get("forbidden_spans") or []),
        evidence_doc_ids=list(query.get("evidence_doc_ids") or []),
        kind=str(query.get("kind") or "positive"),
    )


#: The category a query that declares none is counted under.
UNCATEGORIZED = "uncategorized"


class SpanRecall(Scorer):
    """memrank's deterministic scorer, under a name a person can import and point at.

    It is the scorer the demo benchmark has always used -- ``score_query(spec_from_query(q),
    documents)`` per query, the mean over a unit's queries -- lifted out of that benchmark's
    import namespace so somebody who brings only questions does not have to write a
    :meth:`~memrank.core.Benchmark.score` of their own to get a number (ATO-2135).

    The name says what it decides and no more. It marks whether a query's gold SPANS appear
    verbatim in a retrieved document; it is a retrieval proxy and never a claim about whether an
    answer to the question would be correct. :attr:`METRIC_LABEL` is the string it writes into
    every unit it scores, so the caveat travels with the number rather than with the
    documentation.
    """

    #: The kind of number this scorer produces, in `Benchmark.quality_metric`'s vocabulary.
    quality_metric = "substring_recall"
    #: The one scored key an aggregation may be written over.
    criterion_names = ("composite",)
    identity = "span-recall"
    #: The per-unit honest label, carried in the scored dict's ``metric`` key.
    METRIC_LABEL = "evidence_recall (retrieval proxy; not answer correctness)"

    def hit(self, query: dict[str, Any], retrieved: list[Document]) -> MatchResult:
        """Mark one query against what came back for it."""
        return score_query(spec_from_query(query), retrieved)

    def score(self, unit: BenchmarkUnit,
              responses: Sequence[AdapterResponse]) -> dict[str, Any]:
        """Mark one unit: composite, per-category breakdown, query count and the label."""
        by_id = {r.query_id: r for r in responses}
        per_category: dict[str, list[int]] = defaultdict(list)
        all_hits: list[int] = []
        for query in unit.queries:
            response = by_id.get(query["id"])
            marked = 1 if self.hit(query, response.documents if response else []).hit else 0
            all_hits.append(marked)
            per_category[query.get("category", UNCATEGORIZED)].append(marked)
        composite = sum(all_hits) / len(all_hits) if all_hits else 0.0
        return {
            "composite": composite,
            "per_category": {c: sum(v) / len(v) for c, v in per_category.items()},
            "n_queries": len(all_hits),
            "metric": self.METRIC_LABEL,
        }
