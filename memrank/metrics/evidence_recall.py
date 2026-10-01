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
"""LoCoMo's official evidence-recall metric, at session granularity, judge-free.

`recall_acc` is the one metric LoCoMo publishes that isolates the memory system from the
reader and the judge, and no surveyed harness computes it
(docs/research/2026-08-13-how-locomo-is-actually-evaluated.md). The official harness checked
gold `evidence` dia_ids against its own retrieval store; memrank's engines return opaque
documents, so presence is decided by DETERMINISTIC CONTENT MATCHING against the session text
the engines ingested -- semantics fixed in
docs/superpowers/specs/2026-08-13-evidence-recall-design.md, versioned by
``EVIDENCE_MATCHER_VERSION`` so a future matcher cannot silently redefine the metric.

Declared limitation, by design: an engine that stores paraphrases legitimately misses a
content match. This measures evidence retrievability under an extractive reading -- per-query
``matched``/``unmatched`` detail makes the cause auditable -- and it is a retrieval diagnostic,
never the quality headline.
"""

from __future__ import annotations

from typing import Any

from memrank.core import AdapterResponse, BenchmarkUnit, Document

#: Bumped when the matching semantics change; recorded next to the metric on the artifact.
EVIDENCE_MATCHER_VERSION = 1

#: Character-shingle size. Eight characters spans a word boundary or two, so shared shingles
#: mean shared surface text rather than shared alphabet.
_SHINGLE_SIZE = 8

#: Jaccard similarity over shingles at or above which a retrieved text counts as the session.
_JACCARD_THRESHOLD = 0.5


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _shingles(text: str) -> set[str]:
    if len(text) < _SHINGLE_SIZE:
        return {text} if text else set()
    return {text[i:i + _SHINGLE_SIZE] for i in range(len(text) - _SHINGLE_SIZE + 1)}


def _matches(session_text: str, retrieved_text: str) -> bool:
    """Containment first (verbatim echo, whole-session snippets), shingle Jaccard second
    (partial snippets, truncations). Both deterministic; no model anywhere."""
    if not session_text or not retrieved_text:
        return False
    if session_text in retrieved_text or retrieved_text in session_text:
        return True
    a, b = _shingles(session_text), _shingles(retrieved_text)
    union = len(a | b)
    return union > 0 and len(a & b) / union >= _JACCARD_THRESHOLD


def _rendered_session_texts(unit: BenchmarkUnit) -> dict[str, str]:
    """Session doc id -> the speaker-attributed rendering the engines ingested.

    The `messages` rendering, NOT the raw JSON `content` blob: matching must run against what
    an engine could actually have stored.
    """
    def rendered(doc: Document) -> str:
        parts = [m.get("content", "") for m in (doc.messages or [])]
        return _normalize(" ".join(p for p in parts if p))
    return {doc.id: rendered(doc) for doc in unit.documents}


def evidence_recall(unit: BenchmarkUnit,
                    responses: list[AdapterResponse]) -> dict[str, Any]:
    """Official semantics at session granularity: per query with evidence, the fraction of its
    gold sessions matched by any retrieved document; mean over those queries. Queries with no
    `evidence` are excluded from the denominator (the official scorer would divide by zero);
    a unit with none reports ``None``, never 0.0.
    """
    sessions = _rendered_session_texts(unit)
    retrieved_by_query: dict[str, list[str]] = {
        r.query_id: [_normalize(d.content or "") for d in (r.documents or [])]
        for r in responses
    }
    per_query: list[dict[str, Any]] = []
    recalls: list[float] = []
    for query in unit.queries:
        gold_ids = list(query.get("evidence_doc_ids") or [])
        if not gold_ids:
            continue
        retrieved = retrieved_by_query.get(query["id"], [])
        matched = [gid for gid in gold_ids
                   if any(_matches(sessions.get(gid, ""), text) for text in retrieved)]
        unmatched = [gid for gid in gold_ids if gid not in matched]
        recalls.append(len(matched) / len(gold_ids))
        per_query.append({"query_id": query["id"], "matched": matched,
                          "unmatched": unmatched, "recall": recalls[-1]})
    return {
        "evidence_recall": (sum(recalls) / len(recalls)) if recalls else None,
        "n_queries_with_evidence": len(recalls),
        "evidence_matcher_version": EVIDENCE_MATCHER_VERSION,
        "per_query": per_query,
    }
