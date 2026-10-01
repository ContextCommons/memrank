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
"""LongMemEval's two published retrieval metrics, judge-free.

`recall_all@k` and `ndcg_any@k` (`src/retrieval/eval_utils.py`) are the only LongMemEval numbers
that measure the memory system rather than the reader or the judge. They are immune to every
divergence the ADR accepts -- judge model, reading strategy, answer length -- which makes them the
one place a memrank LongMemEval figure can be compared to a published one without an argument
about instruments.

**`recall_all@k` is a per-question 0/1 indicator: 1 only when EVERY gold document is inside the
top k.** It is not fractional recall. Reporting fractional recall under this name inflates the
number substantially, and doing so is the single easiest way to publish a wrong LongMemEval
figure -- which is why the distinction is asserted by a test rather than only described here.

Two exclusions, both RETRIEVAL-ONLY: the 30 abstention items (no ground-truth location exists)
and the 51 questions whose evidence is assistant-side (the paper indexes user-side utterances
only). The author confirmed in issue #16 that these must not reach the QA denominator, which
stays 500; several publications made exactly that error. The loader marks both via
``retrieval_scoreable``.

Presence is decided by DETERMINISTIC CONTENT MATCHING, reusing `memrank.metrics.evidence_recall`'s
matcher (containment, then 8-gram shingle Jaccard) so the two benchmarks share one definition of
"this retrieved text is that session", versioned by ``EVIDENCE_MATCHER_VERSION``. Only the
matcher is shared; the text it runs against differs, see ``_session_texts``.

An id join is not available and never will be: engines return their own opaque ids -- which is
what made the original gold-id proxy score a constant 0.0 -- and the positional handles the
harness mints are deliberately withheld from engines, because the dataset's real ids leak the
answer location.

Declared limitation, inherited: an engine that stores paraphrases legitimately misses a content
match. These are retrieval diagnostics reported beside the judged number, never folded into it.
"""

from __future__ import annotations

import math
from typing import Any

from memrank.core import AdapterResponse, BenchmarkUnit
from memrank.metrics.evidence_recall import EVIDENCE_MATCHER_VERSION, _matches, _normalize

#: The cutoffs the paper reports at session granularity (Table 3).
DEFAULT_KS = (5, 10)


def _session_texts(unit: BenchmarkUnit) -> dict[str, str]:
    """Session doc id -> the text an engine actually ingested.

    ``Document.content`` rather than the `messages` join `evidence_recall` uses for LoCoMo: since
    the dated-transcript rendering landed, LongMemEval's ``content`` carries a ``Session date:``
    header and ``role:`` prefixes that the bare message join does not, and matching an engine's
    verbatim echo against a text we never ingested fails for a reason that has nothing to do with
    retrieval. Match against what was stored.
    """
    return {doc.id: _normalize(doc.content or "") for doc in unit.documents}


def _rank_of_gold(gold_ids: list[str], sessions: dict[str, str],
                  retrieved: list[str]) -> dict[str, int]:
    """Gold doc id -> the rank of the first retrieved text that matches it (0-based).

    Rank matters here in a way it does not for LoCoMo's `recall_acc`: both LongMemEval metrics
    are cut at k, so a gold document matched only by the twentieth result is a miss at k=10.
    """
    first: dict[str, int] = {}
    for rank, text in enumerate(retrieved):
        for gid in gold_ids:
            if gid not in first and _matches(sessions.get(gid, ""), text):
                first[gid] = rank
    return first


def recall_all_at_k(gold_ranks: dict[str, int], n_gold: int, k: int) -> float:
    """1.0 only when ALL ``n_gold`` documents were matched within the top ``k``.

    The official `recall_all` -- not fractional. Two of three gold documents inside k scores
    **zero**, which is the whole point of the metric and the trap in reimplementing it.
    """
    if n_gold == 0:
        return 0.0
    return 1.0 if sum(1 for rank in gold_ranks.values() if rank < k) == n_gold else 0.0


def ndcg_any_at_k(gold_ranks: dict[str, int], n_gold: int, k: int) -> float:
    """Binary-relevance NDCG at ``k``: any gold document at a position is relevant.

    Ideal ranking puts every gold document first, so IDCG sums the first ``min(k, n_gold)``
    discounts. Rewards ranking evidence high, where `recall_all@k` only asks whether it made the
    cut at all -- which is why the paper reports both.
    """
    if n_gold == 0:
        return 0.0
    dcg = sum(1.0 / math.log2(rank + 2) for rank in gold_ranks.values() if rank < k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(k, n_gold)))
    return dcg / idcg if idcg else 0.0


def retrieval_metrics(unit: BenchmarkUnit, responses: list[AdapterResponse],
                      ks: tuple[int, ...] = DEFAULT_KS, *,
                      ranked: bool = True) -> dict[str, Any]:
    """Per-unit `recall_all@k` and `ndcg_any@k`, plus the per-query detail behind them.

    ``ranked=False`` suppresses both metrics entirely rather than scoring them. Both are cut at
    k, so they are undefined for an arm that returns the whole store in ingest order -- and
    scoring one anyway produces a 0.0 that reads as total retrieval failure when the arm in fact
    returned everything. Measured: `fixed-context` returns all 40-53 sessions and scored `recall_all@5` =
    0.0. The official harness makes the same exclusion by construction -- its retrieval metrics
    are computed only over `run_retrieval.py`'s ranked output, never over the long-context modes.
    """
    if not ranked:
        unscored: dict[str, Any] = {"n_retrieval_scoreable": 0, "retrieval_metrics_apply": False,
                                    "evidence_matcher_version": EVIDENCE_MATCHER_VERSION,
                                    "per_query": []}
        for k in ks:
            unscored[f"recall_all@{k}"] = None
            unscored[f"ndcg_any@{k}"] = None
        return unscored
    sessions = _session_texts(unit)
    retrieved_by_query: dict[str, list[str]] = {
        r.query_id: [_normalize(d.content or "") for d in (r.documents or [])]
        for r in responses
    }
    scored: dict[int, dict[str, list[float]]] = {k: {"recall": [], "ndcg": []} for k in ks}
    per_query: list[dict[str, Any]] = []
    for query in unit.queries:
        # Both exclusions live behind one loader-set flag, so a caller cannot apply one and
        # forget the other, and neither can leak into a QA denominator computed elsewhere.
        if not query.get("retrieval_scoreable"):
            continue
        gold_ids = list(query.get("evidence_doc_ids") or [])
        if not gold_ids:
            continue
        ranks = _rank_of_gold(gold_ids, sessions, retrieved_by_query.get(query["id"], []))
        row: dict[str, Any] = {"query_id": query["id"], "n_gold": len(gold_ids),
                               "matched_ranks": dict(sorted(ranks.items()))}
        for k in ks:
            recall = recall_all_at_k(ranks, len(gold_ids), k)
            ndcg = ndcg_any_at_k(ranks, len(gold_ids), k)
            scored[k]["recall"].append(recall)
            scored[k]["ndcg"].append(ndcg)
            row[f"recall_all@{k}"] = recall
            row[f"ndcg_any@{k}"] = ndcg
        per_query.append(row)

    out: dict[str, Any] = {
        # The retrieval denominator, which is NOT the QA denominator. Named so nobody has to
        # infer which questions these means cover.
        "n_retrieval_scoreable": len(per_query),
        "retrieval_metrics_apply": True,
        "evidence_matcher_version": EVIDENCE_MATCHER_VERSION,
        "per_query": per_query,
    }
    for k in ks:
        # None, never 0.0, when nothing was scoreable -- a slice of only abstention items has no
        # recall, and that is not a recall of zero.
        vals = scored[k]
        out[f"recall_all@{k}"] = (sum(vals["recall"]) / len(vals["recall"])
                                  if vals["recall"] else None)
        out[f"ndcg_any@{k}"] = (sum(vals["ndcg"]) / len(vals["ndcg"])
                                if vals["ndcg"] else None)
    return out


def weighted_mean(rows: list[tuple[float | None, int]]) -> float | None:
    """Micro-mean of ``(value, weight)`` pairs, or None when every weight is zero.

    Weighted rather than a mean of means so a unit covering three scoreable queries outweighs
    one covering a single query, and -- the case that actually bites -- so a unit EXCLUDED from a
    metric contributes nothing rather than contributing a zero. LongMemEval's 30 abstention
    items and 51 assistant-side questions are excluded from retrieval only; averaging unit means
    would silently score them 0 and drag the number down by 16%.
    """
    total = sum(w for _, w in rows)
    if not total:
        return None
    return sum((v or 0.0) * w for v, w in rows) / total
