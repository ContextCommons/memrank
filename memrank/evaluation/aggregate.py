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
"""One (adapter x benchmark) cell's numbers, reduced into the artifact dict.

`_aggregate_cell` builds the dict ten modules downstream read byte-for-byte (registry,
record, leaderboard, API projection, the uploader) -- treat its key set as a contract;
`tests/orchestration/test_golden_run.py` pins it.
"""
from __future__ import annotations

from typing import Any

from memrank.core import Benchmark
from memrank.evaluation.constants import DEFAULT_JUDGE_WORKERS
from memrank.evaluation.result import EvalResult
from memrank.evaluation.unit_outcome import UnitOutcome, failure_summary
from memrank.metrics import cost
from memrank.metrics.scoring import score_query, spec_from_query


def _benchmark_params(benchmark: Benchmark) -> dict[str, Any]:
    """Benchmark-specific knobs that MUST travel into artifacts + the receipt
    hash. Sourced from the benchmark's own contract method so a new
    benchmark-specific dimension is never silently dropped."""
    return benchmark.config_for_receipt()

def _ingest_throughput(summary: dict[str, Any] | None) -> dict[str, float] | None:
    """What ingest COST, in summed document latency -- NOT elapsed time.

    Percentiles alone cannot say what ingest cost a run: p50=12,750 ms is the same number whether
    seventy documents paid it or one did. Derived rather than measured separately, so it can never
    disagree with the summary it comes from.

    THIS USED TO CALL ITSELF WALL-CLOCK AND IS NOT. ``count * mean_ms`` sums every document's
    latency, so under ``--workers N`` it overstates elapsed time by roughly N: a run that took
    11m 30s reported 50m 49s, and `runs show --full` printed it as "ingest total". The field is now
    named for what it measures. Elapsed time comes from the run's own timestamps --
    ``status.json``'s ``started_at``/``synced_at``, which is what `scripts/internal/inspect-run.py timing`
    prints beside this so the two can never be confused again.

    ``documents_per_second`` inherits the same meaning: per-worker throughput, not the run's. It is
    a property of the ENGINE (how fast one document is absorbed) rather than of the harness (how
    many were absorbed at once), which is the useful reading for comparing engines.

    ``None`` rather than zero for an empty ingest -- the read-only controls (`no-context`, `fixed-context`) ingest
    nothing, and 0 docs/sec would read as an engine that is infinitely slow rather than one that
    was never asked.
    """
    if not summary or not summary.get("count"):
        return None
    seconds = summary["count"] * summary["mean_ms"] / 1000.0
    return {"documents": summary["count"],
            "latency_total_seconds": seconds,
            # Retained under its old name so existing artifacts and readers keep resolving; both
            # keys carry the same number, and the new one says which number it is.
            "total_seconds": seconds,
            "documents_per_second": summary["count"] / seconds if seconds else None}


def _corpus_size(ingested: list[dict[str, Any]]) -> dict[str, int]:
    """How much data this cell actually pushed through: documents, bytes and tokens.

    Tokens use the same pinned encoding as ``context_tokens_mean`` (:mod:`memrank.metrics.cost`), so
    the corpus and what was retrieved from it are counted in one unit and can be divided by each
    other. Bytes are UTF-8, which is what crossed the wire. """
    contents = [doc["content"] or "" for doc in ingested]
    return {
        "corpus_documents": len(contents),
        "corpus_bytes": sum(len(text.encode("utf-8")) for text in contents),
        "corpus_tokens": sum(cost.count_tokens(text) for text in contents),
    }


def _mean_composite(per_unit_scores) -> float | None:
    """The cell's composite, or ``None`` when the benchmark reports no self-contained score.

    All-or-nothing rather than a mean over whichever units happened to report one: a benchmark
    either defines a composite or does not (locomo, longmemeval and beam now do not -- their
    quality metric is the judge's), and averaging a subset would publish a number computed over
    a denominator nobody chose. No units at all is likewise ``None`` rather than 0.0, which is a
    score an engine did not earn.
    """
    if not per_unit_scores:
        return None
    values = [s.get("composite") for s in per_unit_scores]
    if any(v is None for v in values):
        return None
    return sum(values) / len(values)


def build_result(adapter, benchmark, units, per_unit_scores, per_query, *,
                 model, token_budget, receipt, k, repeats, retrieve_summary,
                 latency_metrics, token_metrics, workers=1, judge_workers=DEFAULT_JUDGE_WORKERS,
                 judged_metrics=None, ingest_summary=None,
                 unit_outcomes: list[UnitOutcome] | None = None) -> EvalResult:
    """Build the final :class:`EvalResult` for one (adapter, benchmark) cell.

    ``unit_outcomes`` is one record per unit ATTEMPTED. A failed unit is absent from
    ``per_unit_scores``, so the composite below is a mean over the units that ran and the counts
    beside it are what say how many that was.
    """
    composite = _mean_composite(per_unit_scores)
    outcomes = list(unit_outcomes or [])
    counts = failure_summary(outcomes)
    # Run-level metrics the benchmark reduces from its own per-unit scores. Empty for a benchmark
    # that declares none, so BEAM's artifact is unchanged. Without this, LongMemEval's
    # recall_all@k and LoCoMo's evidence_recall existed only as N per-unit values under
    # `per_unit` -- computed, stored, and unreadable, because `composite` (the one thing that was
    # reduced) is None for both by design.
    rolled = benchmark.rollup(per_unit_scores,
                              ranked=getattr(adapter, "ranks_results", True))
    ctx_tokens = (sum(q["context_tokens"] for q in per_query) / len(per_query)
                  if per_query else 0.0)
    ingested = [
        {"id": d.id, "doc_id": (d.metadata or {}).get("doc_id", d.id), "content": d.content}
        for unit in units for d in unit.documents
    ]
    per_query_dollars = cost.price_per_query(round(ctx_tokens), model)
    corpus = _corpus_size(ingested)
    return EvalResult(
        adapter=adapter.name,
        benchmark=benchmark.name,
        composite=composite,
        per_unit=per_unit_scores,
        per_query=per_query,
        ingested_documents=ingested,
        latency_metrics=latency_metrics,
        retrieve_latency_summary=retrieve_summary,
        ingest_latency_summary=ingest_summary,
        ingest_throughput=_ingest_throughput(ingest_summary),
        token_metrics=token_metrics,
        corpus_documents=corpus["corpus_documents"],
        corpus_bytes=corpus["corpus_bytes"],
        corpus_tokens=corpus["corpus_tokens"],
        # Concurrent (workers>1) runs contend on the backend, so latency is NOT the
        # sequential-idle-host metric; recall is unaffected. Flagged for downstream.
        workers=workers,
        latency_contended=workers > 1,
        # Recorded but deliberately NOT folded into `latency_contended`: judge concurrency spends
        # a provider quota, not the engine under test, so it cannot make an engine's latency
        # percentiles contended. It is here so a five-hour run and a one-hour run of the same cell
        # are distinguishable afterwards.
        judge_workers=judge_workers,
        context_tokens_mean=ctx_tokens,
        est_dollars_per_query=per_query_dollars,
        est_dollars_per_cell=per_query_dollars * len(per_query),
        # What the estimate ASSUMES, recorded beside it. Pricing a regex-extraction engine at
        # gpt-4o-mini's input rate is defensible as "what this context would cost you"; it is
        # misleading only when the assumed model is invisible, which it was.
        cost_basis={"model": model, "kind": "prompt_only",
                    "pricing_table_version": cost.PRICING_TABLE_VERSION,
                    "excludes": ["ingest_extraction", "embeddings",
                                 "answer_generation", "infrastructure"]},
        cleanup_is_destructive=getattr(adapter, "cleanup_is_destructive", None),
        substring_recall_supported=getattr(benchmark, "substring_recall_supported", True),
        composite_rankable=getattr(benchmark, "composite_rankable", True),
        quality_metric=getattr(benchmark, "quality_metric", "substring_recall"),
        question_text_public=getattr(benchmark, "question_text_public", False),
        receipt=receipt.to_dict(),
        n_units=len(units),
        k=k,
        repeats=repeats,
        unit_outcomes=[o.to_dict() for o in outcomes],
        units_total=counts["units_total"],
        units_failed=counts["units_failed"],
        unit_failure_rate=counts["unit_failure_rate"],
        rollup=rolled,
        benchmark_config=_benchmark_params(benchmark),
        judged_metrics=judged_metrics,
    )


def _aggregate_cell(adapter, benchmark, units, per_unit_scores, per_query, *,
                    model, token_budget, receipt, k, repeats, retrieve_summary,
                    latency_metrics, token_metrics, workers=1, judge_workers=DEFAULT_JUDGE_WORKERS,
                    judged_metrics=None, ingest_summary=None,
                    unit_outcomes=None) -> dict[str, Any]:
    """The artifact dict for one cell -- exactly ``build_result(...).to_dict()``, kept as
    a function so the dict form and the typed form can never drift apart."""
    return build_result(
        adapter, benchmark, units, per_unit_scores, per_query,
        model=model, token_budget=token_budget, receipt=receipt, k=k, repeats=repeats,
        retrieve_summary=retrieve_summary, latency_metrics=latency_metrics,
        token_metrics=token_metrics, workers=workers, judge_workers=judge_workers,
        judged_metrics=judged_metrics, ingest_summary=ingest_summary,
        unit_outcomes=unit_outcomes).to_dict()


def _drill_unit(unit, responses, token_budget, budget_mode: str = "matched") -> list[dict[str, Any]]:
    """Build the per-query drill-in: retrieved docs + match + context tokens.

    ``budget_mode`` is the target's ``context_budget``, threaded here for the same reason
    ``_apply_judge`` receives it: an uncapped arm's context is not truncated, and recording the cap
    for it reports a run that did not happen.
    """
    by_id = {r.query_id: r for r in responses}
    rows: list[dict[str, Any]] = []
    for query in unit.queries:
        r = by_id.get(query["id"])
        docs = r.documents if r else []
        match = score_query(spec_from_query(query), docs)
        rows.append({
            "query_id": query["id"],
            # The haystack this question was asked against. Recorded HERE because it is free here
            # and unrecoverable later: a query id belongs to the benchmark's own vocabulary, and
            # nothing downstream can map one back to the unit it came from. The per-question
            # drilldown groups by it (`memrank/analysis/question_digest.py`).
            "unit_id": unit.unit_id,
            "text": query.get("text", ""),
            "category": query.get("category"),
            "hit": match.hit,
            "matched_span": match.matched_span,
            "matched_doc_id": match.matched_doc_id,
            "context_tokens": cost.context_tokens(docs, token_budget, budget_mode),
            # Recorded HERE, not only at judge time, so an unjudged run answers "did the cap
            # bite?" for free. That question is about retrieval and the budget, not about
            # grading, and making it need a judged run made it needlessly expensive to ask.
            #
            # A judged run OVERWRITES this from `_context_text`, which is authoritative because
            # it measures the string actually sent. The two agree except within a token or two at
            # the boundary: this sums per-document counts (matching `context_tokens` beside it),
            # while `_context_text` tokenises the "\n\n"-joined text and so also counts the
            # separators. Same family as the number it sits next to; the exact one wins when it
            # exists.
            "context_truncated": cost.context_truncated(docs, token_budget, budget_mode),
            "retrieved": [
                {"id": d.id, "content": d.content,
                 "score": (d.metadata or {}).get("score")}
                for d in docs
            ],
            "missed_required_spans": (
                [] if match.hit or query.get("kind") == "negative"
                else list(query.get("required_spans") or [])
            ),
        })
    return rows
