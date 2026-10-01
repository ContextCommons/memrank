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
"""The judge stage: coverage gate, per-query grading, and the judged-metrics reduce.
"""
from __future__ import annotations

import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from typing import Any

from memrank.errors import MemrankError
from memrank.evaluation.constants import DEFAULT_JUDGE_WORKERS
from memrank.evaluation.observer import NULL_OBSERVER, EvalObserver
from memrank.judging.judge import (
    MAX_VERDICT_ATTEMPTS,
    JudgeConfig,
    JudgedQuery,
    UnparseableVerdict,
)
from memrank.judging.prompts import JUDGE_PROMPT_VERSION
from memrank.judging.shape import GENERIC_BINARY_CATEGORIES, BinaryJudgeShape, JudgeShape
from memrank.metrics import cost


class EmptyJudgeCoverage(MemrankError, ValueError):
    """A judged run has no judgeable queries. Subclasses ValueError so library
    callers can catch either; the CLI translates it to a clean Typer error."""


class AllVerdictsUnparseable(MemrankError):
    """Every gradeable query in a cell failed to produce a parseable verdict.

    Raised rather than reported, because the alternative is the state ATO-1885 found: a run that
    COMPLETES with a normal-looking receipt at `judged_coverage` 0.0, having billed every attempt
    of every grading. Losing one query to a malformed reply is survivable and stays survivable
    (`_judge_one_query`); losing all of them is a broken judge, not a thin result.

    Structural rather than a warning on purpose. The only signal on this path was
    ``observer.warning``, and a run called without an observer gets `NULL_OBSERVER`, whose
    `warning` does nothing -- so a run with nobody listening said nothing at all.
    """


def assert_judge_coverage(units, judge: JudgeConfig | None, benchmark_name: str,
                          shape: JudgeShape) -> None:
    """Refuse a judged run with no judgeable queries. Call this BEFORE any
    backend work (preflight/ingest/retrieve) or judge-client setup so an
    empty-coverage run never burns a live run or builds the Anthropic client.

    ``shape`` is REQUIRED, and used to default to a generic binary shape over
    ``GENERIC_BINARY_CATEGORIES``. That default made a forgotten argument silently become the
    DEMO benchmark's five-category vocabulary, under which every real benchmark's categories read
    as unknown -- so `memrank submit <locomo|longmemeval|beam> --judge` died in preflight with a
    label-bug error pointing at the loader, which was not where the bug was. It is the same
    failure `b84f86b` removed when it retired the global `JUDGE_VALID_CATEGORIES` allowlist: one
    global vocabulary standing in for a benchmark's own. Ask the benchmark.
    """
    if judge is None or judge.allow_empty_coverage:
        return
    if not any(shape.is_judgeable(q) for unit in units for q in unit.queries):
        raise EmptyJudgeCoverage(
            f"--judge has 0 judgeable queries for benchmark '{benchmark_name}' "
            "(no query is gradeable under this benchmark's judge shape). The run "
            "would have no quality signal; pass --allow-empty-judge-coverage to permit.")

def _judge_receipt_config(judge: JudgeConfig | None) -> dict[str, Any]:
    if judge is None:
        return {}
    return {"judge_enabled": True, "answer_model": judge.answer_model,
            "answer_temperature": judge.answer_temperature,
            "judge_model": judge.judge_model, "judge_prompt_version": JUDGE_PROMPT_VERSION,
            "judge_samples": judge.samples, "judge_token_budget": judge.token_budget,
            "judge_no_context_control": judge.no_context_control,
            "judge_cache_enabled": judge.cache,
            "judge_allow_empty_coverage": judge.allow_empty_coverage,
            **judge.answer_prompt.provenance()}


def _judge_cfg_from_receipt(cell: dict[str, Any]) -> JudgeConfig:
    """Rebuild the judge configuration a checkpointed run was using -- the inverse of
    :func:`_judge_receipt_config`.

    The prompt version is CHECKED, not restored: the cache is keyed on it, and re-judging half a
    run under a different prompt would blend two grading regimes into one number while every cached
    grade silently kept the old one.
    """
    config = (cell.get("receipt") or {}).get("config") or {}
    if not config.get("judge_enabled"):
        raise MemrankError(
            "this checkpoint is from an unjudged run, so there is nothing to resume. Re-run it "
            "with --judge rather than grading retrieval that was never meant to be graded.")
    recorded = config.get("judge_prompt_version")
    if recorded != JUDGE_PROMPT_VERSION:
        raise MemrankError(
            f"the checkpoint was retrieved under judge prompt {recorded!r} and this build grades "
            f"with {JUDGE_PROMPT_VERSION!r}. Resuming would mix two grading regimes in one score.")
    return JudgeConfig(
        answer_model=config.get("answer_model", JudgeConfig.answer_model),
        answer_temperature=config.get("answer_temperature", JudgeConfig.answer_temperature),
        judge_model=config.get("judge_model", JudgeConfig.judge_model),
        samples=config.get("judge_samples", 1),
        token_budget=config.get("judge_token_budget", 5000),
        no_context_control=config.get("judge_no_context_control", True),
        cache=config.get("judge_cache_enabled", True),
        allow_empty_coverage=config.get("judge_allow_empty_coverage", False))

def _context_text(retrieved: list[dict[str, Any]], token_budget: int,
                  budget_mode: str = "matched") -> tuple[str, int, bool]:
    """The text handed to the reader, plus how many tokens it was and whether the cap bit.

    ``budget_mode`` comes from the target: every real engine is "matched" (capped at the shared
    --token-budget, the fairness control), the full-context arm is "uncapped". Returning the token
    count and truncation flag makes the cap auditable -- before this, a run could be silently
    context-starved with no trace in the artifact.
    """
    joined = "\n\n".join(d.get("content", "") for d in retrieved)
    full_tokens = cost.count_tokens(joined) if joined else 0
    if budget_mode == "uncapped":
        return joined, full_tokens, False
    truncated = full_tokens > token_budget
    text = cost.truncate_to_tokens(joined, token_budget)
    return text, min(full_tokens, token_budget), truncated


def _judge_one_query(q, row, cfg: JudgeConfig, complete,
                     budget_mode: str, shape: JudgeShape,
                     observer: EvalObserver = NULL_OBSERVER) -> JudgedQuery | None:
    """Grade ONE judgeable query, reporting it against the ``judge`` stage. None if ungradable.

    Timed and counted whatever the outcome, including the unparseable one: the denominator is
    judgeable queries ATTEMPTED, and a query that burned five LLM calls before its verdict came
    back malformed spent exactly as much of the run as one that parsed. Dropping it would leave
    the bar permanently short of a total it could then never reach.
    """
    context, ctx_tokens, ctx_truncated = _context_text(
        row["retrieved"], cfg.token_budget, budget_mode)
    row["context_tokens_sent"] = ctx_tokens
    row["context_truncated"] = ctx_truncated
    started = time.perf_counter()
    try:
        jq = shape.grade(complete, cfg, query=q, context=context)
    except UnparseableVerdict as exc:
        # One query the judge could not be made to grade. Recorded and skipped rather than
        # propagated: a supermemory cell finished all 385 queries, judged nearly all of
        # them, then died on a single reply that quoted the candidate with unescaped
        # quotes -- discarding every result and all the spend behind it. What makes ONE such
        # query survivable rather than silent is `n_unjudged_unparseable_verdict` in the receipt;
        # this warning reaches nobody on the default library path, where `NULL_OBSERVER` swallows
        # it. When NOTHING grades, `_apply_judge` raises rather than leaning on either (ATO-1885).
        observer.warning(f"query {q['id']!r} left unjudged: {exc}")
        return None
    finally:
        observer.item_done("judge", seconds=time.perf_counter() - started)
    row["generated_answer"] = jq.generated_answer
    row["sufficiency"] = None if jq.sufficiency is None else asdict(jq.sufficiency)
    row["correctness"] = asdict(jq.correctness)
    row["answered_without_context"] = jq.answered_without_context
    if jq.nuggets is not None:
        # Raw per-cell artifact only -- every published allowlist drops unknown per-query keys
        # (compare._whitelist_query, leaderboard._safe_query), which is right, since these
        # rationales quote the response. It is here so a human can check WHICH criteria a
        # response missed; an aggregate nobody can audit is how a wrong judge goes unnoticed.
        row["nuggets"] = [asdict(n) for n in jq.nuggets]
    return jq


def _apply_judge(units, per_query, cfg: JudgeConfig, complete,
                 budget_mode: str = "matched",
                 shape: JudgeShape | None = None,
                 judge_workers: int = DEFAULT_JUDGE_WORKERS,
                 observer: EvalObserver = NULL_OBSERVER) -> dict[str, Any]:
    """Grade every judgeable query and reduce the verdicts into metrics.

    Three phases, and the split is what makes concurrency safe rather than merely fast:

      1. CLASSIFY, sequentially and without IO -- which queries are judgeable, and why the rest are
         not. Pure today and kept pure, so the skip accounting can never race.
      2. GRADE, concurrently -- the only IO, and independent per query: `_judge_one_query` mutates
         its own row and nothing else.
      3. REDUCE, sequentially, in the classification's order.

    Because the reduce is ordered and single-threaded, no accumulator needs a lock and the metrics
    are bit-identical at any worker count. That equality is the property under test, not an
    incidental nicety -- a faster judge that quietly returns a different number is worthless.
    """
    if shape is None:
        shape = BinaryJudgeShape(GENERIC_BINARY_CATEGORIES)
    rows = {r["query_id"]: r for r in per_query}
    total = 0
    # Keyed by the shape's own reason, so a shape that grades on something other than a gold
    # answer reports why it skipped rather than borrowing a reason that does not apply to it.
    skipped: dict[str, int] = {}
    unsupported: set[str] = set()
    judgeable: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for unit in units:
        # A unit that FAILED during ingest or retrieval has no drill rows at all, so there is
        # nothing to grade and nothing it can be graded against -- it leaves the judged
        # denominator exactly as it leaves the composite's, rather than counting as N queries
        # the judge declined. Whole-unit, because a MISSING row for a unit that did run is the
        # loader defect the assertion below still catches.
        if unit.queries and not any(q["id"] in rows for q in unit.queries):
            continue
        for q in unit.queries:
            total += 1
            reason = shape.unjudged_reason(q)
            if reason is not None:
                skipped[reason] = skipped.get(reason, 0) + 1
                if reason == "unsupported_category":
                    unsupported.add(q.get("category", "uncategorized"))
                continue
            assert q["id"] in rows, f"judge: query {q['id']!r} not in drill rows"
            judgeable.append((q, rows[q["id"]]))

    # Announced before the first call, not after it: judging is ~15 minutes of silence otherwise,
    # and a run that had not yet published this stage read as frozen at the previous one.
    observer.stage_started("judge")
    graded = _grade_all(judgeable, cfg, complete, budget_mode, shape, judge_workers,
                        observer=observer)

    suff: list[bool] = []
    # Scores, not booleans. A binary shape puts 1.0/0.0 here so the mean is unchanged; a
    # nugget shape puts its per-question mean here, and this loop never learns the difference.
    corr: list[float] = []
    corr_ctx: list[float] = []
    # Per-category scores ride alongside the flat list: these benchmarks' category structure is
    # the point (LoCoMo's sizes span 96-841), and a single micro-mean hides it.
    per_category: dict[str, list[float]] = {}
    # Scores keyed by the grading prompt that produced them, which is NOT the same partition as
    # category: LongMemEval's 30 abstention items keep their PARENT type in `per_category` and
    # appear here under "abstention" as well, because the protocol reports them both ways.
    per_prompt_key: dict[str, list[float]] = {}
    # One row per judged query: did the cap bite, how much text was sent, and how many evidence
    # documents the question needs. The third is what turns a truncation rate into a diagnosis.
    truncation: list[dict[str, Any]] = []
    n_judged = n_unparseable = 0
    for (q, _row), jq in zip(judgeable, graded, strict=True):
        if jq is None:
            n_unparseable += 1
            continue
        truncation.append({"truncated": _row.get("context_truncated"),
                           "tokens_sent": _row.get("context_tokens_sent"),
                           "n_evidence": len(q.get("gold_ids") or ())})
        n_judged += 1
        corr.append(jq.score)
        per_category.setdefault(q.get("category", "uncategorized"), []).append(jq.score)
        per_prompt_key.setdefault(
            q.get("judge_prompt_key") or q.get("category", "uncategorized"), []
        ).append(jq.score)
        if not jq.answered_without_context:
            corr_ctx.append(jq.score)
        if jq.sufficiency is not None:
            suff.append(jq.sufficiency.passed)
    if n_unparseable and not n_judged:
        raise AllVerdictsUnparseable(
            f"the judge ({cfg.judge_model}) returned no parseable verdict for any of the "
            f"{n_unparseable} gradeable queries, after {MAX_VERDICT_ATTEMPTS} attempts each -- "
            "every one of them billed. A judged run with nothing judged is not a result. Check "
            "that this model returns the verdict JSON this build parses (bare, or inside a "
            "single markdown code fence); the incumbent judges do.")
    return _judge_metrics(cfg, suff, corr, corr_ctx, n_judged, skipped,
                          sorted(unsupported), total, n_unparseable,
                          per_category=per_category, n_scoreable=len(judgeable),
                          truncation=truncation,
                          # The shape names any further metrics its protocol publishes. Derived
                          # from the cells above rather than recomputed from verdicts, so the
                          # headline and its decomposition cannot drift apart.
                          extra=shape.aggregates(per_category, per_prompt_key))


def _grade_all(judgeable, cfg: JudgeConfig, complete, budget_mode: str, shape: JudgeShape,
               judge_workers: int,
               observer: EvalObserver = NULL_OBSERVER) -> list[JudgedQuery | None]:
    """Grade the judgeable queries, in input order, on ``judge_workers`` threads.

    ``ThreadPoolExecutor.map`` preserves input order -- the same guarantee `_run_units_concurrent`
    relies on -- so the caller's reduce sees one deterministic sequence whatever the worker count.

    One worker takes the sequential path outright rather than a pool of size one: judging is the
    stage a run dies in, and a default that adds a thread boundary to a code path that does not
    need one is a needless difference between what most runs do and what is easiest to debug.
    """
    def grade(item):
        q, row = item
        return _judge_one_query(q, row, cfg, complete, budget_mode, shape,
                                observer=observer)

    if judge_workers <= 1:
        return [grade(item) for item in judgeable]
    with ThreadPoolExecutor(max_workers=min(judge_workers, len(judgeable) or 1)) as ex:
        return list(ex.map(grade, judgeable))


def _truncation_metrics(truncation: list[dict[str, Any]] | None) -> dict[str, Any]:
    """How often the context cap bit, and whether it bit hardest where evidence is spread out.

    `_judge_one_query` has always recorded `context_truncated` and `context_tokens_sent` per
    query and nothing aggregated them, so a context-starved run left no figure saying so.

    The cross-tab against evidence count is the discriminator between "the memory system missed
    it" and "the cap dropped what retrieval found": on LongMemEval, sessions average 2,290 tokens
    against a 5,000-token cap, while 300 of 470 non-abstention questions need two or more
    evidence sessions. Cells carry their n so a thin bucket is visible as thin.
    """
    rows = truncation or []
    if not rows:
        return {"context_truncation_rate": None, "context_tokens_sent_mean": None,
                "context_truncation_by_evidence_count": {}}
    by_evidence: dict[int, list[bool]] = defaultdict(list)
    for row in rows:
        by_evidence[int(row.get("n_evidence") or 0)].append(bool(row.get("truncated")))
    return {
        "context_truncation_rate": sum(bool(r.get("truncated")) for r in rows) / len(rows),
        "context_tokens_sent_mean": sum(int(r.get("tokens_sent") or 0) for r in rows) / len(rows),
        "context_truncation_by_evidence_count": {
            # String keys: this dict is JSON-serialised into the artifact, where integer keys
            # would come back as strings anyway and compare unequal on reload.
            str(n): {"rate": sum(flags) / len(flags), "n": len(flags)}
            for n, flags in sorted(by_evidence.items())
        },
    }


def _judge_metrics(cfg, suff, corr, corr_ctx, n_judged, skipped: dict[str, int],
                   unsupported_categories, total, n_unparseable=0, *,
                   per_category: dict[str, list[float]] | None = None,
                   n_scoreable: int | None = None,
                   truncation: list[dict[str, Any]] | None = None,
                   extra: dict[str, Any] | None = None) -> dict[str, Any]:
    def mean(xs):
        # `float(x)` rather than counting truthy values: correctness now arrives as a score in
        # [0, 1] while sufficiency is still a bool, and True floats to 1.0, so both read the same
        # here and the binary result is bit-for-bit what counting produced.
        return sum(float(x) for x in xs) / len(xs) if xs else None
    metrics = {
        "retrieval_sufficiency": mean(suff),
        "answer_correctness": mean(corr),
        "n_judged": n_judged,
        "n_unjudged": sum(skipped.values()) + n_unparseable,
        "n_unjudged_no_gold": skipped.get("no_gold", 0),
        "n_unjudged_unsupported_category": skipped.get("unsupported_category", 0),
        # A rubric-scored benchmark skips on a missing RUBRIC, not a missing gold answer. Its own
        # key rather than folded into `no_gold`, which would report a cause that does not exist
        # for that shape and send someone looking for gold answers it never wanted.
        "n_unjudged_no_rubric": skipped.get("no_rubric", 0),
        # Queries the judge could not be made to grade as JSON, after retries. Its own reason
        # rather than being folded into the others: a benchmark gap and a judge malfunction are
        # different problems, and only one of them is worth re-running to fix.
        "n_unjudged_unparseable_verdict": n_unparseable,
        "unsupported_categories": unsupported_categories,
        "judged_coverage": (n_judged / total) if total else 0.0,
        # The explicit denominator: how many queries the shape could grade. A subset run can
        # never masquerade as a full one when the artifact says what "all of them" meant --
        # the numerator/denominator mix-up is the failure behind the field's one retracted
        # LoCoMo number.
        "n_scoreable": n_scoreable if n_scoreable is not None else n_judged + n_unparseable,
        # Cells carry their n so a reader can detect mislabeled or refiltered comparisons --
        # LoCoMo's per-category counts {282, 321, 96, 841} are a fingerprint.
        "answer_correctness_per_category": {
            category: {"mean": mean(scores), "n": len(scores)}
            for category, scores in sorted((per_category or {}).items())
        },
        **_truncation_metrics(truncation),
        **(extra or {}),
    }
    if cfg.no_context_control:
        metrics["answer_correctness_context_dependent"] = mean(corr_ctx)
    return metrics

def judge_cost_estimate(units, cfg, benchmark: str, slice_: str | None = None, *,
                        targets: int = 1, shape: JudgeShape) -> str:
    """What this judged run is about to spend, said before it spends it.

    Reported, never enforced. This used to be ``assert_judge_budget``, refusing a run whose
    ``--max-judge-calls`` was too small; that ceiling is gone. A cap on the JUDGE bounds the
    measurement rather than the system under test -- which queries got graded would depend on where a
    counter ran out, i.e. on ordering -- and no evaluation framework caps its scorer. They bound
    examples (lm-evaluation-harness ``--limit``, HELM ``--max-eval-instances``), which memrank
    spells ``--unit`` and its eval slices. What remains useful is the arithmetic itself, printed the
    way HELM prints estimated token usage: enough to decide whether to start.

    The upper figure is :data:`memrank.judging.judge.MAX_VERDICT_ATTEMPTS` times the plan, since a
    verdict that will not parse is re-asked up to that many times and those retries are real egress.
    It is a loose bound -- answer-generation calls are never re-asked, and a grading that exhausts
    its attempts abandons the rest of its query, so a judge that never parses at all measurably
    spends LESS than the plan.

    ``targets`` is the fan-out width, and it is not decoration: one local sweep shares a single
    counter across every engine, while each submitted target is its own task.
    """
    from memrank.judging.judge import MAX_VERDICT_ATTEMPTS, required_calls

    needed = required_calls(units, cfg, targets=targets, shape=shape)
    where = f"{benchmark}/{slice_}" if slice_ else benchmark
    # The SHAPE decides judgeability, exactly as `required_calls` above and the judging loop do.
    # Asking the module-level predicate here reported "0 judgeable queries" for a rubric-scored
    # benchmark in the same sentence as a 654-call estimate.
    judgeable = [q for unit in units for q in unit.queries if shape.is_judgeable(q)]
    negatives = sum(1 for q in judgeable if q.get("kind") == "negative")
    # Per-query costs read off the SHAPE, and off a REAL query of each polarity. A shape whose
    # cost varies per query -- one call per rubric criterion -- cannot be priced from a bare
    # `{"kind": ...}` stub, which quotes the cost of a question with no criteria at all. The stub
    # survives only as the last resort when a run has no query of that polarity.
    positive = next((q for q in judgeable if q.get("kind") != "negative"), {"kind": "positive"})
    negative = next((q for q in judgeable if q.get("kind") == "negative"), {"kind": "negative"})
    # Judgeable counts, not total queries: quoting all 152 of locomo/smoke taught the reader a
    # rule that over-costs every judged run by 2.2x, and it got written into a repro report.
    detail = (f"{len(judgeable)} judgeable queries "
              f"({len(judgeable) - negatives} positive, {negatives} negative) and judging spends "
              f"{shape.calls_per_query(cfg, positive)} calls per positive and "
              f"{shape.calls_per_query(cfg, negative)} per negative")
    # The split is only worth explaining where it changes the answer. On one target the two
    # halves are just the per-query cost; on a sweep they diverge, and a reader given "5 per
    # positive x 4 engines" would compute a number 40% too high and raise the cap for nothing.
    if targets > 1:
        detail += f", over {targets} engines sharing one counter"
        if cfg.no_context_control and cfg.cache:
            detail += (f" -- of which {shape.control_calls(cfg, positive)} per query is the "
                       "no-context control, cached across engines and so charged once")
    return (f"~{needed} judge calls: {where} has {detail}. Up to "
            f"{needed * MAX_VERDICT_ATTEMPTS} if every verdict has to be re-asked.")
