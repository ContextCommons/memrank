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
"""A run's result as the runner recorded it: every question, every answer, nothing graded yet.

Built from the plan and the step state. Each question carries what grading it needs
(``grading``: the reference answers or rubric, and the keys the benchmark's judge shape reads),
so the result alone is enough to judge it -- which is what the runner does
(:mod:`memrank.loop.judge`, then :func:`memrank.service.scoring.apply_verdicts`).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from memrank import __version__
from memrank.service.cases import Case, Plan
from memrank.service.machine import Recorded, RunState
from memrank.service.protocol import (
    CaseOutcome,
    Identity,
    Interval,
    Op,
    QuestionOutcome,
    RunCreate,
    RunResult,
)
from memrank.service.scoring import AWAITING_JUDGE, NOT_JUDGED, summarize

#: The query keys grading reads besides the question text and category: what a benchmark's
#: judge shape reads, and an evaluation file's grader and choices.
GRADING_KEYS = ("gold_answers", "rubric", "kind", "judge_prompt_key", "grader", "choices")

_NO_SCORE = Interval(mean=None, ci95=(None, None), n=0)


def _asks(state: RunState) -> dict[tuple[str, int], Recorded]:
    return {(r.case_id, r.question_index): r for r in state.recorded.values()
            if r.op is Op.ASK and r.question_index is not None}


def _outcome(case: Case, query: dict[str, Any]) -> QuestionOutcome:
    reference = [str(r) for r in (query.get("gold_answers") or query.get("rubric") or [])]
    return QuestionOutcome(case_id=case.id, question_id=query["id"], question=query["text"],
                           reference=reference, category=query.get("category"),
                           grading={k: query[k] for k in GRADING_KEYS if k in query},
                           status="pending")


def _question(case: Case, index: int, asked: Recorded | None,
              failed_case: str | None) -> QuestionOutcome:
    outcome = _outcome(case, case.queries[index])
    if asked is None:
        if failed_case is not None:
            outcome.status, outcome.error = "failed", failed_case
        return outcome
    outcome.elapsed_ms = asked.result.elapsed_ms
    if asked.result.error is not None:
        outcome.status, outcome.error = "failed", asked.result.error
    else:
        outcome.status, outcome.answer = "not_judged", asked.result.answer
    return outcome


def _case(case: Case, state: RunState) -> CaseOutcome:
    asks = _asks(state)
    failed = state.failed_cases.get(case.id)
    questions = [_question(case, i, asks.get((case.id, i)), failed)
                 for i in range(len(case.queries))]
    pending = any(q.status == "pending" for q in questions)
    status = "failed" if failed else ("pending" if pending else "complete")
    return CaseOutcome(case_id=case.id, status=status, error=failed,
                       restarts=state.restarts.get(case.id, 0), questions=questions)


def _percentile(values: list[float], fraction: float) -> float | None:
    """Nearest-rank percentile; None when there is nothing to rank."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def _latency(state: RunState) -> dict[str, float | None]:
    by_op: dict[Op, list[float]] = defaultdict(list)
    for recorded in state.recorded.values():
        by_op[recorded.op].append(recorded.result.elapsed_ms)
    return {f"{op.value}_{name}": _percentile(by_op[op], fraction)
            for op in (Op.FEED, Op.ASK) for name, fraction in (("p50", 0.5), ("p95", 0.95))}


def _identity(request: RunCreate, plan: Plan) -> Identity:
    return Identity(
        evaluation=plan.evaluation, dataset_version=plan.dataset_version,
        task_version=plan.task_version, seed=request.seed, cases_requested=request.cases,
        questions_per_case=request.questions, case_ids=[c.id for c in plan.cases],
        agent=request.agent, judge_requested=request.judge, judge_model=None, judge_samples=0,
        memrank_version=__version__, evaluation_version=plan.version,
        evaluation_fingerprint=plan.fingerprint, evaluation_source=plan.source)


def build_result(run_id: str, *, request: RunCreate, plan: Plan, state: RunState) -> RunResult:
    """The run as recorded: answers, failures and latency, with no answer graded."""
    result = RunResult(
        run_id=run_id, evaluation=plan.evaluation, agent=request.agent,
        identity=_identity(request, plan), status="complete", judged=False,
        notice=AWAITING_JUDGE if request.judge else NOT_JUDGED, score=_NO_SCORE,
        per_category={}, failure_rate=0.0, questions=0, failed=0, unjudged=0,
        latency_ms=_latency(state), cases=[_case(case, state) for case in plan.cases])
    return summarize(result)
