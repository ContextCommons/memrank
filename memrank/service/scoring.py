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
"""Scores from graded questions: the one place a run's numbers are computed.

Pure functions over :class:`~memrank.service.protocol.RunResult`. The runner builds a result
with every answer recorded and nothing graded; the runner's judge
(:func:`memrank.loop.judge.judge`) calls :func:`apply_verdicts` with the verdicts and gets the
scored result.

A question the agent failed on -- an error, or a case whose reset or feed failed -- scores 0 in
a judged run: a failure is an outcome of the agent, and leaving it out would reward failing on
hard questions. The failure rate is reported beside the score. A question the judge could not
grade is left out of the score and counted as ``unjudged``. A run that is not judged has no
score at all, not even for its failures, and says so in its notice.

Variability is a 95% bootstrap interval that resamples whole cases, because questions about
one conversation are not independent draws (:func:`memrank.instrument.statistics.bootstrap_ci`).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from memrank.instrument.statistics import bootstrap_ci
from memrank.service.protocol import Interval, QuestionOutcome, RunResult

NOT_JUDGED = ("NOT JUDGED: this run was created with judging off. It records answers, failures "
              "and latency; it has no quality score.")
AWAITING_JUDGE = ("NOT JUDGED YET: answers are recorded; the scores appear once they are "
                  "judged (`memrank run --resume <run-id>` finishes judging and the upload).")


def _interval(scored: list[tuple[str, str, float]]) -> Interval:
    """Mean of ``(case_id, question_id, score)`` rows with a case-clustered 95% interval."""
    if not scored:
        return Interval(mean=None, ci95=(None, None), n=0)
    pairs = [(qid, 0.0, score) for _, qid, score in scored]
    groups: dict[str, str | None] = {qid: case_id for case_id, qid, _ in scored}
    mean = sum(score for _, _, score in scored) / len(scored)
    return Interval(mean=mean, ci95=bootstrap_ci(pairs, groups), n=len(scored))


def summarize(result: RunResult) -> RunResult:
    """``result`` with its score, per-category scores and counts recomputed from its questions."""
    rows = [q for case in result.cases for q in case.questions]
    scored = [(q.case_id, q.question_id, q.score, q.category or "uncategorized") for q in rows
              if result.judged and q.status in ("judged", "failed") and q.score is not None]
    by_category: dict[str, list[tuple[str, str, float]]] = defaultdict(list)
    for case_id, qid, score, category in scored:
        by_category[category].append((case_id, qid, score))
    failed = sum(1 for q in rows if q.status == "failed")
    return result.model_copy(update={
        "score": _interval([(c, q, s) for c, q, s, _ in scored]),
        "per_category": {name: _interval(r) for name, r in sorted(by_category.items())},
        "failure_rate": failed / len(rows) if rows else 0.0, "questions": len(rows),
        "failed": failed, "unjudged": sum(1 for q in rows if q.status == "unjudged")})


def _graded(question: QuestionOutcome, verdict: dict[str, Any] | None) -> QuestionOutcome:
    if question.status == "failed":
        return question.model_copy(update={"score": 0.0})
    if verdict is None:
        return question
    return question.model_copy(update={
        "status": verdict["status"], "score": verdict.get("score"),
        "passed": verdict.get("passed"), "rationale": verdict.get("rationale"),
        "error": verdict.get("error")})


def apply_verdicts(result: RunResult, verdicts: dict[str, dict[str, Any]], *,
                   judge_model: str | None, judge_samples: int,
                   judge_prompt_version: str | None = None) -> RunResult:
    """The judged result: each answer's verdict applied, failures scored 0, scores recomputed.

    Args:
        result: The run as the runner recorded it, every answer ``not_judged``.
        verdicts: ``{question_id: {"status", "score", "passed", "rationale", "error"}}`` for
            every answered question.
        judge_model: The model that judged, recorded in the result's identity; None when no
            answer needed one (every grader was deterministic or a program).
        judge_samples: How many gradings each verdict is a vote over.
        judge_prompt_version: The judge prompts' version when the model judged, else None.
    """
    cases = [case.model_copy(update={"questions": [_graded(q, verdicts.get(q.question_id))
                                                   for q in case.questions]})
             for case in result.cases]
    identity = result.identity.model_copy(update={
        "judge_model": judge_model, "judge_samples": judge_samples,
        "judge_prompt_version": judge_prompt_version})
    judged = result.model_copy(update={"cases": cases, "identity": identity, "judged": True,
                                       "notice": None})
    return summarize(judged)
