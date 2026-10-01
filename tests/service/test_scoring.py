"""Scores from verdicts: failures count as wrong, unjudged is left out, intervals by case."""

from __future__ import annotations

import pytest

from memrank.service.protocol import (
    AgentRef,
    CaseOutcome,
    Identity,
    Interval,
    QuestionOutcome,
    RunResult,
)
from memrank.service.scoring import apply_verdicts, summarize

PASS = {"status": "judged", "score": 1.0, "passed": True, "rationale": "ok"}
FAIL = {"status": "judged", "score": 0.0, "passed": False, "rationale": "no"}
UNPARSEABLE = {"status": "unjudged", "error": "judge replied nonsense"}


def question(case: str, qid: str, category: str, *, error: str | None = None) -> QuestionOutcome:
    return QuestionOutcome(case_id=case, question_id=qid, question="?", reference=["x"],
                           category=category, status="failed" if error else "not_judged",
                           answer=None if error else "x", error=error)


def recorded() -> RunResult:
    agent = AgentRef(name="a")
    identity = Identity(evaluation="locomo", dataset_version="d", task_version=2, seed=0,
                        cases_requested=None, questions_per_case=None, case_ids=["c1", "c2"],
                        agent=agent, judge_requested=True, judge_model=None, judge_samples=0,
                        memrank_version="0")
    cases = [CaseOutcome(case_id="c1", status="complete",
                         questions=[question("c1", "q1", "temporal"),
                                    question("c1", "q2", "temporal")]),
             CaseOutcome(case_id="c2", status="complete",
                         questions=[question("c2", "q3", "single-hop"),
                                    question("c2", "q4", "single-hop", error="timeout")])]
    return summarize(RunResult(
        run_id="r", evaluation="locomo", agent=agent, identity=identity, status="complete",
        judged=False, notice="pending", score=Interval(mean=None, ci95=(None, None), n=0),
        per_category={}, failure_rate=0.0, questions=0, failed=0, unjudged=0, latency_ms={},
        cases=cases))


def test_an_unjudged_run_has_counts_and_no_score():
    result = recorded()
    assert result.score.mean is None and result.per_category == {}
    assert (result.questions, result.failed, result.failure_rate) == (4, 1, 0.25)


@pytest.mark.parametrize(("verdicts", "mean", "n", "unjudged"), [
    ({"q1": PASS, "q2": PASS, "q3": PASS}, 3 / 4, 4, 0),
    ({"q1": PASS, "q2": FAIL, "q3": FAIL}, 1 / 4, 4, 0),
    ({"q1": PASS, "q2": UNPARSEABLE, "q3": PASS}, 2 / 3, 3, 1),
], ids=["failure-scores-zero", "wrong-answers", "unparseable-left-out"])
def test_verdicts_score_the_run(verdicts, mean, n, unjudged):
    scored = apply_verdicts(recorded(), verdicts, judge_model="j", judge_samples=1)
    assert scored.judged and scored.notice is None and scored.identity.judge_model == "j"
    assert scored.score.mean == pytest.approx(mean) and scored.score.n == n
    assert scored.unjudged == unjudged
    assert set(scored.per_category) == {"temporal", "single-hop"}


def test_applying_the_same_verdicts_twice_changes_nothing():
    verdicts = {"q1": PASS, "q2": FAIL, "q3": PASS}
    once = apply_verdicts(recorded(), verdicts, judge_model="j", judge_samples=1)
    assert apply_verdicts(once, verdicts, judge_model="j", judge_samples=1) == once
