"""report.html escapes everything it shows and says NOT JUDGED where nothing was graded."""

from __future__ import annotations

from memrank.loop.report_html import render
from memrank.service.protocol import (
    AgentRef,
    CaseOutcome,
    Identity,
    Interval,
    QuestionOutcome,
    ReportedTiming,
    RunResult,
)

EMPTY = Interval(mean=None, ci95=(None, None), n=0)


def result(answer: str, judged: bool) -> RunResult:
    question = QuestionOutcome(case_id="c1", question_id="q1", question="What <b>?</b>",
                               reference=["x"], category="temporal",
                               status="judged" if judged else "not_judged", answer=answer,
                               score=1.0 if judged else None, passed=judged or None)
    agent = AgentRef(name="a")
    identity = Identity(evaluation="locomo", dataset_version="d", task_version=2, seed=0,
                        cases_requested=1, questions_per_case=None, case_ids=["c1"],
                        agent=agent, judge_requested=True,
                        judge_model="j" if judged else None, judge_samples=1,
                        memrank_version="0")
    return RunResult(run_id="r", evaluation="locomo", agent=agent, identity=identity,
                     status="complete", judged=judged,
                     notice=None if judged else "NOT JUDGED: off", score=EMPTY,
                     per_category={}, failure_rate=0.0, questions=1, failed=0, unjudged=0,
                     latency_ms={}, cases=[CaseOutcome(case_id="c1", status="complete",
                                                       questions=[question])])


def test_agent_text_is_escaped_never_rendered():
    page = render(result("<script>alert(1)</script>", judged=True))
    assert "<script>" not in page and "&lt;script&gt;" in page
    assert "What &lt;b&gt;?&lt;/b&gt;" in page


def test_an_unjudged_run_says_so_first_and_grades_nothing():
    page = render(result("blue", judged=False))
    assert page.index("NOT JUDGED") < page.index("Every question")
    assert "correct" not in page.split("Every question")[1]


def test_the_page_fetches_nothing():
    page = render(result("blue", judged=True))
    assert "src=" not in page and "href=" not in page and "@import" not in page


def test_retrieval_appears_only_when_the_agent_reported_it():
    assert "etrieval" not in render(result("blue", judged=True))
    plain = result("blue", judged=True)
    question = plain.cases[0].questions[0].model_copy(
        update={"elapsed_ms": 1700.0, "retrieval_ms": 180.0})
    reported = plain.model_copy(update={
        "retrieval": ReportedTiming(p50_ms=180.0, p95_ms=180.0, reported=1, answers=1),
        "cases": [plain.cases[0].model_copy(update={"questions": [question]})]})
    page = render(reported)
    assert "Retrieval 180 ms typical, 180 ms slow, reported by your agent, 1 of 1 questions" in page
    assert "1.70 s (retrieval 180 ms)" in page and "<th>Retrieval time</th>" in page


def test_a_millisecond_retrieval_reads_in_ms_not_rounded_seconds():
    plain = result("blue", judged=True)
    question = plain.cases[0].questions[0].model_copy(
        update={"elapsed_ms": 1700.0, "retrieval_ms": 7.0})
    reported = plain.model_copy(update={
        "retrieval": ReportedTiming(p50_ms=7.0, p95_ms=7.0, reported=1, answers=1),
        "cases": [plain.cases[0].model_copy(update={"questions": [question]})]})
    assert "1.70 s (retrieval 7 ms)" in render(reported)
