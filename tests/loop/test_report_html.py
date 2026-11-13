"""report.html escapes everything it shows and says NOT JUDGED where nothing was graded."""

from __future__ import annotations

import pytest

from memrank.loop.readout import (
    ANSWER_TIME,
    INGESTION_TIME,
    RETRIEVAL_TIME,
    RUNTIME_NOT_RECORDED,
    RUNTIME_NOT_RECORDED_WHY,
    TOTAL_RUNTIME,
)
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
    assert "Retrieval time per question" not in render(result("blue", judged=True))
    plain = result("blue", judged=True)
    question = plain.cases[0].questions[0].model_copy(
        update={"elapsed_ms": 1700.0, "retrieval_ms": 180.0})
    reported = plain.model_copy(update={
        "retrieval": ReportedTiming(p50_ms=180.0, p95_ms=180.0, reported=1, answers=1),
        "cases": [plain.cases[0].model_copy(update={"questions": [question]})]})
    page = render(reported)
    assert ("<th>Retrieval time per question</th><td>180 ms median, 180 ms slow (1 in 20); "
            "reported by the agent, part of the answer time, not added to it</td>") in page
    assert "1.70 s (retrieval 180 ms)" in page


def test_speed_puts_the_total_runtime_first_then_ingestion_then_answer():
    plain = result("blue", judged=True).model_copy(update={"latency_ms": {
        "ask_p50": 7300.0, "ask_p95": 11_800.0, "feed_p50": 42.0, "feed_p95": 60.0}})
    page = render(plain.model_copy(update={"runtime_ms": 724_000.0}))
    speed = page.split("<table id=speed>")[1].split("</table>")[0]
    assert speed.index("<th>Total evaluation runtime</th><td>12 min 4 s</td>") == 4
    ingestion = speed.index("<th>Ingestion time per conversation</th><td>42 ms median, 60 ms slow")
    answer = speed.index("<th>Answer time per question</th><td>7.3 s median, 11.8 s slow (1 in 20)")
    assert ingestion < answer
    assert "typical" not in page and "average" not in page.split("Every question")[0]


def test_a_run_without_a_recorded_runtime_says_not_recorded_and_why():
    page = render(result("blue", judged=True))
    speed = page.split("<table id=speed>")[1].split("</table>")[0]
    assert speed.index(f"<th>Total evaluation runtime</th><td>{RUNTIME_NOT_RECORDED} "
                       f"({RUNTIME_NOT_RECORDED_WHY})</td>") == 4


def test_a_millisecond_retrieval_reads_in_ms_not_rounded_seconds():
    plain = result("blue", judged=True)
    question = plain.cases[0].questions[0].model_copy(
        update={"elapsed_ms": 1700.0, "retrieval_ms": 7.0})
    reported = plain.model_copy(update={
        "retrieval": ReportedTiming(p50_ms=7.0, p95_ms=7.0, reported=1, answers=1),
        "cases": [plain.cases[0].model_copy(update={"questions": [question]})]})
    assert "1.70 s (retrieval 7 ms)" in render(reported)


@pytest.mark.parametrize("retrieval", [None, ReportedTiming(p50_ms=180.0, p95_ms=420.0,
                                                            reported=1, answers=1)])
def test_speed_shows_each_timing_exactly_once(retrieval):
    timed = result("blue", judged=True).model_copy(update={
        "runtime_ms": 97_000.0, "retrieval": retrieval,
        "latency_ms": {"ask_p50": 76.0, "ask_p95": 115.0, "feed_p50": 87.0, "feed_p95": 117.0}})
    speed = render(timed).split("<table id=speed>")[1].split("</table>")[0]
    names = [TOTAL_RUNTIME, INGESTION_TIME, ANSWER_TIME]
    if retrieval is not None:
        names.append(RETRIEVAL_TIME)
    for name in names:
        assert speed.count(f"<th>{name}</th>") == 1, speed
    assert speed.count(f"<th>{RETRIEVAL_TIME}</th>") == (retrieval is not None), speed
    places = [speed.index(f"<th>{name}</th>") for name in names]
    assert places == sorted(places), speed
