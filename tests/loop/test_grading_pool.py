"""The grading pool: never in a lane's way, stale verdicts dropped, one retry, then ungraded.

No test here waits on time. A grader that must stay busy blocks on an event the test sets; the
one bounded wait is a guard against a hang, and its result is asserted.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from memrank.loop.event_log import EventLog
from memrank.loop.grading_pool import GradingPool, JudgingChanged, RunUngraded
from memrank.loop.live_answers import fold
from memrank.service.protocol import JudgeIdentity, QuestionOutcome
from tests.loop.conftest import StubGrading, passed

#: A guard against a hang, never a timing assumption: each wait it bounds is asserted.
HANG_GUARD_S = 30.0
RESUME = "memrank run --resume r1 --agent agent.yaml"


def outcome(qid: str, answer: str = "blue") -> QuestionOutcome:
    return QuestionOutcome(case_id="c1", question_id=qid, question=f"{qid}?",
                           reference=["blue"], category=None, status="not_judged",
                           answer=answer, elapsed_ms=10.0)


def pool(folder: Path, grade=passed, **grading) -> GradingPool:
    started = GradingPool(EventLog(folder), StubGrading(grade, **grading), resume=RESUME)
    started.start()
    return started


def test_handing_an_answer_over_returns_while_its_grading_is_blocked(tmp_path):
    release, grading = threading.Event(), threading.Event()

    def blocked(question):
        grading.set()
        assert release.wait(HANG_GUARD_S)
        return passed(question)

    graders = pool(tmp_path, blocked)
    answer = graders.log.answer(position=0, answer_ref="s1", outcome=outcome("q0"))
    graders.submit(answer)  # returns with the grader still holding the answer
    assert grading.wait(HANG_GUARD_S) and fold(graders.log.read_after(0)).verdicts == {}
    release.set()
    assert graders.finish() == {"q0": {**passed(answer.outcome), "error": None,
                                       "probability": None, "decision": None}}


def test_a_verdict_on_an_answer_replaced_meanwhile_is_dropped(tmp_path):
    release = threading.Event()

    def by_answer(question):
        if question.answer == "red":
            assert release.wait(HANG_GUARD_S)
            return {"status": "judged", "score": 0.0, "passed": False, "rationale": "old"}
        return {"status": "judged", "score": 1.0, "passed": True, "rationale": "new"}

    graders = pool(tmp_path, by_answer)
    graders.submit(graders.log.answer(position=0, answer_ref="s1", outcome=outcome("q0", "red")))
    graders.submit(graders.log.answer(position=0, answer_ref="s7", outcome=outcome("q0")))
    release.set()
    assert graders.finish()["q0"]["rationale"] == "new"
    verdicts = [e for e in graders.log.read_after(0) if e.kind == "verdict"]
    assert [(v.answer_ref, v.rationale) for v in verdicts] == [("s7", "new")]


def test_a_failed_grading_is_retried_once_at_the_end(tmp_path):
    calls: list[str] = []

    def flaky(question):
        calls.append(question.question_id)
        if calls.count(question.question_id) == 1 and question.question_id == "q1":
            raise ConnectionError("judge went away")
        return passed(question)

    graders = pool(tmp_path, flaky)
    for i in range(3):
        graders.submit(graders.log.answer(position=i, answer_ref=f"s{i}",
                                          outcome=outcome(f"q{i}")))
    assert sorted(graders.finish()) == ["q0", "q1", "q2"] and calls.count("q1") == 2


def test_answers_still_ungraded_after_the_retry_leave_the_run_unfinished(tmp_path):
    def refusing(question):
        if question.question_id != "q0":
            raise ConnectionError("judge went away")
        return passed(question)

    graders = pool(tmp_path, refusing)
    for i in range(3):
        graders.submit(graders.log.answer(position=i, answer_ref=f"s{i}",
                                          outcome=outcome(f"q{i}")))
    with pytest.raises(RunUngraded, match=f"(?s)2 answers could not be graded.*`{RESUME}`") as left:
        graders.finish()
    assert left.value.ungraded == 2 and isinstance(left.value.__cause__, ConnectionError)
    assert "First error: judge went away" in str(left.value)


def test_a_resumed_pool_grades_only_what_the_log_left_ungraded(tmp_path):
    first = pool(tmp_path, lambda q: (_ for _ in ()).throw(ConnectionError("down"))
                 if q.question_id in ("q2", "q3") else passed(q))
    for i in range(4):
        first.submit(first.log.answer(position=i, answer_ref=f"s{i}", outcome=outcome(f"q{i}")))
    with pytest.raises(RunUngraded):
        first.finish()
    first.close()

    graded: list[str] = []
    resumed = pool(tmp_path, lambda q: graded.append(q.question_id) or passed(q))
    assert sorted(resumed.finish()) == ["q0", "q1", "q2", "q3"] and sorted(graded) == ["q2", "q3"]
    assert [e.kind for e in resumed.log.read_after(0)].count("judging") == 1


def test_a_resume_under_another_judge_is_refused(tmp_path):
    haiku = JudgeIdentity(judge="anthropic:claude-haiku-4-5", judge_provider="anthropic",
                          judge_model="claude-haiku-4-5")
    pool(tmp_path, identity=haiku, samples=1, prompt_version="v1").close()
    with pytest.raises(JudgingChanged, match="graded by anthropic:claude-haiku-4-5, not its own"):
        pool(tmp_path)
