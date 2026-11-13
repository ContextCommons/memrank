"""``memrank run`` judges each answer as it lands, and its result is the one judging after gave.

The whole loop in process -- engine, lanes, the full-context reference agent -- with the judge
standing in through the grading seam (:mod:`memrank.loop.grading`).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, replace

import pytest

from memrank.judging.judge import JudgeConfig
from memrank.judging.providers import ADAPTERS
from memrank.loop import judge as loop_judge
from memrank.loop.event_log import EVENTS_FILE, EventLog
from memrank.loop.grading_pool import RunUngraded
from memrank.loop.judge import DEFAULT_JUDGE
from memrank.loop.run import RunOptions, run_agent
from memrank.loop.stages import Ceilings
from memrank.service.protocol import Ack, RunResult, StepResult
from tests.loop.conftest import OrgApi, passed, record_keys, run_id_of, stub_grading
from tests.loop.test_run import anthropic_answering, full_context, quiet
from tests.service.conftest import FakeJudge

LOCOMO = RunOptions(evaluation="locomo")
#: A guard against a hang, never a timing assumption: each wait it bounds is asserted.
HANG_GUARD_S = 30.0


@pytest.fixture
def org() -> OrgApi:
    return OrgApi()


def judged_after(engine, run_id: str) -> RunResult:
    """What judging the recorded run whole gives -- the path before answers were judged live."""
    return loop_judge.judge(engine.result(run_id), "org-key",
                            JudgeConfig(judge=DEFAULT_JUDGE, cache=False))


@dataclass
class Releasing:
    """The engine, setting ``done`` once the service holds every step's answer."""

    inner: object
    done: threading.Event

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def post(self, run_id: str, step_id: str, result: StepResult) -> Ack:
        ack = self.inner.post(run_id, step_id, result)
        if self.inner.status(run_id).done:
            self.done.set()
        return ack


def test_a_judge_that_is_held_up_does_not_hold_up_the_agent(engine, tmp_path, org,
                                                            monkeypatch):
    steps_done = threading.Event()

    def held(question):  # graded only once every step is in: a lane that waited would hang
        assert steps_done.wait(HANG_GUARD_S)
        return passed(question)

    stub_grading(monkeypatch, held)
    done = run_agent(Releasing(engine, steps_done), full_context(), LOCOMO, tmp_path,
                     org.hosted(), quiet([]))
    assert done.result.judged and done.result.score.mean == 1.0


class WaitsForAVerdict:
    """The full-context agent, except that asking ``question_id`` waits until a verdict has
    landed: a run that graded only after its last step would never get past it."""

    def __init__(self, question_id: str, graded: threading.Event) -> None:
        self.inner = full_context().connector
        self.question_id, self.graded = question_id, graded

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def ask(self, step):
        if step.question is not None and step.question.id == self.question_id:
            assert self.graded.wait(HANG_GUARD_S)
        return self.inner.ask(step)


def test_answers_are_graded_while_the_agent_is_still_answering(engine, tmp_path, org,
                                                               monkeypatch):
    graded = threading.Event()

    def grade(question):
        verdict = passed(question)
        graded.set()
        return verdict

    stub_grading(monkeypatch, grade)
    agent = replace(full_context(), connector=WaitsForAVerdict("c1_q1", graded))
    done = run_agent(engine, agent, RunOptions(evaluation="locomo", ceilings=Ceilings(ingest=1, answer=1)), tmp_path,
                     org.hosted(), quiet([]))
    assert done.result.judged  # c1_q1 was asked only after c1_q0 had been graded


def test_every_answer_and_verdict_is_logged_and_the_result_is_judging_afters(
        engine, tmp_path, org, monkeypatch):
    monkeypatch.setitem(ADAPTERS, "anthropic", anthropic_answering(FakeJudge()))
    record_keys(monkeypatch, [])
    done = run_agent(engine, full_context(), LOCOMO, tmp_path, org.hosted(), quiet([]))
    assert done.result.runtime_ms is not None  # the run's own clock, which a rejudge has not
    untimed = done.result.model_copy(update={"runtime_ms": None})
    assert untimed == judged_after(engine, done.result.run_id)
    kinds = [event.kind for event in _events(done.folder)]
    assert kinds.count("judging") == 1 and kinds.index("judging") == 0
    assert kinds.count("answer") == 3 and kinds.count("verdict") == 3
    assert org.uploads == [done.result]


def test_a_judge_failing_mid_run_is_retried_at_the_end(engine, tmp_path, org, monkeypatch):
    calls: list[str] = []

    def once_down(question):
        calls.append(question.question_id)
        if calls.count("c1_q1") == 1 and question.question_id == "c1_q1":
            raise ConnectionError("judge went away")
        return passed(question)

    stub_grading(monkeypatch, once_down)
    done = run_agent(engine, full_context(), LOCOMO, tmp_path, org.hosted(), quiet([]))
    assert {q.status for c in done.result.cases for q in c.questions} == {"judged"}
    assert calls.count("c1_q1") == 2


def test_a_judge_down_to_the_end_leaves_the_run_resumable_and_resume_grades_the_rest(
        engine, tmp_path, org, monkeypatch):
    def down_for_c2(question):
        if question.case_id == "c2":
            raise ConnectionError("judge went away")
        return passed(question)

    stub_grading(monkeypatch, down_for_c2)
    echoed: list[str] = []
    with pytest.raises(RunUngraded, match="1 answer could not be graded") as left:
        run_agent(engine, full_context(), LOCOMO, tmp_path, org.hosted(), quiet(echoed))
    assert left.value.ungraded == 1 and org.states[-1] == "stopped" and org.uploads == []
    run_id = run_id_of(echoed)
    assert engine.status(run_id).done  # the agent answered everything

    graded: list[str] = []
    stub_grading(monkeypatch, lambda q: graded.append(q.question_id) or passed(q))
    done = run_agent(engine, full_context(), RunOptions(resume=run_id), tmp_path, org.hosted(),
                     quiet([]))
    assert graded == ["c2_q0"] and done.result.judged and org.uploads == [done.result]


def test_a_run_without_a_log_is_graded_from_its_result_on_resume(engine, tmp_path, org,
                                                                 monkeypatch):
    """A run recorded before answers were logged, or whose runner stopped between an answer and
    its log line: the answers the service holds are logged from its result, then graded."""
    monkeypatch.setitem(ADAPTERS, "anthropic", anthropic_answering(FakeJudge()))
    stub_grading(monkeypatch, lambda q: (_ for _ in ()).throw(ConnectionError("down")))
    echoed: list[str] = []
    with pytest.raises(RunUngraded):
        run_agent(engine, full_context(), LOCOMO, tmp_path, org.hosted(), quiet(echoed))
    run_id = run_id_of(echoed)
    (tmp_path / run_id / EVENTS_FILE).unlink()
    org.events.clear()  # a run from before the log never sent its events either

    record_keys(monkeypatch, [])
    done = run_agent(engine, full_context(), RunOptions(resume=run_id), tmp_path, org.hosted(),
                     quiet([]))
    assert done.result.model_copy(update={"runtime_ms": None}) == judged_after(engine, run_id)
    assert [e.kind for e in _events(done.folder)].count("answer") == 3


def _events(folder):
    return EventLog(folder).read_after(0)
