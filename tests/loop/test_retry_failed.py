"""``memrank run --resume <id> --retry-failed``: a case whose feed failed is run again, the
result and the upload then hold its answers, and only the new answers are judged at a cost."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from memrank.connect.base import AgentError
from memrank.connect.spec import AgentSpec
from memrank.judging.judge import JudgeConfig
from memrank.loop import judge as loop_judge
from memrank.loop import run as loop_run
from memrank.loop.run import RESULT_BEFORE_RETRY, RunOptions, run_agent
from memrank.service.protocol import AgentRef, RunResult, Step
from tests.loop.conftest import OrgApi
from tests.loop.test_run import full_context, quiet
from tests.service.conftest import FakeJudge


class FeedFailsFor:
    """The full-context agent, except that feeding ``case_id`` fails while ``down`` is set."""

    def __init__(self, case_id: str) -> None:
        self.inner = full_context().connector
        self.case_id, self.down = case_id, True

    def reset(self, step: Step) -> None:
        self.inner.reset(step)

    def feed(self, step: Step) -> None:
        if self.down and step.case_id == self.case_id:
            raise AgentError("engine feed failed: out of shared memory", status=502)
        self.inner.feed(step)

    def ask(self, step: Step) -> str | None:
        return self.inner.ask(step)

    def close(self) -> None:
        self.inner.close()


@pytest.fixture
def paid(monkeypatch, tmp_path) -> FakeJudge:
    """The judge the org pays for, behind the real judge cache in a scratch directory."""
    monkeypatch.setenv("MEMRANK_CACHE_DIR", str(tmp_path / "cache"))
    fake = FakeJudge()

    def judge(result: RunResult, api_key, cfg: JudgeConfig, judged=lambda: None) -> RunResult:
        return loop_judge.judge(result, api_key, replace(cfg, completer=fake), judged=judged)

    monkeypatch.setattr(loop_run, "judge", judge)
    return fake


def test_a_failed_case_is_retried_on_request_and_only_new_answers_are_paid(engine, tmp_path,
                                                                           paid):
    org, connector = OrgApi(), FeedFailsFor("c2")
    agent = AgentSpec(ref=AgentRef(name="a"), connector=connector, start=None, description=None)
    first = run_agent(engine, agent, RunOptions(evaluation="locomo"), tmp_path, org.hosted(),
                      quiet([]))
    assert [c.status for c in first.result.cases] == ["complete", "failed"]
    assert paid.calls == 2  # c1's two answers

    resumed = RunOptions(resume=first.result.run_id)
    assert run_agent(engine, agent, resumed, tmp_path, org.hosted(), quiet([])).result == \
        first.result  # without the flag nothing is retried

    connector.down = False
    echoed: list[str] = []
    retried = run_agent(engine, agent, replace(resumed, retry_failed=True), tmp_path,
                        org.hosted(), quiet(echoed)).result
    assert any("Retrying 1 case with failures from a fresh reset, asking only what failed: c2" in line for line in echoed)
    assert [c.status for c in retried.cases] == ["complete", "complete"]
    assert retried.cases[1].restarts == 1 and retried.failed == 0
    assert paid.calls == 3  # c2's one new answer; c1's came from the judge cache
    assert org.uploads[-1] == retried
    before = json.loads((first.folder / RESULT_BEFORE_RETRY).read_text())
    assert before["cases"][1]["status"] == "failed"


def test_retrying_a_run_with_no_failed_case_changes_nothing(engine, tmp_path, paid):
    org = OrgApi()
    first = run_agent(engine, full_context(), RunOptions(evaluation="locomo"), tmp_path,
                      org.hosted(), quiet([]))
    echoed: list[str] = []
    again = run_agent(engine, full_context(),
                      RunOptions(resume=first.result.run_id, retry_failed=True), tmp_path,
                      org.hosted(), quiet(echoed))
    assert again.result == first.result and paid.calls == 3
    assert any("No failed case or answer to retry" in line for line in echoed)
    assert not (first.folder / RESULT_BEFORE_RETRY).exists()


class AskFailsFor:
    """The full-context agent, except that asking ``question_id`` fails while ``down`` is set:
    a provider outage inside a case that otherwise completes. Records every feed and ask."""

    def __init__(self, question_id: str) -> None:
        self.inner = full_context().connector
        self.question_id, self.down = question_id, True
        self.fed: list[str | None] = []
        self.asked: list[str] = []

    def reset(self, step: Step) -> None:
        self.inner.reset(step)

    def feed(self, step: Step) -> None:
        if step.case_id in ("c1", "c2"):  # not the separation probe
            self.fed.append(step.session_id)
        self.inner.feed(step)

    def ask(self, step: Step) -> str | None:
        assert step.question is not None
        if step.case_id in ("c1", "c2"):
            self.asked.append(step.question.id)
        if self.down and step.question.id == self.question_id:
            raise AgentError("router 502: ServiceUnavailableException", status=502)
        return self.inner.ask(step)

    def close(self) -> None:
        self.inner.close()


def test_a_failed_answer_in_a_complete_case_is_asked_again_alone(engine, tmp_path, paid):
    org, connector = OrgApi(), AskFailsFor("c1_q1")
    agent = AgentSpec(ref=AgentRef(name="a"), connector=connector, start=None, description=None)
    first = run_agent(engine, agent, RunOptions(evaluation="locomo"), tmp_path, org.hosted(),
                      quiet([]))
    assert [c.status for c in first.result.cases] == ["complete", "complete"]
    assert first.result.failed == 1 and paid.calls == 2

    connector.down, connector.fed, connector.asked = False, [], []
    retried = run_agent(engine, agent, RunOptions(resume=first.result.run_id, retry_failed=True),
                        tmp_path, org.hosted(), quiet([])).result
    assert connector.asked == ["c1_q1"] and len(connector.fed) == 1  # fed again, one question
    assert connector.fed[0] is not None and connector.fed[0].endswith("-c1-1")
    c1 = retried.cases[0]
    assert c1.restarts == 1 and [q.status for q in c1.questions] == ["judged", "judged"]
    assert retried.failed == 0 and retried.cases[1] == first.result.cases[1]
    assert paid.calls == 3  # only the new answer; c1_q0 and c2_q0 came from the judge cache
    assert org.uploads[-1] == retried
