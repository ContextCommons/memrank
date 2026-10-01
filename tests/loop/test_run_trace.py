"""A run keeps its trace as it goes, and a judge Anthropic refuses ends it as a judge failure.

The loop's half of every ending (:mod:`memrank.loop.ending`): the run's id, link and folder
as soon as they exist, the recorded answers once the steps are done, every failed step counted
by its cause -- and Anthropic's own errors while judging named as :class:`JudgeFailed` rather
than escaping as an SDK exception the CLI would call a bug in memrank.
"""

from __future__ import annotations

import anthropic
import httpx
import pytest

from memrank.loop import run as loop_run
from memrank.loop.explain import JudgeFailed
from memrank.loop.run import run_agent
from memrank.loop.trace import Stage
from tests.loop.test_run import LOCOMO, CreditTooLow, full_context, org, quiet  # noqa: F401

NO_CREDIT = "the agent's model provider refused the calls (credit balance too low)"
CREDIT = {"type": "error", "error": {"type": "invalid_request_error",
                                     "message": "Your credit balance is too low"}}


def only_the_check_answers(model: str, system: str, user: str) -> str:
    """An agent whose account ran dry after the check: every real question is refused."""
    if "secret code word" in user:
        return "I don't know"
    raise CreditTooLow("Your credit balance is too low to access the Anthropic API.")


def refusing_judge(result, api_key, cfg, judged=lambda: None):
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(400, json=CREDIT, request=request)
    raise anthropic.BadRequestError("Error code: 400", response=response, body=CREDIT)


def test_a_refused_judge_ends_the_run_as_a_judge_failure_with_its_trace(
        engine, tmp_path, org, monkeypatch):  # noqa: F811 - the fixture
    monkeypatch.setattr(loop_run, "judge", refusing_judge)
    tools = quiet([])
    with pytest.raises(JudgeFailed) as failed:
        run_agent(engine, full_context(only_the_check_answers), LOCOMO, tmp_path,
                  org.hosted(), tools)
    assert (failed.value.status, failed.value.reason) == (400, "credit balance too low")
    assert "Your credit balance is too low" in failed.value.details[0]
    trace = tools.trace
    assert trace.stage is Stage.JUDGING and trace.run_id and trace.url
    assert trace.folder == tmp_path / trace.run_id
    assert trace.recorded is not None and trace.recorded.failed == trace.recorded.questions
    assert dict(trace.failures) == {NO_CREDIT: trace.recorded.failed}
    assert org.states[-1] == "stopped"


def test_any_other_judge_failure_is_left_as_it_is(engine, tmp_path, org, monkeypatch):  # noqa: F811
    def broken(result, api_key, cfg, judged=lambda: None):
        raise ValueError("a bug in the judge")

    monkeypatch.setattr(loop_run, "judge", broken)
    with pytest.raises(ValueError, match="a bug in the judge"):
        run_agent(engine, full_context(), LOCOMO, tmp_path, org.hosted(), quiet([]))
