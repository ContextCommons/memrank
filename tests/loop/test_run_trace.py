"""A run keeps its trace as it goes; a judge Anthropic refuses leaves it ungraded, ended as
a judge failure.

The loop's half of every ending (:mod:`memrank.loop.ending`): the run's id, link and folder
as soon as they exist, the recorded answers once the steps are done, every failed step counted
by its cause -- and Anthropic's own errors while judging named as :class:`JudgeFailed` rather
than escaping as an SDK exception the CLI would call a bug in memrank.
"""

from __future__ import annotations

import pytest

from memrank.judging.providers import ProviderError
from memrank.loop.explain import JudgeFailed
from memrank.loop.grading_pool import RunUngraded
from memrank.loop.run import run_agent
from memrank.loop.trace import Stage
from tests.loop.conftest import stub_grading
from tests.loop.test_run import LOCOMO, CreditTooLow, full_context, org, quiet  # noqa: F401

NO_CREDIT = "the agent's model provider refused the calls (credit balance too low)"
CREDIT = {"type": "error", "error": {"type": "invalid_request_error",
                                     "message": "Your credit balance is too low"}}


def only_the_check_answers(model: str, system: str, user: str) -> str:
    """An agent whose account ran dry after the check: every real question is refused."""
    if "secret code word" in user:
        return "I don't know"
    raise CreditTooLow("Your credit balance is too low to access the Anthropic API.")


def refusing(question):
    raise ProviderError(CREDIT["error"]["message"], provider="anthropic", status_code=400)


def test_a_refused_judge_leaves_the_run_ungraded_and_ends_as_a_judge_failure(
        engine, tmp_path, org, monkeypatch):  # noqa: F811 - the fixture
    stub_grading(monkeypatch, refusing)
    tools = quiet([])
    with pytest.raises(RunUngraded) as ungraded:
        run_agent(engine, full_context(), LOCOMO, tmp_path, org.hosted(), tools)
    failed = ungraded.value.__cause__
    assert isinstance(failed, JudgeFailed)
    assert (failed.status, failed.reason) == (400, "credit balance too low")
    assert "Your credit balance is too low" in failed.details[0]
    assert ungraded.value.ungraded == 3 and "memrank run --resume" in str(ungraded.value)
    trace = tools.trace
    assert trace.stage is Stage.JUDGING and trace.run_id and trace.url
    assert trace.folder == tmp_path / trace.run_id
    assert trace.recorded is not None and trace.recorded.failed == 0
    assert org.states[-1] == "stopped"


def test_an_agent_failing_every_question_is_traced_by_its_cause(
        engine, tmp_path, org):  # noqa: F811 - the fixture
    tools = quiet([])
    run_agent(engine, full_context(only_the_check_answers), LOCOMO, tmp_path, org.hosted(),
              tools)
    trace = tools.trace
    assert trace.recorded is not None and trace.recorded.failed == trace.recorded.questions
    assert dict(trace.failures) == {NO_CREDIT: trace.recorded.failed}


def test_any_other_grading_error_is_kept_as_the_cause(engine, tmp_path, org, monkeypatch):  # noqa: F811
    def broken(question):
        raise ValueError("a bug in the judge")

    stub_grading(monkeypatch, broken)
    with pytest.raises(RunUngraded, match="First error: a bug in the judge") as ungraded:
        run_agent(engine, full_context(), LOCOMO, tmp_path, org.hosted(), quiet([]))
    assert isinstance(ungraded.value.__cause__, ValueError)
