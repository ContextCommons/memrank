"""A failure on the agent's side leads with whose side it is on, then keeps the raw detail."""

from __future__ import annotations

import pytest

from memrank.agents import shipped
from memrank.connect.base import AgentError
from memrank.connect.spec import AgentSpec
from memrank.loop import explain
from memrank.service.protocol import AgentRef

KEY = "ANTHROPIC_API_KEY"
REFUSED = {"detail": {"code": "provider_refused", "provider": "anthropic", "status": 400,
                      "key_name": KEY, "message": "BadRequestError: credit balance is too low"}}
UNANSWERED = {"detail": {"code": "provider_failed", "provider": "anthropic", "status": None,
                         "key_name": KEY, "message": "APIConnectionError: Connection error."}}


def agent(path: str | None) -> AgentSpec:
    return AgentSpec(ref=AgentRef(name="a", spec_path=path), connector=None,  # type: ignore[arg-type]
                     start=None, description=None)


SHIPPED = str(shipped()["full-context"])

#: (case, agent file, where the key is, the agent's status and body, what the lead must say)
CASES = [
    ("shipped-agent-key-in-env", SHIPPED, "env", 502, REFUSED,
     ("refused by its provider (anthropic, 400)", f"the {KEY} on this machine, read from "
      "the environment", "credit, permissions")),
    ("shipped-agent-key-in-wallet", SHIPPED, "wallet", 502, REFUSED,
     ("refused by its provider", "read from memrank's wallet (")),
    ("shipped-agent-provider-never-answered", SHIPPED, "env", 502, UNANSWERED,
     ("failed before its provider answered (anthropic)",)),
    ("your-agent-refused", "/work/my-agent.yaml", "env", 502, REFUSED,
     ("refused by its provider", "your agent calls its model with")),
    ("your-agent-http-error", "/work/my-agent.yaml", "env", 500, {"error": "boom"},
     ("answered with an error (500)", "the agent's side, not memrank's")),
    ("a-command-that-exited", "/work/my-agent.yaml", "env", None, None,
     ("The agent's call failed", "the agent's side, not memrank's")),
]


@pytest.mark.parametrize(("path", "where", "status", "body", "says"),
                         [case[1:] for case in CASES], ids=[case[0] for case in CASES])
def test_the_lead_says_whose_side_and_the_detail_is_kept(monkeypatch, tmp_path, path, where,
                                                         status, body, says):
    monkeypatch.delenv(KEY, raising=False)
    if where == "env":
        monkeypatch.setenv(KEY, "sk-secret-value")
    else:
        monkeypatch.setattr("memrank.secrets.wallet.get", lambda name: "sk-secret-value")
    failure = AgentError("POST /v1/chat/completions returned 502: {...raw payload...}",
                         status=status, body=body)
    lead, detail = explain.explained(failure, agent(path)).split("\n", 1)
    assert all(part in lead for part in says), lead
    assert detail == f"  what it said: {failure}"
    assert "sk-secret-value" not in lead + detail


def status_error(status: int, message: str, kind: str = "invalid_request_error"):
    import anthropic
    import httpx

    body = {"type": "error", "error": {"type": kind, "message": message}}
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(status, json=body, request=request, headers={"request-id": "req_1"})
    return anthropic.APIStatusError(f"Error code: {status}", response=response, body=body)


@pytest.mark.parametrize(("status", "message", "reason"), [
    (400, "Your credit balance is too low to access the Anthropic API.",
     "credit balance too low"),
    (401, "invalid x-api-key", "the key was not accepted"),
    (403, "not allowed", "the key is not allowed to do this"),
    (429, "rate limited", "too many requests, even after waiting"),
    (529, "Overloaded", "the provider is overloaded"),
    (500, "oops", "it failed on its side (500)"),
], ids=["no-credit", "bad-key", "forbidden", "rate-limited", "overloaded", "server"])
def test_a_judge_refusal_is_named_with_anthropics_own_words(status, message, reason):
    failure = explain.judge_failure(status_error(status, message))
    assert failure is not None and (failure.status, failure.reason) == (status, reason)
    assert failure.details == (f"{status} invalid_request_error: {message} (req_1)",)


def test_a_judge_that_never_answered_keeps_the_whole_chain():
    import anthropic
    import httpx

    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    try:
        try:
            raise TypeError("unexpected keyword argument 'proxies'")
        except TypeError as inner:
            raise anthropic.APIConnectionError(request=request) from inner
    except anthropic.APIConnectionError as sdk:
        failure = explain.judge_failure(sdk)
    assert failure is not None and failure.status is None
    assert failure.details[1] == "caused by TypeError: unexpected keyword argument 'proxies'"


def test_anything_else_is_not_a_judge_failure():
    assert explain.judge_failure(ValueError("a bug")) is None


@pytest.mark.parametrize(("status", "body", "cause"), [
    (502, REFUSED, "the agent's model provider refused the calls (credit balance too low)"),
    (502, UNANSWERED, "the agent's model provider did not answer"),
    (500, {"error": "boom"}, "your agent answered with an error (500)"),
    (None, None, "your agent's call failed"),
], ids=["refused", "unanswered", "http-error", "no-status"])
def test_failed_steps_are_counted_by_a_plain_cause(status, body, cause):
    assert explain.cause(AgentError("x", status=status, body=body)) == cause
