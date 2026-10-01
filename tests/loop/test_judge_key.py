"""A judged run with no org key asks for one at a terminal, checks it, saves it, and goes on."""

from __future__ import annotations

import json

import anthropic
import httpx
import pytest

from memrank.errors import ActionRequired
from memrank.judging import client as judging_client
from memrank.loop.judge_key import JudgeKeyMissing, KeyPrompt, judge_key

RERUN = "memrank run locomo --agent full-context --no-judge"


def commands(step) -> tuple[str, ...]:
    """Every command a step offers, in the order its steps give them."""
    return tuple(command for part in step.steps for command in part.commands)


def rerun() -> str:
    """The rerun line, built on demand the way the CLI builds it."""
    return RERUN
GOOD, BAD = "sk-good", "sk-bad"


def api(save_status: int = 201) -> tuple[httpx.Client, list[dict]]:
    """An org with no saved key; every save is recorded, and answered with ``save_status``."""
    saved: list[dict] = []

    def handle(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            saved.append(json.loads(request.content))
            return httpx.Response(save_status, json={"detail": "refused"})
        return httpx.Response(200, json={})

    return httpx.Client(base_url="http://api", transport=httpx.MockTransport(handle)), saved


def prompt(*typed: str | type[BaseException]) -> tuple[KeyPrompt, list[str]]:
    """What the user types, in order (an exception type is raised instead); what they see."""
    answers, shown = iter(typed), []

    def read(label: str) -> str:
        answer = next(answers)
        if isinstance(answer, type):
            raise answer()
        return answer

    verify = {GOOD: None, BAD: "invalid x-api-key"}.get
    return KeyPrompt(read=read, verify=verify, echo=shown.append), shown


@pytest.mark.parametrize("typed", [(GOOD,), (BAD, GOOD)], ids=["accepted", "retry-after-refusal"])
def test_a_typed_key_is_checked_saved_to_the_org_and_used(typed):
    http, saved = api()
    ask, shown = prompt(*typed)
    assert judge_key(http, "acme", ask, rerun) == GOOD
    assert saved == [{"name": "ANTHROPIC_API_KEY", "value": GOOD}]
    assert shown[0].startswith("acme has no Anthropic key yet")
    assert shown[-1] == "Saved ANTHROPIC_API_KEY to acme."
    assert ("Anthropic did not accept that key: invalid x-api-key" in shown) is (BAD in typed)
    assert not any(GOOD in line or BAD in line for line in shown)  # the key is never echoed


@pytest.mark.parametrize("typed", [("",), ("   ",), (KeyboardInterrupt,), (BAD, "")],
                         ids=["empty", "blank", "ctrl-c", "empty-after-refusal"])
def test_cancelling_saves_nothing_and_offers_the_no_judge_rerun(typed):
    http, saved = api()
    with pytest.raises(ActionRequired) as step:
        judge_key(http, "acme", prompt(*typed)[0], rerun)
    assert commands(step.value) == (RERUN,) and "No key was saved" in step.value.statement
    assert saved == []


def test_without_a_terminal_nothing_is_asked_and_both_commands_are_given():
    http, saved = api()
    with pytest.raises(JudgeKeyMissing) as step:
        judge_key(http, "acme", None, rerun)
    assert commands(step.value) == ("memrank secrets set ANTHROPIC_API_KEY --org acme", RERUN)
    assert saved == []


def test_a_role_that_may_not_save_is_told_who_can():
    http, _ = api(save_status=403)
    with pytest.raises(ActionRequired, match="Only an owner of acme"):
        judge_key(http, "acme", prompt(GOOD)[0], rerun)


def refusal(status: int, message: str) -> anthropic.APIStatusError:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    body = {"type": "error", "error": {"type": "x", "message": message}}
    response = httpx.Response(status, request=request, json=body)
    return anthropic.APIStatusError(message, response=response, body=body)


@pytest.mark.parametrize(("status", "reason"), [
    (401, "invalid x-api-key"),
    (400, "Your credit balance is too low"),
    (403, "not allowed"),
], ids=["rejected", "unfunded", "forbidden"])
def test_a_key_anthropic_refuses_is_reported_with_its_reason(monkeypatch, status, reason):
    def refuse(model: str, system: str, user: str) -> str:
        raise refusal(status, reason)

    monkeypatch.setattr(judging_client, "anthropic_completer", lambda cfg, api_key: refuse)
    assert judging_client.key_refusal("sk") == reason


def test_an_accepted_key_is_one_small_call_and_an_outage_is_not_a_verdict(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(judging_client, "anthropic_completer",
                        lambda cfg, api_key: lambda model, system, user: calls.append(api_key))
    assert judging_client.key_refusal("sk") is None and calls == ["sk"]

    def overloaded(model: str, system: str, user: str) -> str:
        raise refusal(529, "overloaded")

    monkeypatch.setattr(judging_client, "anthropic_completer", lambda cfg, api_key: overloaded)
    with pytest.raises(anthropic.APIStatusError):
        judging_client.key_refusal("sk")
