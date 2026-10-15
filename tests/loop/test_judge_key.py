"""A judged run with no org key asks for one at a terminal, checks it, saves it, and goes on."""

from __future__ import annotations

import json

import anthropic
import httpx
import pytest

from memrank.errors import ActionRequired
from memrank.judging import client as judging_client
from memrank.judging.providers import Endpoint, ProviderError, resolve
from memrank.loop.judge_key import JudgeKeyMissing, KeyPrompt, judge_key

HAIKU = resolve("anthropic:claude-haiku-4-5")
GPT = resolve("openai:gpt-5")
ROUTER = resolve("router:bedrock/claude-haiku-4-5", (Endpoint(
    name="router", adapter="openai", api_base="https://llm.internal.example/v1",
    key_ref="ATOMIC_ROUTER_API_KEY", fingerprint="e1@1"),))

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

    refusals = {GOOD: None, BAD: "invalid x-api-key"}
    return KeyPrompt(read=read, verify=lambda target, key: refusals[key],
                     echo=shown.append), shown


@pytest.mark.parametrize("typed", [(GOOD,), (BAD, GOOD)], ids=["accepted", "retry-after-refusal"])
def test_a_typed_key_is_checked_saved_to_the_org_and_used(typed):
    http, saved = api()
    ask, shown = prompt(*typed)
    assert judge_key(http, "acme", HAIKU, ask, rerun) == GOOD
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
        judge_key(http, "acme", HAIKU, prompt(*typed)[0], rerun)
    assert commands(step.value) == (RERUN,) and "No key was saved" in step.value.statement
    assert saved == []


def test_without_a_terminal_nothing_is_asked_and_both_commands_are_given():
    http, saved = api()
    with pytest.raises(JudgeKeyMissing) as step:
        judge_key(http, "acme", HAIKU, None, rerun)
    assert commands(step.value) == ("memrank secrets set ANTHROPIC_API_KEY --org acme", RERUN)
    assert saved == []


def test_a_built_in_providers_key_is_asked_for_by_its_own_name():
    http, saved = api()
    asked: list[str] = []
    ask = KeyPrompt(read=lambda name: asked.append(name) or GOOD,
                    verify=lambda target, key: None, echo=lambda line: None)
    assert judge_key(http, "acme", GPT, ask, rerun) == GOOD
    assert asked == ["OPENAI_API_KEY"] and saved == [{"name": "OPENAI_API_KEY", "value": GOOD}]


def test_a_missing_endpoint_key_is_refused_with_the_steps_that_save_it_never_asked():
    http, saved = api()
    with pytest.raises(JudgeKeyMissing) as step:
        judge_key(http, "acme", ROUTER, prompt(GOOD)[0], rerun)
    assert commands(step.value) == (
        "memrank secrets set ATOMIC_ROUTER_API_KEY --org acme",
        "memrank endpoints add router --adapter openai --base-url <URL> "
        "--key-ref ATOMIC_ROUTER_API_KEY --org acme", RERUN)
    assert "the endpoint 'router'" in step.value.statement and saved == []
    assert "llm.internal.example" not in str(step.value)


def test_a_role_that_may_not_save_is_told_who_can():
    http, _ = api(save_status=403)
    with pytest.raises(ActionRequired, match="Only an owner of acme"):
        judge_key(http, "acme", HAIKU, prompt(GOOD)[0], rerun)


def refusal(status: int, message: str) -> anthropic.APIStatusError:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    body = {"type": "error", "error": {"type": "x", "message": message}}
    response = httpx.Response(status, request=request, json=body)
    return anthropic.APIStatusError(message, response=response, body=body)


def sdk_answering(monkeypatch, answer) -> list[str]:
    """Stand the Anthropic SDK client in for one whose every call goes to ``answer``."""
    keys: list[str] = []

    class Client:
        def __init__(self, *, api_key: str, base_url: str) -> None:
            keys.append(api_key)
            self.messages = type("Messages", (), {"create": staticmethod(answer)})()

    monkeypatch.setattr(anthropic, "Anthropic", Client)
    return keys


@pytest.mark.parametrize(("status", "reason"), [
    (401, "invalid x-api-key"),
    (400, "Your credit balance is too low"),
    (403, "not allowed"),
], ids=["rejected", "unfunded", "forbidden"])
def test_a_key_anthropic_refuses_is_reported_with_its_reason(monkeypatch, status, reason):
    def refuse(**_: object) -> None:
        raise refusal(status, reason)

    sdk_answering(monkeypatch, refuse)
    assert judging_client.key_refusal(HAIKU, "sk") == reason


def test_an_accepted_key_is_one_small_call_and_an_outage_is_not_a_verdict(monkeypatch):
    keys = sdk_answering(monkeypatch, lambda **_: type("Reply", (), {"content": []})())
    assert judging_client.key_refusal(HAIKU, "sk") is None and keys == ["sk"]

    def overloaded(**_: object) -> None:
        raise refusal(529, "overloaded")

    sdk_answering(monkeypatch, overloaded)
    with pytest.raises(ProviderError) as raised:
        judging_client.key_refusal(HAIKU, "sk")
    assert raised.value.status_code == 529
