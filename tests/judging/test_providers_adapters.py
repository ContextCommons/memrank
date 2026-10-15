"""The ``openai`` and ``anthropic`` adapters, and how their errors reach the rate-limit gate."""

from __future__ import annotations

import json
from functools import partial

import anthropic
import httpx
import pytest

from memrank import rate_limit
from memrank.judging.client import build_completer
from memrank.judging.judge import JUDGE, READER, JudgeConfig
from memrank.judging.providers import ADAPTERS, Endpoint, ProviderError, resolve
from memrank.judging.providers.anthropic import anthropic_send
from memrank.judging.providers.openai import openai_send
from memrank.loop.explain import judge_failure

REPLY = {"choices": [{"index": 0, "finish_reason": "stop",
                      "message": {"role": "assistant", "content": "an answer"}}]}


def _server(*replies: httpx.Response):
    """A mock server answering ``replies`` in order, keeping every request it was sent."""
    sent: list[httpx.Request] = []
    queue = list(replies)

    def handle(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return queue.pop(0)

    return httpx.MockTransport(handle), sent


def _send(transport, spec="openai:gpt-5", endpoints=(), sleeps=None):
    return openai_send(resolve(spec, endpoints), "sk-test", max_tokens=512,
                       transport=transport, sleep=(sleeps if sleeps is not None else []).append)


def test_openai_sends_the_documented_chat_completions_body():
    transport, sent = _server(httpx.Response(200, json=REPLY))
    answer = _send(transport)("gpt-5", "the system", "the user", {"temperature": 0.0})
    assert answer == "an answer"
    request = sent[0]
    assert str(request.url) == "https://api.openai.com/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer sk-test"
    assert json.loads(request.content) == {
        "model": "gpt-5", "max_completion_tokens": 512, "temperature": 0.0,
        "messages": [{"role": "system", "content": "the system"},
                     {"role": "user", "content": "the user"}]}


def test_an_openai_compatible_endpoint_gets_max_tokens_and_no_empty_system():
    router = Endpoint(name="router", adapter="openai", api_base="https://router.invalid/v1",
                      key_ref="K")
    transport, sent = _server(httpx.Response(200, json=REPLY))
    _send(transport, "router:bedrock/claude-haiku-4-5", [router])(
        "bedrock/claude-haiku-4-5", "", "u", {})
    body = json.loads(sent[0].content)
    assert body["max_tokens"] == 512 and "max_completion_tokens" not in body
    assert body["messages"] == [{"role": "user", "content": "u"}]
    assert str(sent[0].url) == "https://router.invalid/v1/chat/completions"


def test_a_rejected_temperature_is_the_providers_error_and_is_not_dropped():
    said = {"error": {"message": "Unsupported parameter: 'temperature'", "code": None}}
    transport, sent = _server(httpx.Response(400, json=said, headers={"x-request-id": "r1"}))
    with pytest.raises(ProviderError) as raised:
        _send(transport)("gpt-5", "s", "u", {"temperature": 0.0})
    error = raised.value
    assert (error.status_code, error.provider, error.request_id) == (400, "openai", "r1")
    assert "temperature" in error.message and len(sent) == 1
    assert not rate_limit.is_rate_limited(error)


def test_transient_errors_are_retried_twice_then_raised():
    sleeps: list[float] = []
    transport, sent = _server(*[httpx.Response(503, text="busy")] * 3)
    with pytest.raises(ProviderError, match="503"):
        _send(transport, sleeps=sleeps)("gpt-5", "s", "u", {})
    assert len(sent) == 3 and sleeps == [1.0, 2.0]


def test_a_transient_error_that_clears_returns_the_answer():
    transport, sent = _server(httpx.Response(504, text="gateway timeout"),
                              httpx.Response(200, json=REPLY))
    assert _send(transport)("gpt-5", "s", "u", {}) == "an answer" and len(sent) == 2


def test_a_429_reaches_the_gate_as_a_rate_limit_with_its_wait():
    said = {"error": {"message": "slow down", "code": "rate_limit_exceeded"}}
    transport, _ = _server(httpx.Response(429, json=said, headers={"retry-after": "7"}))
    with pytest.raises(ProviderError) as raised:
        _send(transport)("gpt-5", "s", "u", {})
    assert rate_limit.is_rate_limited(raised.value)
    assert rate_limit.retry_after_seconds(raised.value) == 7.0


def test_insufficient_quota_is_a_refusal_not_a_rate_limit():
    said = {"error": {"message": "You exceeded your current quota", "type": "insufficient_quota",
                      "code": "insufficient_quota"}}
    transport, sent = _server(httpx.Response(429, json=said))
    with pytest.raises(ProviderError) as raised:
        _send(transport)("gpt-5", "s", "u", {})
    assert raised.value.quota_exhausted and not rate_limit.is_rate_limited(raised.value)
    assert len(sent) == 1


def test_a_reply_with_no_text_raises_with_its_finish_reason():
    empty = {"choices": [{"finish_reason": "length",
                          "message": {"role": "assistant", "content": ""}}]}
    transport, _ = _server(httpx.Response(200, json=empty))
    with pytest.raises(ProviderError, match="finish_reason='length'"):
        _send(transport)("gpt-5", "s", "u", {})


def test_build_completer_sends_each_role_to_its_provider_with_its_sampling(monkeypatch):
    transport, sent = _server(httpx.Response(200, json=REPLY), httpx.Response(200, json=REPLY))
    monkeypatch.setitem(ADAPTERS, "openai", partial(openai_send, transport=transport))
    cfg = JudgeConfig(reader="openai:gpt-5-mini", judge="openai:gpt-5", cache=False)
    complete, counter = build_completer(cfg, keys={"OPENAI_API_KEY": "sk"})
    complete(READER, "s", "u")
    complete(JUDGE, "s", "u")
    bodies = [json.loads(request.content) for request in sent]
    assert [(b["model"], b.get("temperature")) for b in bodies] == [
        ("gpt-5-mini", 0.0), ("gpt-5", None)]
    assert counter() == 2


def _anthropic_refusing(monkeypatch, status: int, message: str) -> None:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    body = {"type": "error", "error": {"type": "x", "message": message}}
    response = httpx.Response(status, request=request, json=body,
                              headers={"retry-after": "3"})

    def create(**_: object) -> None:
        raise anthropic.APIStatusError(message, response=response, body=body)

    class Client:
        def __init__(self, *, api_key: str, base_url: str) -> None:
            self.base_url = base_url
            self.messages = type("Messages", (), {"create": staticmethod(create)})()

    monkeypatch.setattr(anthropic, "Anthropic", Client)


def test_anthropic_refusals_become_provider_errors_the_gate_and_ending_still_read(monkeypatch):
    _anthropic_refusing(monkeypatch, 429, "rate limited")
    send = anthropic_send(resolve("anthropic:claude-haiku-4-5"), "k", max_tokens=512)
    with pytest.raises(ProviderError) as raised:
        send("claude-haiku-4-5", "s", "u", {})
    assert (raised.value.provider, raised.value.status_code) == ("anthropic", 429)
    assert raised.value.message == "rate limited"
    assert rate_limit.is_rate_limited(raised.value)
    assert rate_limit.retry_after_seconds(raised.value) == 3.0
    failure = judge_failure(raised.value, resolve("anthropic:claude-haiku-4-5"))
    assert failure is not None and failure.status == 429


def test_the_builtin_anthropic_names_its_endpoint_explicitly_whatever_the_environment(
        monkeypatch):
    seen: list[str] = []

    class Client:
        def __init__(self, *, api_key: str, base_url: str) -> None:
            seen.append(base_url)

    # The SDK's own override no longer redirects anthropic: -- a stand-in is a named endpoint.
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:8199")
    monkeypatch.setattr(anthropic, "Anthropic", Client)
    anthropic_send(resolve("anthropic:claude-haiku-4-5"), "k", max_tokens=512)
    assert seen == ["https://api.anthropic.com"]
