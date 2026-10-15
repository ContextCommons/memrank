"""The Jev client: the System One request, its retries, its cache and its counter."""

from __future__ import annotations

import json

import httpx
import pytest

from memrank import rate_limit
from memrank.judging.jev import JevClient
from memrank.judging.providers import ProviderError, resolve

TARGET = resolve("typesafe:jev-1.14.0")
ANSWER = {"model": "jev-1.14.0", "answers": {"q": {"type": "noul", "noul": 0.9}},
          "usage": {"input_tokens": 12, "output_tokens": 2}}
QUESTIONS = {"q": {"type": "noul", "instructions": "Does `state` say yes?"}}


class FakeClock:
    """A clock that only moves when something sleeps on it."""

    def __init__(self) -> None:
        self.t = 0.0
        self.slept: list[float] = []

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.t += seconds


def _server(*replies: httpx.Response):
    sent: list[httpx.Request] = []
    queue = list(replies)

    def handle(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return queue.pop(0)

    return httpx.MockTransport(handle), sent


def _client(transport, cache_dir=None, clock=None):
    clock = clock or FakeClock()
    gate = rate_limit.RateLimitGate(now=clock.now, sleep=clock.sleep, jitter=lambda: 0.0)
    return JevClient(TARGET, "ts-test", cache_dir=cache_dir, transport=transport, gate=gate,
                     sleep=clock.sleep), clock


def test_the_request_is_the_documented_systemone_body():
    transport, sent = _server(httpx.Response(200, json=ANSWER,
                                             headers={"x-typesafe-request-id": "req_1"}))
    client, _ = _client(transport)
    decision = client.decide({"a": "yes"}, QUESTIONS, question_set="qs-1")
    request = sent[0]
    assert str(request.url) == "https://api.typesafe.ai/v1/systemone"
    assert request.headers["authorization"] == "Bearer ts-test"
    assert json.loads(request.content) == {"model": "jev-1.14.0", "state": {"a": "yes"},
                                           "questions": QUESTIONS}
    assert (decision.model, decision.request_id, decision.usage["input_tokens"]) == (
        "jev-1.14.0", "req_1", 12)
    assert client.requests == 1


def test_a_429_waits_as_long_as_retry_after_says_for_every_worker():
    transport, sent = _server(httpx.Response(429, headers={"retry-after": "7"}),
                              httpx.Response(200, json=ANSWER))
    client, clock = _client(transport)
    client.decide("s", QUESTIONS, question_set="qs-1")
    assert len(sent) == 2
    assert clock.slept == [7.0]  # the gate's pause, which every worker waits on
    assert client.requests == 1


def test_a_529_overload_is_retried_with_exponential_backoff():
    transport, sent = _server(httpx.Response(529), httpx.Response(529),
                              httpx.Response(200, json=ANSWER))
    client, clock = _client(transport)
    assert client.decide("s", QUESTIONS, question_set="qs-1").answers["q"]["noul"] == 0.9
    assert clock.slept == [1.0, 2.0]
    assert len(sent) == 3


def test_overload_that_never_clears_raises_rather_than_falling_back():
    transport, sent = _server(*[httpx.Response(529, json={"detail": {"message": "busy"}})] * 8)
    client, _ = _client(transport)
    with pytest.raises(ProviderError) as raised:
        client.decide("s", QUESTIONS, question_set="qs-1")
    assert raised.value.status_code == 529 and len(sent) == 8
    assert client.requests == 0


def test_a_refusal_is_a_provider_error_with_the_message_and_request_id():
    body = {"detail": {"error_type": "api_usage_error", "message": "Unknown model: jev-9"}}
    transport, sent = _server(httpx.Response(400, json=body,
                                             headers={"x-typesafe-request-id": "req_9"}))
    client, _ = _client(transport)
    with pytest.raises(ProviderError) as raised:
        client.decide("s", QUESTIONS, question_set="qs-1")
    error = raised.value
    assert (error.status_code, error.message, error.request_id) == (
        400, "Unknown model: jev-9", "req_9")
    assert error.provider == "typesafe" and len(sent) == 1
    assert rate_limit.is_rate_limited(ProviderError("x", provider="typesafe", status_code=429))


def test_a_reply_that_is_not_an_answer_is_refused():
    transport, _ = _server(httpx.Response(200, json={"nothing": True}))
    client, _ = _client(transport)
    with pytest.raises(ProviderError, match="not a System One answer"):
        client.decide("s", QUESTIONS, question_set="qs-1")


def test_the_cache_replays_a_request_and_is_keyed_by_the_question_set(tmp_path):
    transport, sent = _server(httpx.Response(200, json=ANSWER),
                              httpx.Response(200, json=ANSWER))
    client, _ = _client(transport, cache_dir=tmp_path)
    first = client.decide("s", QUESTIONS, question_set="qs-1")
    again = client.decide("s", QUESTIONS, question_set="qs-1")
    assert again == first and len(sent) == 1 and client.requests == 1
    client.decide("s", QUESTIONS, question_set="qs-2")
    assert len(sent) == 2
    stored = sorted(p.name for p in tmp_path.iterdir())
    assert len(stored) == 2 and all(n.startswith("typesafe-") and n.endswith(".json") for n in stored)
    assert all("Does" not in p.read_text() for p in tmp_path.iterdir())  # answers, not prompts
