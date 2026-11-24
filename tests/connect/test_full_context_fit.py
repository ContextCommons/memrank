"""The full-context agent answers from the most recent history the reader's input limit admits."""

from __future__ import annotations

import httpx
import pytest

from memrank.benchmarks.answer_prompts import LOCOMO_READER
from memrank.judging.providers import usage
from memrank.judging.providers.errors import ProviderError
from memrank.reference.fit import MAX_FIT_ROUNDS
from memrank.reference.full_context import create_app
from tests.asgi_transport import asgi_transport

SESSION = "conv-1"
QUESTION = "What colour is the car?"
TURN_CHARS = 1000


def provider(limit: int, calls: list[tuple[str, str]]):
    """A reader that counts one token per character and refuses, as Bedrock through the router
    does, a prompt over ``limit``; an accepted call is billed its prompt's length."""

    def complete(model: str, system: str, user: str) -> str:
        calls.append((system, user))
        tokens = len(system) + len(user)
        if tokens > limit:
            raise ProviderError(
                f"invalid request: ValidationException: The model returned the following "
                f"errors: prompt is too long: {tokens} tokens > {limit} maximum",
                provider="router", status_code=400)
        usage.record("router", "claude-haiku-4-5", tokens, 3)
        return "blue" if "blue" in user else "I don't know"

    return complete


def client(complete) -> httpx.Client:
    app = create_app(complete, "m", LOCOMO_READER, provider="router", key_name="ROUTER_KEY")
    return httpx.Client(transport=asgi_transport(app), base_url="http://agent")


def feed(agent: httpx.Client, messages: list[dict[str, str]]) -> None:
    reply = agent.post("/v1/chat/completions", json={
        "messages": messages, "user": SESSION, "metadata": {"memrank_op": "feed"}})
    assert reply.status_code == 200


def ask(agent: httpx.Client):
    return agent.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": QUESTION}], "user": SESSION})


def long_history() -> list[dict[str, str]]:
    """Two dated sessions of ten turns each; the newest turn says the car is blue."""
    messages = []
    for session in range(2):
        messages.append({"role": "system", "content": f"Session date: 2024-0{session + 1}-01"})
        for turn in range(10):
            text = f"s{session}t{turn} " + "x" * (TURN_CHARS - 5)
            messages.append({"role": "user", "content": text})
    messages[-1]["content"] = "The car is blue."
    return messages


def test_a_history_that_fits_is_sent_exactly_as_before():
    calls: list[tuple[str, str]] = []
    agent = client(provider(10**9, calls))
    feed(agent, [{"role": "system", "content": "Session date: 2024-01-01"},
                 {"role": "user", "content": "I bought a car."},
                 {"role": "assistant", "content": "Nice."},
                 {"role": "system", "content": "Session date: 2024-02-01"},
                 {"role": "user", "content": "I painted it blue."}])
    assert ask(agent).json()["choices"][0]["message"]["content"] == "blue"
    # The context format every finished run was answered with, written out rather than derived.
    context = ("[Session date: 2024-01-01]\nI bought a car.\nNice.\n\n"
               "[Session date: 2024-02-01]\nI painted it blue.")
    assert calls == [LOCOMO_READER.render(question=QUESTION, context=context, query_date=None)]


def test_a_too_long_history_is_cut_from_the_oldest_end_at_a_turn(capsys):
    calls: list[tuple[str, str]] = []
    agent = client(provider(12_000, calls))
    feed(agent, long_history())
    reply = ask(agent)
    assert reply.status_code == 200
    assert reply.json()["choices"][0]["message"]["content"] == "blue"
    first, answered = calls[0][1], calls[-1][1]
    assert 1 < len(calls) <= MAX_FIT_ROUNDS + 1
    assert "s0t0 " in first and "s0t0 " not in answered  # the oldest turn went
    assert "The car is blue." in answered and QUESTION in answered
    kept = [line for line in answered.splitlines() if line.startswith(("s0t", "s1t"))]
    assert all(len(line) == TURN_CHARS for line in kept)  # whole turns, never part of one
    assert answered.index("[Session date:") < answered.index(kept[0])  # its date kept with it
    assert "dropped the oldest" in capsys.readouterr().err


def test_the_usage_reported_is_the_answered_request():
    calls: list[tuple[str, str]] = []
    agent = client(provider(12_000, calls))
    feed(agent, long_history())
    billed = ask(agent).json()["memrank_usage"]
    system, user = calls[-1]
    assert billed == [{"component": "reader", "provider": "router",
                       "model": "claude-haiku-4-5", "input_tokens": len(system) + len(user),
                       "output_tokens": 3}]


def test_a_history_that_still_does_not_fit_fails_loudly():
    calls: list[tuple[str, str]] = []
    agent = client(provider(100, calls))  # not even the question and instruction fit
    feed(agent, long_history())
    reply = ask(agent)
    assert reply.status_code == 502
    assert "prompt is too long" in reply.json()["detail"]["message"]
    assert len(calls) <= MAX_FIT_ROUNDS + 1


@pytest.mark.parametrize("status", [400, 500])
def test_any_other_refusal_is_not_retried(status):
    calls: list[tuple[str, str]] = []

    def refuses(model: str, system: str, user: str) -> str:
        calls.append((system, user))
        raise ProviderError("invalid request: something else", provider="router",
                            status_code=status)

    agent = client(refuses)
    feed(agent, long_history())
    assert ask(agent).status_code == 502
    assert len(calls) == 1


def test_the_same_history_and_question_are_cut_the_same_way():
    answered = []
    for _ in range(2):
        calls: list[tuple[str, str]] = []
        agent = client(provider(12_000, calls))
        feed(agent, long_history())
        ask(agent)
        answered.append(calls)
    assert answered[0] == answered[1]
