"""A provider call's billed tokens, as its response says, collected for an agent to report."""

from __future__ import annotations

from types import SimpleNamespace

import anthropic
import httpx

from memrank.judging.providers import resolve
from memrank.judging.providers.anthropic import anthropic_send
from memrank.judging.providers.openai import openai_send
from memrank.judging.providers.usage import collecting, record
from memrank.service.protocol import TokenUsage


def _openai(body: dict):
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=body))
    return openai_send(resolve("openai:gpt-5.4"), "sk", max_tokens=64, transport=transport)


REPLY = {"choices": [{"message": {"content": "blue"}, "finish_reason": "stop"}]}


def test_an_openai_call_records_its_prompt_and_completion_tokens():
    send = _openai({**REPLY, "usage": {"prompt_tokens": 120, "completion_tokens": 4}})
    with collecting("reader") as used:
        assert send("gpt-5.4", "", "q", {}) == "blue"
    assert used.reported() == [TokenUsage(component="reader", provider="openai",
                                          model="gpt-5.4", input_tokens=120, output_tokens=4)]


def test_a_response_with_no_usage_makes_the_step_report_nothing():
    with collecting("reader") as used:
        _openai(REPLY)("gpt-5.4", "", "q", {})
    assert used.reported() is None


def test_outside_a_collection_recording_does_nothing():
    record("openai", "gpt-5.4", 1, 1)  # the judge's calls: nobody is collecting
    with collecting("reader") as used:
        pass
    assert used.reported() == []


def test_an_anthropic_call_counts_cached_input_as_input(monkeypatch):
    billed = SimpleNamespace(input_tokens=100, output_tokens=7, cache_creation_input_tokens=20,
                             cache_read_input_tokens=300)

    class Client:
        def __init__(self, **kwargs):
            self.messages = SimpleNamespace(create=lambda **kw: SimpleNamespace(
                content=[anthropic.types.TextBlock(type="text", text="blue")], usage=billed))

    monkeypatch.setattr(anthropic, "Anthropic", Client)
    send = anthropic_send(resolve("anthropic:claude-haiku-4-5"), "k", max_tokens=64)
    with collecting("reader") as used:
        send("claude-haiku-4-5", "", "q", {})
    assert used.reported() == [TokenUsage(component="reader", provider="anthropic",
                                          model="claude-haiku-4-5", input_tokens=420,
                                          output_tokens=7)]
