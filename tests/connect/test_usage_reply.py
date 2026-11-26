"""An agent's reported token usage: read from ask and feed replies, refused when malformed."""

from __future__ import annotations

import json

import httpx
import pytest

from memrank.connect.base import AgentError
from memrank.connect.command import CommandConnector, CommandSpec
from memrank.connect.http import HttpConnector, HttpSpec
from memrank.connect.presets import USAGE_FIELD, preset_mapping
from memrank.connect.reply import ReplyReader, UsageReader, combined
from memrank.service.protocol import Op, Question, Session, Step, TokenUsage, Turn

READER = {"component": "reader", "provider": "anthropic", "model": "claude-haiku-4-5",
          "input_tokens": 1834, "output_tokens": 12}
EMBEDDER = {"component": "embedder", "provider": "openai", "model": "text-embedding-3-small",
            "input_tokens": 9}


def test_usage_is_read_as_entries_and_an_embedding_has_no_output():
    answer = ReplyReader("a", usage="u").read({"a": "blue", "u": [READER, EMBEDDER]}, "reply")
    assert answer.usage == [TokenUsage(**READER), TokenUsage(**EMBEDDER)]
    assert answer.usage[1].output_tokens is None


def test_absent_usage_is_not_reported_and_an_empty_list_is_nothing_paid():
    assert ReplyReader("a", usage="u").read({"a": "x"}, "reply").usage is None
    assert ReplyReader("a").read({"a": "x", "u": [READER]}, "reply").usage is None
    assert ReplyReader("a", usage="u").read({"a": "x", "u": []}, "reply").usage == []


@pytest.mark.parametrize("bad", [5, {"component": "reader"}, [{**READER, "input_tokens": -1}],
                                 [{**READER, "input_tokens": 1.5}], [{**READER, "cost": 1}],
                                 [{**READER, "model": ""}]])
def test_malformed_usage_is_the_agents_error_naming_the_reply(bad):
    with pytest.raises(AgentError, match="the ask response"):
        UsageReader("u").read({"u": bad}, "the ask response")


def test_a_step_of_several_replies_reports_them_all_or_none():
    entry = TokenUsage(**READER)
    assert combined([None, [entry], [], [entry]]) == [entry, entry]
    assert combined([None, None]) is None


def _step(op: Op) -> Step:
    turns = [Turn(role="user", speaker="A", text="hi")]
    sessions = [Session(id="s1", turns=turns), Session(id="s2", turns=turns)]
    return Step(op=op, case_id="c", session_id="sid", sessions=sessions,
                question=Question(id="q", text="what?"))


def test_openai_chat_reads_usage_from_every_feed_request_and_from_the_answer():
    def reply(request: httpx.Request) -> httpx.Response:
        op = json.loads(request.content)["metadata"]["memrank_op"]
        content = "" if op == "feed" else "blue"
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}],
                                         USAGE_FIELD: [READER] if op == "ask" else [EMBEDDER]})

    spec = HttpSpec(base_url="http://agent", vars={"model": "m"},
                    **preset_mapping("openai-chat", {"model": "m"}))
    connector = HttpConnector(spec, transport=httpx.MockTransport(reply))
    assert connector.feed(_step(Op.FEED)) == [TokenUsage(**EMBEDDER)] * 2
    assert connector.ask(_step(Op.ASK)).usage == [TokenUsage(**READER)]


def test_an_openai_server_that_reports_no_usage_reports_none():
    plain = httpx.MockTransport(lambda request: httpx.Response(200, json={
        "choices": [{"message": {"content": "blue"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2}}))
    spec = HttpSpec(base_url="http://agent", vars={"model": "m"},
                    **preset_mapping("openai-chat", {"model": "m"}))
    connector = HttpConnector(spec, transport=plain)
    assert connector.feed(_step(Op.FEED)) is None
    assert connector.ask(_step(Op.ASK)).usage is None


def test_a_command_feed_reports_usage_only_when_its_spec_opts_in(tmp_path):
    agent = tmp_path / "agent.py"
    agent.write_text("import json, sys\nsys.stdin.read()\n"
                     f"print(json.dumps({{'usage': [{EMBEDDER!r}], 'answer': 'blue'}}))\n")
    argv = ["{python}", str(agent)]
    plain = CommandConnector(CommandSpec(feed=argv, ask=argv))
    assert plain.feed(_step(Op.FEED)) is None
    opted = CommandConnector(CommandSpec(feed=argv, ask=argv, feed_json={"usage": "usage"},
                                         ask_json={"answer": "answer", "usage": "usage"}))
    assert opted.feed(_step(Op.FEED)) == [TokenUsage(**EMBEDDER)]
    assert opted.ask(_step(Op.ASK)).usage == [TokenUsage(**EMBEDDER)]


def test_a_command_feed_that_promises_json_and_prints_text_fails(tmp_path):
    agent = tmp_path / "agent.py"
    agent.write_text("print('stored')\n")
    connector = CommandConnector(CommandSpec(feed=["{python}", str(agent)], ask=["true"],
                                             feed_json={"usage": "usage"}))
    with pytest.raises(AgentError, match="feed command's output is not JSON"):
        connector.feed(_step(Op.FEED))
