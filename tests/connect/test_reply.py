"""The agent's own retrieval time, read from its reply by both connectors that can carry it."""

from __future__ import annotations

import json
import math
import sys

import pytest

from memrank.connect.base import AgentError
from memrank.connect.command import CommandConnector, CommandSpec
from memrank.connect.reply import ReplyReader
from tests.connect.test_http import ASK, connector, recording

REPLY = ReplyReader("answer", "timings.retrieval_ms")


def test_a_reported_retrieval_time_is_read_beside_the_answer():
    answer = REPLY.read({"answer": "A", "timings": {"retrieval_ms": 12}}, "the reply")
    assert (answer.text, answer.retrieval_ms) == ("A", 12.0)


def test_an_answer_that_reports_none_has_none_never_zero():
    assert REPLY.read({"answer": "A"}, "the reply").retrieval_ms is None


def test_a_spec_that_names_no_path_reads_no_retrieval_time():
    reply = {"answer": "A", "timings": {"retrieval_ms": 12}}
    assert ReplyReader("answer").read(reply, "the reply").retrieval_ms is None


@pytest.mark.parametrize("value", ["12", -1, True, math.inf, [12]])
def test_a_retrieval_time_that_is_not_a_non_negative_number_fails_naming_the_reply(value):
    with pytest.raises(AgentError, match="in the ask response; the retrieval time must be"):
        REPLY.read({"answer": "A", "timings": {"retrieval_ms": value}}, "the ask response")


def test_the_http_connector_reads_it_from_the_ask_response():
    _, transport = recording({"answer": "A", "meta": {"retrieval_ms": 7.5}})
    agent = connector(transport, ask={"path": "/ask", "answer": "answer",
                                      "retrieval_ms": "meta.retrieval_ms"})
    assert agent.ask(ASK).retrieval_ms == 7.5


def test_the_http_connector_fails_on_a_bad_value():
    _, transport = recording({"answer": "A", "meta": {"retrieval_ms": "fast"}})
    agent = connector(transport, ask={"path": "/ask", "answer": "answer",
                                      "retrieval_ms": "meta.retrieval_ms"})
    with pytest.raises(AgentError, match="'fast' in the ask response"):
        agent.ask(ASK)


def _command(tmp_path, printed: str, **spec) -> CommandConnector:
    script = tmp_path / "agent.py"
    script.write_text(f"print({printed!r})")
    argv = [sys.executable, str(script)]
    return CommandConnector(CommandSpec(feed=argv, ask=argv, **spec))


def test_the_command_connector_reads_json_output_when_the_spec_opts_in(tmp_path):
    printed = json.dumps({"answer": "A", "retrieval_ms": 3})
    agent = _command(tmp_path, printed,
                     ask_json={"answer": "answer", "retrieval_ms": "retrieval_ms"})
    answer = agent.ask(ASK)
    assert (answer.text, answer.retrieval_ms) == ("A", 3.0)


def test_without_opting_in_the_command_output_is_the_plain_answer(tmp_path):
    printed = json.dumps({"answer": "A", "retrieval_ms": 3})
    answer = _command(tmp_path, printed).ask(ASK)
    assert (answer.text, answer.retrieval_ms) == (printed, None)


def test_output_that_is_not_json_when_the_spec_says_it_is_fails(tmp_path):
    agent = _command(tmp_path, "just text", ask_json={"answer": "answer"})
    with pytest.raises(AgentError, match="not JSON, as the spec's ask_json says"):
        agent.ask(ASK)


def test_the_http_connector_without_the_path_is_unchanged():
    _, transport = recording({"answer": "A", "meta": {"retrieval_ms": 7.5}})
    assert connector(transport).ask(ASK).retrieval_ms is None
