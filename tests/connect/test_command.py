"""The command connector drives a real program: argv templates, stdin feed, stdout answer."""

from __future__ import annotations

import sys

import pytest

from memrank.connect.base import AgentError, ConnectorConfigError
from memrank.connect.command import CommandConnector, CommandSpec
from memrank.service.protocol import Op, Question, Session, Step, Turn

AGENT = """
import pathlib, sys
store = pathlib.Path(sys.argv[1]) / sys.argv[3]
if sys.argv[2] == "feed":
    store.write_text(sys.stdin.read())
elif sys.argv[2] == "ask":
    print("remembered: " + store.read_text().splitlines()[-1] + " / " + sys.argv[4])
else:
    sys.exit("unknown op")
"""


@pytest.fixture
def agent(tmp_path) -> CommandConnector:
    script = tmp_path / "agent.py"
    script.write_text(AGENT)
    base = [sys.executable, str(script), str(tmp_path)]
    return CommandConnector(CommandSpec(feed=[*base, "feed", "{session_id}"],
                                        ask=[*base, "ask", "{session_id}", "{question}"],
                                        reset=[*base, "bad", "{session_id}"]))


def test_feed_writes_the_transcript_and_ask_reads_the_answer(agent):
    session = Session(id="s", turns=[Turn(role="user", speaker="Ann", text="blue car")])
    agent.feed(Step(op=Op.FEED, case_id="c", session_id="k", sessions=[session]))
    answer = agent.ask(Step(op=Op.ASK, case_id="c", session_id="k",
                            question=Question(id="q", text="colour?")))
    assert answer == "remembered: Ann: blue car / colour?"


def test_a_failing_program_is_an_agent_error_with_its_stderr(agent):
    with pytest.raises(AgentError, match="unknown op"):
        agent.reset(Step(op=Op.RESET, case_id="c", session_id="k"))


def test_a_missing_program_is_refused_before_the_run():
    with pytest.raises(ConnectorConfigError, match="not on PATH"):
        CommandConnector(CommandSpec(feed=["no-such-agent-binary"], ask=["no-such-agent-binary"]))
