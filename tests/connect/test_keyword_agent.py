"""The zero-cost witness agent passes the leak probe through the real command connector."""

from __future__ import annotations

import sys
from pathlib import Path

from memrank.connect.command import CommandConnector, CommandSpec
from memrank.connect.probe import leak_probe

SCRIPT = Path(__file__).resolve().parents[1] / "fixtures" / "agents" / "keyword_agent.py"


def test_the_keyword_agent_keeps_sessions_apart(tmp_path):
    base = [sys.executable, str(SCRIPT), str(tmp_path)]
    agent = CommandConnector(CommandSpec(
        reset=[*base, "reset", "{session_id}"], feed=[*base, "feed", "{session_id}"],
        ask=[*base, "ask", "{session_id}", "{question}"]))
    leak_probe(agent, token="c0ffee")
