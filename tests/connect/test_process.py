"""Starting an agent: ready, exited, or never ready -- decided with a fake process and clock."""

from __future__ import annotations

import itertools
import sys

import pytest

from memrank import agents
from memrank.connect.base import ConnectorConfigError
from memrank.connect.process import AgentStartFailed, StartSpec, _argv, started, wait_ready
from memrank.connect.spec import load_agent

SPEC = StartSpec(argv=["agent"], ready="http://127.0.0.1:1/health", ready_timeout_s=3)


class Process:
    def __init__(self, exits_after: int | None) -> None:
        self.polls, self.exits_after, self.returncode = 0, exits_after, None

    def poll(self):
        self.polls += 1
        if self.exits_after is not None and self.polls > self.exits_after:
            self.returncode = 7
        return self.returncode


def attempt(ready_on: int | None, exits_after: int | None, tmp_path):
    answers = itertools.count(1)
    ticks = itertools.count()
    log = tmp_path / "agent.log"
    log.write_text("listening failed: port in use\n")
    wait_ready(Process(exits_after), SPEC, log,  # type: ignore[arg-type]
               ready=lambda url: ready_on is not None and next(answers) >= ready_on,
               clock=lambda: float(next(ticks)), sleep=lambda s: None)


@pytest.mark.parametrize(("ready_on", "exits_after"), [(1, None), (3, None)],
                         ids=["ready-at-once", "ready-after-polls"])
def test_an_agent_that_answers_is_ready(ready_on, exits_after, tmp_path):
    attempt(ready_on, exits_after, tmp_path)


@pytest.mark.parametrize(("ready_on", "exits_after", "match"), [
    (None, 1, "(?s)exited with 7.*port in use"),
    (None, None, "(?s)did not answer within 3.0s.*port in use"),
], ids=["exited", "never-ready"])
def test_an_agent_that_never_answers_fails_the_run_with_its_log(ready_on, exits_after, match,
                                                                tmp_path):
    with pytest.raises(AgentStartFailed, match=match):
        attempt(ready_on, exits_after, tmp_path)


def test_start_argv_names_the_run_s_evaluation() -> None:
    spec = StartSpec(argv=["{python}", "serve", "--evaluation", "{evaluation}"], ready="u")
    assert _argv(spec, "beam:100k") == [sys.executable, "serve", "--evaluation", "beam:100k"]


def test_an_unknown_start_placeholder_is_refused_before_anything_starts(tmp_path) -> None:
    spec = StartSpec(argv=["{python}", "--eval", "{evaluaton}"], ready="u")
    with pytest.raises(ConnectorConfigError, match="evaluaton"):
        with started(spec, tmp_path / "agent.log", "locomo"):
            pass
    assert not (tmp_path / "agent.log").read_text()


def test_the_full_context_agent_is_started_for_the_run_s_evaluation() -> None:
    spec = load_agent(agents.resolve("full-context"))
    assert spec.start is not None
    argv = _argv(spec.start, "longmemeval")
    assert argv[argv.index("--evaluation") + 1] == "longmemeval"
