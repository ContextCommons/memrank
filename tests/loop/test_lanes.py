"""``--concurrency N``: cases side by side, each case in order, the result the same as one lane."""

from __future__ import annotations

import threading
from collections import defaultdict

import pytest

from memrank.connect.base import AgentUnreachable
from memrank.connect.spec import AgentSpec
from memrank.loop import run as loop_run
from memrank.loop.lanes import Lanes, _lane
from memrank.loop.run import RunAborted, RunOptions, run_agent
from memrank.service.protocol import AgentRef, Op, RunResult, Step, StepResult
from tests.loop.conftest import OrgApi, run_id_of
from tests.loop.test_run import quiet

#: A guard against a hang if lanes did not run side by side -- never a condition under test.
HANG_GUARD_S = 30


class Memory:
    """A thread-safe agent that remembers per session and answers "blue" when it heard it.

    ``together`` makes the first reset of each case wait until that many cases have reached
    theirs, which only happens when that many lanes run at once.
    """

    def __init__(self, together: int = 1, down_on: str | None = None) -> None:
        self.lock = threading.Lock()
        self.memory: dict[str, str] = {}
        self.ops: dict[str, list[str]] = defaultdict(list)
        self.barrier = threading.Barrier(together)
        self.down_on = down_on
        self.seen_probe = 0

    def _note(self, step: Step, op: str) -> None:
        with self.lock:
            self.ops[str(step.case_id)].append(op)

    def reset(self, step: Step) -> None:
        probe = str(step.case_id).startswith("memrank-probe")
        if not probe and not self.ops.get(str(step.case_id)):
            self.barrier.wait(timeout=HANG_GUARD_S)
        self._note(step, "reset")
        with self.lock:
            self.memory[str(step.session_id)] = ""

    def feed(self, step: Step) -> None:
        if step.case_id == self.down_on:
            raise AgentUnreachable("refused")
        self._note(step, "feed")
        text = " ".join(t.text for s in step.sessions or [] for t in s.turns)
        with self.lock:
            self.memory[str(step.session_id)] = text

    def ask(self, step: Step) -> str:
        self._note(step, "ask")
        with self.lock:
            return "blue" if "blue" in self.memory.get(str(step.session_id), "") \
                else "I don't know"

    def close(self) -> None:
        pass


def spec(agent: Memory) -> AgentSpec:
    return AgentSpec(ref=AgentRef(name="memory"), connector=agent, start=None, description=None)


@pytest.fixture(autouse=True)
def org(monkeypatch) -> OrgApi:
    fake = OrgApi()
    monkeypatch.setattr(loop_run, "judge", lambda result, key, judged=None: result)
    return fake


def comparable(result: RunResult) -> dict:
    """The result without what differs between any two runs: its id and its timings."""
    data = result.model_dump(mode="json", exclude={"run_id", "latency_ms"})
    for case in data["cases"]:
        for question in case["questions"]:
            question.pop("elapsed_ms")
    return data


def test_two_lanes_run_both_cases_at_once_and_each_case_in_order(engine, tmp_path, org):
    agent = Memory(together=2)
    options = RunOptions(evaluation="locomo", judge=False, concurrency=2)
    run_agent(engine, spec(agent), options, tmp_path, org.hosted(None), quiet([]))
    assert agent.ops["c1"] == ["reset", "feed", "ask", "ask"]
    assert agent.ops["c2"] == ["reset", "feed", "ask"]


@pytest.mark.parametrize("concurrency", [2, 4])
def test_the_result_does_not_depend_on_how_many_lanes_ran(engine, tmp_path, org, concurrency):
    one = run_agent(engine, spec(Memory()), RunOptions(evaluation="locomo", judge=False),
                    tmp_path / "one", org.hosted(None), quiet([]))
    many = run_agent(engine, spec(Memory(together=2)),
                     RunOptions(evaluation="locomo", judge=False, concurrency=concurrency),
                     tmp_path / "many", org.hosted(None), quiet([]))
    assert comparable(many.result) == comparable(one.result)


def test_an_unreachable_agent_in_one_lane_stops_every_lane_and_the_run_online(engine, tmp_path,
                                                                             org):
    agent, echoed = Memory(together=2, down_on="c2"), []
    options = RunOptions(evaluation="locomo", judge=False, concurrency=2)
    with pytest.raises(RunAborted, match="memrank run --resume"):
        run_agent(engine, spec(agent), options, tmp_path, org.hosted(None), quiet(echoed))
    assert org.states == ["running", "stopped"]
    assert "--resume" in org.syncs[-1]["error"] and "ask" not in agent.ops["c2"]
    run_id = run_id_of(echoed)
    assert not engine.status(run_id).done  # c2's feed was never recorded: a resume redoes it


def test_a_step_in_flight_when_the_run_stops_is_not_recorded():
    stop, posted = threading.Event(), []

    def carry(step, failed):
        stop.set()  # Ctrl-C arrives while the agent is answering
        return StepResult(error="killed by the interrupt", elapsed_ms=1)

    lanes = Lanes(next=lambda lane, count: Step(step_id="s1", op=Op.ASK), carry=carry,
                  post=lambda step, result: posted.append(result), failed=lambda exc: None)
    _lane(lanes, 0, 1, stop)
    assert posted == []  # left outstanding: the service restarts that case on resume
