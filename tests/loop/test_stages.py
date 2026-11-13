"""Three stage ceilings: feeds, answers and judge calls at once, each a hard ceiling.

Through the real run loop with an in-process service and a thread-safe agent: each pool really
runs that many at once and never more, a conversation's questions wait for its feed, questions
in order never overlap within a conversation, and the result is the same whatever the ceilings.
"""

from __future__ import annotations

import json
import threading
from collections import defaultdict

import click
import pytest

from memrank.connect.base import AgentUnreachable
from memrank.connect.spec import AgentSpec
from memrank.loop import runtime
from memrank.loop import stages as stages_module
from memrank.loop.client import ServiceClient
from memrank.loop.run import Instruments, RunAborted, RunOptions, run_agent
from memrank.loop.stages import Ceilings, Stages, drive
from memrank.service.protocol import AgentRef, Op, Pool, RunResult, Step, StepResult
from tests.asgi_transport import asgi_transport
from tests.loop.conftest import OrgApi, run_id_of, stub_grading
from tests.loop.test_run import quiet

#: A guard against a hang if calls did not run side by side -- never a condition under test.
HANG_GUARD_S = 30


class Memory:
    """A thread-safe agent that remembers per session and answers "blue" when it heard it.

    It records every call per case, and the most calls of each kind it ever had in flight at
    once, overall and per case. ``feeds_together`` and ``asks_together`` make the first that
    many feeds, or asks, wait for one another -- which only happens when that many run at once.
    """

    def __init__(self, feeds_together: int = 1, asks_together: int = 1,
                 down_on: str | None = None) -> None:
        self.lock = threading.Lock()
        self.memory: dict[str, str] = {}
        self.ops: dict[str, list[str]] = defaultdict(list)
        self.barriers = {"feed": threading.Barrier(feeds_together, timeout=HANG_GUARD_S),
                         "ask": threading.Barrier(asks_together, timeout=HANG_GUARD_S)}
        self.joined: dict[str, int] = defaultdict(int)
        self.down_on = down_on
        self.now: dict[str, int] = defaultdict(int)
        self.most: dict[str, int] = defaultdict(int)

    def _enter(self, step: Step, op: str) -> bool:
        """Note the call; whether it belongs to a case (not the leak probe)."""
        case = str(step.case_id)
        with self.lock:
            self.ops[case].append(op)
            for key in (op, f"{op}:{case}"):
                self.now[key] += 1
                self.most[key] = max(self.most[key], self.now[key])
        return not case.startswith("memrank-probe")

    def _leave(self, step: Step, op: str) -> None:
        with self.lock:
            for key in (op, f"{op}:{step.case_id}"):
                self.now[key] -= 1

    def reset(self, step: Step) -> None:
        self._enter(step, "reset")
        with self.lock:
            self.memory[str(step.session_id)] = ""
        self._leave(step, "reset")

    def feed(self, step: Step) -> None:
        if step.case_id == self.down_on:
            raise AgentUnreachable("refused")
        if self._enter(step, "feed"):
            self._meet("feed")
        text = " ".join(t.text for s in step.sessions or [] for t in s.turns)
        with self.lock:
            self.memory[str(step.session_id)] = text
        self._leave(step, "feed")

    def ask(self, step: Step) -> str:
        if self._enter(step, "ask"):
            self._meet("ask")
        with self.lock:
            calls = self.ops[str(step.case_id)]
            since_reset = calls[len(calls) - calls[::-1].index("reset"):] \
                if "reset" in calls else []
            fed = "feed" in since_reset
            heard = "blue" in self.memory.get(str(step.session_id), "")
        self._leave(step, "ask")
        return ("blue" if heard else "I don't know") if fed else "asked before its feed"

    def _meet(self, op: str) -> None:
        """The first calls of ``op``, as many as its barrier's parties, wait for one another;
        it breaks, failing them, unless that many are in flight at once. Later calls pass."""
        barrier = self.barriers[op]
        with self.lock:
            joins = self.joined[op] < barrier.parties
            self.joined[op] += 1
        if joins:
            barrier.wait()

    def close(self) -> None:
        pass


def spec(agent: Memory) -> AgentSpec:
    return AgentSpec(ref=AgentRef(name="memory"), connector=agent, start=None, description=None)


@pytest.fixture(autouse=True)
def org(monkeypatch) -> OrgApi:
    fake = OrgApi()
    stub_grading(monkeypatch)
    return fake


def comparable(result: RunResult) -> dict:
    """The result without what differs between any two runs: its id and its timings."""
    data = result.model_dump(mode="json", exclude={"run_id", "latency_ms", "runtime_ms"})
    for case in data["cases"]:
        for question in case["questions"]:
            question.pop("elapsed_ms")
    return data


def run(engine, folder, org, agent: Memory, ceilings: Ceilings, echoed=None):
    options = RunOptions(evaluation="locomo", judge=None, ceilings=ceilings)
    return run_agent(engine, spec(agent), options, folder, org.hosted(None),
                     quiet([] if echoed is None else echoed))


#: (every, ingest, answer, judge, what each stage resolves to)
RESOLVED = [
    ("nothing-given-is-eight-each", None, None, None, None, (8, 8, 8)),
    ("concurrency-sets-all-three", 40, None, None, None, (40, 40, 40)),
    ("a-specific-option-overrides-it", 40, None, 2, None, (40, 2, 40)),
    ("specific-options-alone", None, 1, 16, 4, (1, 16, 4)),
]


@pytest.mark.parametrize(("every", "ingest", "answer", "judge", "expected"),
                         [row[1:] for row in RESOLVED], ids=[row[0] for row in RESOLVED])
def test_concurrency_is_shorthand_for_all_three_ceilings(every, ingest, answer, judge, expected):
    ceilings = Ceilings.resolve(every, ingest=ingest, answer=answer, judge=judge)
    assert (ceilings.ingest, ceilings.answer, ceilings.judge) == expected


def test_a_ceiling_below_one_is_refused():
    with pytest.raises(ValueError, match="at least 1"):
        Ceilings(answer=0)


def test_two_feeds_run_at_once_and_each_case_stays_in_order(engine, tmp_path, org):
    agent = Memory(feeds_together=2)
    run(engine, tmp_path, org, agent, Ceilings(ingest=2, answer=1))
    assert agent.ops["c1"] == ["reset", "feed", "ask", "ask"]
    assert agent.ops["c2"] == ["reset", "feed", "ask"] and agent.most["feed"] == 2


def test_questions_run_together_across_conversations(engine, tmp_path, org):
    agent = Memory(asks_together=3)  # c1's two questions and c2's one, all at once
    done = run(engine, tmp_path, org, agent, Ceilings(ingest=2, answer=3))
    assert agent.most["ask"] == 3 and agent.most["ask:c1"] == 2
    assert [q.answer for c in done.result.cases for q in c.questions] == \
        ["blue", "blue", "I don't know"]


@pytest.mark.parametrize("ceilings", [Ceilings(ingest=1, answer=1), Ceilings(ingest=2, answer=2)])
def test_no_stage_ever_runs_more_than_its_ceiling(engine, tmp_path, org, ceilings):
    agent = Memory()
    run(engine, tmp_path, org, agent, ceilings)
    assert agent.most["feed"] <= ceilings.ingest and agent.most["ask"] <= ceilings.answer


def test_questions_in_order_never_overlap_within_a_conversation(engine, tmp_path, org):
    agent = Memory(asks_together=2)  # c1's first question and c2's, together
    run(engine, tmp_path, org, agent, Ceilings(ingest=2, answer=2, questions_in_order=True))
    assert agent.most["ask:c1"] == 1 and agent.most["ask"] == 2
    assert agent.ops["c1"] == ["reset", "feed", "ask", "ask"]


@pytest.mark.parametrize("ceilings", [Ceilings(ingest=4, answer=4, judge=4),
                                      Ceilings(ingest=4, answer=4, questions_in_order=True),
                                      Ceilings(ingest=1, answer=3)])
def test_the_result_does_not_depend_on_the_ceilings(engine, tmp_path, org, ceilings):
    one = run(engine, tmp_path / "one", org, Memory(), Ceilings(ingest=1, answer=1, judge=1))
    many = run(engine, tmp_path / "many", org, Memory(), ceilings)
    assert comparable(many.result) == comparable(one.result)


def test_the_same_run_over_http_runs_questions_together(client, tmp_path, org):
    service = ServiceClient("http://service", transport=asgi_transport(client.app))
    agent = Memory(asks_together=3)
    done = run(service, tmp_path, org, agent, Ceilings(ingest=2, answer=3))
    assert agent.most["ask"] == 3 and done.result.failed == 0


def test_the_opening_line_and_the_session_ledger_state_the_ceilings(engine, tmp_path, org):
    echoed: list[str] = []
    tools = Instruments(clock=lambda: 0.0, sleep=lambda s: None,
                        echo=lambda line: echoed.append(click.unstyle(line)))
    ceilings = Ceilings(ingest=3, answer=40, judge=5, questions_in_order=True)
    options = RunOptions(evaluation="locomo", judge=None, ceilings=ceilings)
    done = run_agent(engine, spec(Memory()), options, tmp_path, org.hosted(None), tools)
    assert "memrank: At most 3 feeds, 40 answers at once \u00b7 questions in order" in echoed
    ledger = (done.folder / runtime.SESSIONS_FILE).read_text(encoding="utf-8").splitlines()
    assert json.loads(ledger[0]) == {"event": "start", "created": True, **ceilings.settings()}


def test_an_unreachable_agent_stops_the_run_online_and_leaves_it_resumable(engine, tmp_path,
                                                                          org):
    echoed: list[str] = []
    with pytest.raises(RunAborted, match="memrank run --resume"):
        run(engine, tmp_path, org, Memory(down_on="c2"), Ceilings(ingest=2, answer=2), echoed)
    assert org.states == ["running", "stopped"] and "--resume" in org.syncs[-1]["error"]
    run_id = run_id_of(echoed)
    assert not engine.status(run_id).done  # c2's feed was never recorded: a resume redoes it
    # A new agent that remembers nothing: a resume feeds every unfinished conversation again.
    resumed = run_agent(engine, spec(Memory()), RunOptions(resume=run_id, judge=None), tmp_path,
                        org.hosted(None), quiet([]))
    assert [q.answer for c in resumed.result.cases for q in c.questions] == \
        ["blue", "blue", "I don't know"]


def test_a_stopping_run_posts_none_of_the_steps_in_flight(monkeypatch):
    posted: list[StepResult] = []
    both_out = threading.Barrier(2, timeout=HANG_GUARD_S)
    halted = threading.Event()  # the workers have been told to stop
    real_halt = stages_module._Shared.halt

    def halt_then_tell(shared):
        real_halt(shared)
        halted.set()

    monkeypatch.setattr(stages_module._Shared, "halt", halt_then_tell)
    handed = iter([Step(step_id="s1", op=Op.ASK), Step(step_id="s2", op=Op.ASK)])
    lock = threading.Lock()

    def carry(step: Step) -> StepResult:
        both_out.wait()  # both asks are with the agent when Ctrl-C arrives at the first
        if step.step_id == "s1":
            raise KeyboardInterrupt
        halted.wait(timeout=HANG_GUARD_S)  # s2 comes back only once the stop is known
        return StepResult(answer="blue", elapsed_ms=1)

    def next_step(pool: Pool) -> Step:
        with lock:
            return next(handed, Step(op=Op.DONE)) if pool is Pool.ANSWER else Step(op=Op.DONE)

    stages = Stages(next=next_step, carry=carry, post=lambda step, result: posted.append(result))
    with pytest.raises(KeyboardInterrupt):
        drive(stages, Ceilings(ingest=1, answer=2))
    assert posted == []  # s2 came back after the stop, and is left for a resume


class Feeds:
    """A fake service: ``cases`` cases, each a feed then ``questions`` questions."""

    def __init__(self, cases: int, questions: int) -> None:
        self.lock = threading.Lock()
        self.feeds = [f"c{i}" for i in range(cases)]
        self.fed: list[str] = []
        self.asks: list[str] = []
        self.questions = questions

    def next(self, pool: Pool) -> Step:
        with self.lock:
            if pool is Pool.INGEST and self.feeds:
                return Step(step_id=self.feeds.pop(0), op=Op.FEED)
            if pool is Pool.ANSWER and self.asks:
                return Step(step_id=self.asks.pop(0), op=Op.ASK)
            return Step(op=Op.DONE)

    def post(self, step: Step, result: StepResult) -> None:
        with self.lock:
            if step.op is Op.FEED:
                self.fed.append(str(step.step_id))
                self.asks += [f"{step.step_id}q{n}" for n in range(self.questions)]


def test_an_answer_worker_offered_nothing_wakes_when_a_feed_is_acknowledged():
    service, answered = Feeds(cases=3, questions=4), []
    feed_gate = threading.Semaphore(0)  # each feed is carried only once the last is answered

    def carry(step: Step) -> StepResult:
        if step.op is Op.FEED:
            feed_gate.acquire(timeout=HANG_GUARD_S)
            return StepResult(ok=True, elapsed_ms=1)
        answered.append(step.step_id)
        if len(answered) % 4 == 0:
            feed_gate.release()
        return StepResult(answer="blue", elapsed_ms=1)

    feed_gate.release()  # the first feed goes at once
    drive(Stages(next=service.next, carry=carry, post=service.post), Ceilings(ingest=1, answer=3))
    assert sorted(answered) == sorted(f"c{i}q{n}" for i in range(3) for n in range(4))
