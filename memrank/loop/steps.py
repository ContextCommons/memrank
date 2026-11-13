# Copyright 2026 AtomicStrata
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.
"""Carrying the service's steps to the agent: the call, its timing, and its report.

The service decides what comes next and enforces the order; the runner only measures -- latency
at its own boundary, around the connector call -- and reports faithfully. An agent that fails a
step has that failure recorded and the run moves on. An agent that cannot be reached at all
stops the run, because recording every remaining case as a failure would score the network
rather than the agent; the run stays resumable. How many steps are carried at once is
:mod:`memrank.loop.stages`'.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from memrank.connect.base import AgentError, AgentUnreachable, Answer, Connector, answer_of
from memrank.connect.probe import leak_probe
from memrank.connect.spec import AgentSpec
from memrank.errors import MemrankError
from memrank.loop import stages
from memrank.loop.client import retry_unreached
from memrank.loop.explain import cause, explained
from memrank.loop.live import LiveRun
from memrank.loop.live_answers import Recorder
from memrank.loop.trace import RunTrace
from memrank.service.protocol import (
    Ack,
    Begun,
    Op,
    Pool,
    Reopened,
    RunCreate,
    RunCreated,
    RunResult,
    RunStatus,
    Step,
    StepResult,
)
from memrank.term import style


class RunAborted(MemrankError):
    """The run stopped before it finished. It is saved and can be resumed."""



@dataclass
class Instruments:
    """The runner's clock, pause and output, injectable so tests run with no real time.

    Everything the runner echoes is narration about the run, so it goes to stderr; the answer --
    the run's ending -- is the one thing on stdout (:mod:`memrank.term.style`).
    """

    clock: Callable[[], float] = time.perf_counter
    sleep: Callable[[float], None] = time.sleep
    echo: Callable[[str], None] = style.say
    #: Where the run has got to, for its ending to read (:mod:`memrank.loop.trace`).
    trace: RunTrace = field(default_factory=RunTrace)


def execute(connector: Connector, step: Step, tools: Instruments,
            failed: Callable[[AgentError], None] = lambda exc: None) -> StepResult:
    """Carry one step to the agent. A reached-but-failed call becomes an error result.

    ``failed`` sees the agent's failure itself, with its status and body, before it is reduced
    to the error text the result records.
    """
    call = getattr(connector, step.op.value)
    timing: dict[str, float] = {}

    def attempt() -> str | Answer | None:
        timing["start"] = tools.clock()  # only the attempt that reached the agent is timed
        outcome = call(step)
        timing["end"] = tools.clock()
        return outcome

    try:
        outcome = retry_unreached(attempt, unreached=AgentUnreachable, sleep=tools.sleep)
    except AgentError as exc:
        failed(exc)
        return StepResult(error=str(exc), elapsed_ms=_ms(timing["start"], tools.clock()))
    elapsed = _ms(timing["start"], timing["end"])
    if step.op is Op.ASK:
        answer = answer_of(outcome or "")
        return StepResult(answer=answer.text, retrieval_ms=answer.retrieval_ms, elapsed_ms=elapsed)
    return StepResult(ok=True, elapsed_ms=elapsed)


def _ms(start: float, end: float) -> float:
    return max(0.0, (end - start) * 1000.0)


def _progress(step: Step, result: StepResult) -> str:
    """One step's line: where and what dimmed, so its outcome -- ok or FAILED -- stands out."""
    where = step.progress
    head = f"case {where.case}/{where.cases} {step.case_id}" if where else str(step.case_id)
    what = step.op.value if step.op is not Op.ASK or where is None \
        else f"ask {where.question}/{where.questions}"
    status = (style.bad("FAILED " + (result.error or "")[:80]) if result.error
              else style.good("ok"))
    took = style.dim(f"  {result.elapsed_ms / 1000:.2f}s")
    return f"{style.dim(f'{head}  {what:<10} ')}{status}{took}"


class Evaluator(Protocol):
    """The protocol calls. The in-process engine and the HTTP client both are one."""

    def create(self, request: RunCreate) -> RunCreated: ...

    def begin(self, run_id: str) -> Begun: ...

    def next(self, run_id: str, pool: Pool = Pool.INGEST, in_order: bool = False) -> Step: ...

    def post(self, run_id: str, step_id: str, result: StepResult) -> Ack: ...

    def retry_failed(self, run_id: str) -> Reopened: ...

    def status(self, run_id: str) -> RunStatus: ...

    def result(self, run_id: str) -> RunResult: ...


class ProbeIncomplete(MemrankError):
    """The leak probe could not complete, so no step was run."""


def probe(agent: AgentSpec) -> None:
    try:
        leak_probe(agent.connector)
    except AgentError as exc:
        raise ProbeIncomplete("The check that the agent keeps each conversation separate "
                              "could not complete, so no question was asked.\n"
                              f"{explained(exc, agent)}") from exc


def _once(agent: AgentSpec, tools: Instruments) -> Callable[[AgentError], None]:
    """Explain the run's first failed step in full; the rest are one line each and all are in
    the result, so a run that fails every step does not print the same paragraph every time.
    Every failure is counted by its cause, for the run's ending to sum up."""
    told: list[bool] = []

    def failed(exc: AgentError) -> None:
        tools.trace.failed(cause(exc))
        if not told:
            told.append(True)
            tools.echo(explained(exc, agent))

    return failed


def carry_steps(evaluator: Evaluator, agent: AgentSpec, run_id: str, tools: Instruments,
                ceilings: stages.Ceilings, live: LiveRun, recorder: Recorder) -> None:
    """Begin as the run's runner, then carry every step the service hands out within
    ``ceilings`` until done.

    Beginning tells the service that whatever an earlier runner left outstanding is lost (rule 4
    of :mod:`memrank.service.machine`). Each answer the service accepts is logged and handed to
    the grading pool as it lands; the pool takes it without blocking, so judging never holds an
    answer slot up (D1).
    """
    evaluator.begin(run_id)
    failed = _once(agent, tools)

    def carry(step: Step) -> StepResult:
        try:
            return execute(agent.connector, step, tools, failed)
        except AgentUnreachable as exc:
            raise RunAborted(f"{exc}. The run is saved; once the agent is up, resume it with "
                             f"`memrank run --resume {run_id}` and the same --agent.") from exc

    def post(step: Step, result: StepResult) -> None:
        assert step.step_id is not None  # only DONE carries no step id
        recorder.answered(evaluator.post(run_id, step.step_id, result))
        tools.echo(_progress(step, result))
        live.step_done()

    def next_step(pool: Pool) -> Step:
        return evaluator.next(run_id, pool, ceilings.questions_in_order)

    stages.drive(stages.Stages(next=next_step, carry=carry, post=post), ceilings)
