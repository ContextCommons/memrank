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
"""The runner's loop: ask the service for a step, carry it to the agent, report what happened.

Thin by design. The service decides what comes next and enforces the order; the runner only
measures -- latency at its own boundary, around the connector call -- and reports faithfully.
An agent that fails a step has that failure recorded and the run moves on. An agent that
cannot be reached at all stops the run, because recording every remaining case as a failure
would score the network rather than the agent; the run stays resumable.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from memrank.connect.base import AgentError, AgentUnreachable, Connector
from memrank.connect.probe import leak_probe
from memrank.connect.process import started
from memrank.connect.spec import AgentSpec
from memrank.errors import MemrankError
from memrank.judging.judge import JudgeConfig
from memrank.loop import header, lanes
from memrank.loop.client import retry_unreached
from memrank.loop.explain import cause, explained, judge_failure
from memrank.loop.judge import DEFAULT_JUDGE_MODEL, judge, needs_key
from memrank.loop.judge_key import missing_key
from memrank.loop.live import LiveRun
from memrank.loop.output import RESULT_FILE, write_outputs
from memrank.loop.trace import RunTrace, Stage
from memrank.loop.upload import Hosted, upload
from memrank.service.protocol import (
    Ack,
    Op,
    Reopened,
    RunCreate,
    RunCreated,
    RunResult,
    RunStatus,
    Step,
    StepResult,
)
from memrank.term import style

#: The started agent's own stdout and stderr, kept beside the run it served.
AGENT_LOG = "agent.log"

#: How much of the error that stopped a run its page shows; the terminal has the rest.
STOP_REASON_CHARS = 300


class RunAborted(MemrankError):
    """The run stopped before it finished. It is saved and can be resumed."""


class UploadIncomplete(MemrankError):
    """The run is finished and judged locally but not yet recorded in the organisation."""


@dataclass
class RunOptions:
    evaluation: str = ""
    cases: int | None = None
    questions: int | None = None
    seed: int = 0
    resume: str | None = None
    #: With ``resume``: run every case whose reset or feed failed again, from a fresh reset.
    retry_failed: bool = False
    judge: bool = True
    #: The Anthropic model that grades the answers when ``judge`` is on.
    judge_model: str = DEFAULT_JUDGE_MODEL
    #: How many cases run side by side (:mod:`memrank.loop.lanes`); one is sequential.
    concurrency: int = lanes.DEFAULT_CONCURRENCY


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

    def attempt() -> str | None:
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
        return StepResult(answer=outcome or "", elapsed_ms=elapsed)
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

    def next(self, run_id: str, lane: int = 0, lanes: int = 1) -> Step: ...

    def post(self, run_id: str, step_id: str, result: StepResult) -> Ack: ...

    def retry_failed(self, run_id: str) -> Reopened: ...

    def status(self, run_id: str) -> RunStatus: ...

    def result(self, run_id: str) -> RunResult: ...


class ProbeIncomplete(MemrankError):
    """The leak probe could not complete, so no step was run."""


def _probe(agent: AgentSpec) -> None:
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


def _drive(evaluator: Evaluator, agent: AgentSpec, run_id: str, tools: Instruments,
           concurrency: int, live: LiveRun) -> None:
    """Carry every step the service hands out, ``concurrency`` cases at a time, until done."""

    def carry(step: Step, failed: Callable[[AgentError], None]) -> StepResult:
        try:
            return execute(agent.connector, step, tools, failed)
        except AgentUnreachable as exc:
            raise RunAborted(f"{exc}. The run is saved; once the agent is up, resume it with "
                             f"`memrank run --resume {run_id}` and the same --agent.") from exc

    def post(step: Step, result: StepResult) -> None:
        assert step.step_id is not None  # only DONE carries no step id
        evaluator.post(run_id, step.step_id, result)
        tools.echo(_progress(step, result))
        live.step_done()

    lanes.drive(lanes.Lanes(next=lambda lane, count: evaluator.next(run_id, lane, count),
                            carry=carry, post=post, failed=_once(agent, tools)), concurrency)


@dataclass
class Finished:
    """A run that is recorded locally and in the organisation ``org``."""

    result: RunResult
    folder: Path
    url: str | None
    org: str


def run_agent(evaluator: Evaluator, agent: AgentSpec, options: RunOptions, output_root: Path,
              hosted: Hosted, tools: Instruments | None = None) -> Finished:
    """Register the run online, start the agent if it says how, probe it, drive the run, judge
    it and upload it.

    The run's link is printed before the first step. The run's folder gets ``result.json`` and
    ``report.html`` with the answers as soon as the steps are done, then again once they are
    judged; the judged pair is what is uploaded. A run that stops short -- Ctrl-C, an agent that
    went away, a failed upload -- is marked stopped online and says how to resume it.
    """
    tools = tools or Instruments()
    trace = tools.trace
    run_id = options.resume or _create(evaluator, agent, options)
    trace.run_id, trace.folder = run_id, output_root / run_id
    folder = trace.folder
    if options.retry_failed:
        _retry_failed(evaluator, run_id, folder, tools)
    live = _online(evaluator, agent, hosted, options, tools)
    trace.url = live.url
    try:
        trace.stage = Stage.CHECK
        # A resumed run through a shared service names its evaluation only in its status.
        evaluation = options.evaluation or evaluator.status(run_id).evaluation
        with started(agent.start, folder / AGENT_LOG, evaluation):
            _probe(agent)
            trace.stage = Stage.STEPS
            _drive(evaluator, agent, run_id, tools, options.concurrency, live)
        tools.echo(style.prefixed("All steps done"))
        result = _finished(evaluator, run_id, folder, hosted, tools, live, options.judge_model)
        trace.stage, trace.judged = Stage.UPLOAD, result
        url = _upload(hosted, result, folder, agent, tools)
    except BaseException as exc:
        live.stopped(_stop_reason(exc, run_id, agent))
        raise
    return Finished(result=result, folder=folder, url=url or live.url, org=hosted.org)


#: Where a retry moves the result it makes out of date (:func:`_retry_failed`).
RESULT_BEFORE_RETRY = "result.before-retry.json"


def _retry_failed(evaluator: Evaluator, run_id: str, folder: Path, tools: Instruments) -> None:
    """Reopen the run's failed cases and failed answers, and move aside the result they made
    out of date.

    That result -- judged, perhaps uploaded -- is what a resume would otherwise send again
    without running anything (:func:`_saved`). Kept rather than deleted, as the record of the
    run before its retry. Judging the result again pays only for the new answers: every answer
    judged before is a hit in the judge's cache.
    """
    reopened = evaluator.retry_failed(run_id).cases
    if not reopened:
        tools.echo(style.prefixed("No failed case or answer to retry"))
        return
    tools.echo(style.prefixed(f"Retrying {len(reopened)} case"
                              f"{'' if len(reopened) == 1 else 's'} with failures from a fresh "
                              f"reset, asking only what failed: {', '.join(reopened)}"))
    saved = folder / RESULT_FILE
    if saved.exists():
        saved.replace(folder / RESULT_BEFORE_RETRY)


def _online(evaluator: Evaluator, agent: AgentSpec, hosted: Hosted, options: RunOptions,
            tools: Instruments) -> LiveRun:
    """The run, said to be whose and which, registered as running in the organisation, its
    link printed."""
    run_id, folder = tools.trace.run_id, tools.trace.folder
    assert run_id is not None and folder is not None  # set by run_agent before it comes here
    status = evaluator.status(run_id)
    for line in header.opening(hosted, status, agent.ref.name, options.judge_model, folder):
        tools.echo(line)
    live = LiveRun(hosted=hosted, run_id=run_id, agent=agent.ref.name,
                   evaluation=status.evaluation, status=lambda: evaluator.status(run_id),
                   clock=tools.clock, echo=tools.echo, judged_run=status.judge)
    tools.echo(header.live(live.open()))
    return live


def _upload(hosted: Hosted, result: RunResult, folder: Path, agent: AgentSpec,
            tools: Instruments) -> str | None:
    try:
        return upload(hosted, result, tools.echo)
    except MemrankError as exc:
        raise UploadIncomplete(
            f"{exc}\nThe run is saved in {folder}/. Run `memrank run --resume {result.run_id} "
            f"--agent {agent.ref.spec_path or '<agent file>'}` to send it again.") from exc


def _stop_reason(exc: BaseException, run_id: str, agent: AgentSpec) -> str:
    """Why the run stopped, as its page says it, with the command that continues it."""
    if isinstance(exc, KeyboardInterrupt):
        why = "interrupted (Ctrl-C)"
    elif isinstance(exc, MemrankError):
        why = (str(exc).splitlines() or [type(exc).__name__])[0][:STOP_REASON_CHARS]
    else:
        why = f"{type(exc).__name__}: {exc}"[:STOP_REASON_CHARS]
    return (f"{why} -- resume with `memrank run --resume {run_id} --agent "
            f"{agent.ref.spec_path or '<agent file>'}`")


def _saved(folder: Path) -> RunResult | None:
    """The folder's result when nothing is left to do to it: judged, or never to be judged."""
    path = folder / RESULT_FILE
    if not path.exists():
        return None
    saved = RunResult.model_validate_json(path.read_text(encoding="utf-8"))
    return saved if saved.judged or not saved.identity.judge_requested else None


def _finished(evaluator: Evaluator, run_id: str, folder: Path, hosted: Hosted,
              tools: Instruments, live: LiveRun, judge_model: str) -> RunResult:
    """The run's final result: the saved one on a resume, else recorded and judged here."""
    saved = _saved(folder)
    if saved is not None:
        return saved
    recorded = evaluator.result(run_id)
    write_outputs(folder, recorded)
    tools.trace.recorded = recorded
    if not recorded.identity.judge_requested:
        return recorded
    keyed = needs_key(recorded)
    if keyed and hosted.judge_key is None:
        raise missing_key(hosted.org)
    answers = recorded.questions - recorded.failed
    # The judge and whose key it uses were named before the first step (:mod:`header`).
    tools.echo(style.prefixed(f"Judging {answers} answers ..." if keyed
                              else f"Grading {answers} answers ..."))
    tools.trace.stage = Stage.JUDGING
    live.judging(answers)
    try:
        judged = judge(recorded, hosted.judge_key, JudgeConfig(judge_model=judge_model),
                       judged=live.one_judged)
    except Exception as exc:  # noqa: BLE001 - re-raised: named when it is Anthropic's side
        failure = judge_failure(exc)
        if failure is None:
            raise
        raise failure from exc
    write_outputs(folder, judged)
    return judged


def _create(evaluator: Evaluator, agent: AgentSpec, options: RunOptions) -> str:
    """A new run's id; its size is said with the rest of its opening lines (:mod:`header`)."""
    created = evaluator.create(RunCreate(evaluation=options.evaluation, agent=agent.ref,
                                         cases=options.cases, questions=options.questions,
                                         seed=options.seed, judge=options.judge))
    return created.run_id
