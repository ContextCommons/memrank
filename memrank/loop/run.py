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
"""The runner's loop: register the run, carry the service's steps to the agent, judge, upload.

Thin by design. The service decides what comes next and enforces the order; the runner carries
each step and reports what happened (:mod:`memrank.loop.steps`), as many at once as the run's
ceilings allow (:mod:`memrank.loop.stages`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from memrank.connect.process import started
from memrank.connect.spec import AgentSpec
from memrank.errors import MemrankError
from memrank.loop import header, runtime, stages
from memrank.loop.event_log import EventLog
from memrank.loop.grading import saved_result, start_grading
from memrank.loop.judge import DEFAULT_JUDGE
from memrank.loop.live import LiveRun
from memrank.loop.live_answers import Recorder
from memrank.loop.output import RESULT_FILE, write_outputs
from memrank.loop.steps import (  # re-exported: what a caller of the run loop names
    Evaluator,
    Instruments,
    ProbeIncomplete,
    RunAborted,
    carry_steps,
    execute,
    probe,
)
from memrank.loop.trace import Stage
from memrank.loop.upload import Hosted, upload
from memrank.loop.uploader import Uploader, start_uploading
from memrank.service.protocol import RunCreate, RunResult
from memrank.service.scoring import apply_verdicts
from memrank.term import style

__all__ = ["AGENT_LOG", "Evaluator", "Finished", "Instruments", "ProbeIncomplete",
           "RESULT_BEFORE_RETRY", "RunAborted", "RunOptions", "UploadIncomplete", "execute",
           "run_agent"]

#: The started agent's own stdout and stderr, kept beside the run it served.
AGENT_LOG = "agent.log"

#: How much of the error that stopped a run its page shows; the terminal has the rest.
STOP_REASON_CHARS = 300


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
    #: The ``provider:model`` that grades the answers; ``None`` records them unjudged.
    judge: str | None = DEFAULT_JUDGE
    #: At most how many feeds, answers and judge calls run at once, and whether a case's
    #: questions go in order (:mod:`memrank.loop.stages`). No score depends on the judge
    #: ceiling; answers depend on the others only for an agent that stores what it is asked.
    ceilings: stages.Ceilings = field(default_factory=stages.Ceilings)


@dataclass
class Finished:
    """A run that is recorded locally and in the organisation ``org``."""

    result: RunResult
    folder: Path
    url: str | None
    org: str


def run_agent(evaluator: Evaluator, agent: AgentSpec, options: RunOptions, output_root: Path,
              hosted: Hosted, tools: Instruments | None = None) -> Finished:
    """Register the run online, start the agent if it says how, probe it, drive the run while
    judging each answer as it lands, and upload it.

    The run's link is printed before the first step. Every final answer and every verdict is
    appended to the folder's ``events.jsonl`` as it happens, and sent to the organisation as it
    is appended (:mod:`memrank.loop.uploader`); an API that takes no live results is refused
    before the first step. The folder gets ``result.json`` and
    ``report.html`` with the answers as soon as the steps are done, then again once they are all
    graded; the judged pair is what is uploaded. A run that stops short -- Ctrl-C, an agent that
    went away, answers left ungraded, a failed upload -- is marked stopped online and says how
    to resume it.
    """
    tools = tools or Instruments()
    began = tools.clock()  # the session's time starts here, before its folder exists
    trace = tools.trace
    run_id = options.resume or _create(evaluator, agent, options)
    trace.run_id, trace.folder = run_id, output_root / run_id
    folder = trace.folder
    if options.retry_failed:
        _retry_failed(evaluator, run_id, folder, tools)
    live = _online(evaluator, agent, hosted, options, tools)
    trace.url = live.url
    folder.mkdir(parents=True, exist_ok=True)  # a shared service's run has no folder here yet
    recorder = Recorder(EventLog(folder))
    session = runtime.begin(folder, created=options.resume is None, clock=tools.clock,
                            started=began, settings=options.ceilings.settings())
    try:
        uploader = start_uploading(live.events, recorder.log, hosted.http, tools.echo, tools.sleep)
        trace.stage = Stage.CHECK
        status = evaluator.status(run_id)
        recorder.pool = start_grading(status, hosted, options.judge, recorder.log, live,
                              _resume_command(run_id, agent, options.judge),
                              workers=options.ceilings.judge)
        # A resumed run through a shared service names its evaluation only in its status.
        with started(agent.start, folder / AGENT_LOG, options.evaluation or status.evaluation):
            probe(agent)
            trace.stage = Stage.STEPS
            carry_steps(evaluator, agent, run_id, tools, options.ceilings, live, recorder)
        tools.echo(style.prefixed("All steps done"))
        result = _finished(evaluator, run_id, tools, live, recorder, session)
        trace.stage, trace.judged = Stage.UPLOAD, result
        url = _upload(hosted, result, folder, agent, tools, uploader)
    except BaseException as exc:
        live.stopped(_stop_reason(exc, run_id, agent, options.judge))
        raise
    finally:
        session.end()
        if recorder.pool is not None:
            recorder.pool.close(wait_for_gradings=False)  # drained unless the run is stopping
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
    for line in header.opening(hosted, status, agent.ref.name, options.judge, folder,
                               options.ceilings):
        tools.echo(line)
    live = LiveRun(hosted=hosted, run_id=run_id, agent=agent.ref.name,
                   evaluation=status.evaluation, status=lambda: evaluator.status(run_id),
                   clock=tools.clock, echo=tools.echo, judged_run=status.judge)
    tools.echo(header.live(live.open()))
    return live


def _upload(hosted: Hosted, result: RunResult, folder: Path, agent: AgentSpec,
            tools: Instruments, uploader: Uploader) -> str | None:
    """Send every live event the organisation lacks, then the sealed run."""
    try:
        uploader.drain()
        return upload(hosted, result, tools.echo)
    except MemrankError as exc:
        raise UploadIncomplete(
            f"{exc}\nThe run is saved in {folder}/. Run `memrank run --resume {result.run_id} "
            f"--agent {agent.ref.spec_path or '<agent file>'}` to send it again.") from exc


def _resume_command(run_id: str, agent: AgentSpec, spec: str | None) -> str:
    """The command that continues the run, naming its judge unless it is the default one."""
    words = f"memrank run --resume {run_id} --agent {agent.ref.spec_path or '<agent file>'}"
    return words if spec in (None, DEFAULT_JUDGE) else f"{words} --judge {spec}"


def _stop_reason(exc: BaseException, run_id: str, agent: AgentSpec, spec: str | None) -> str:
    """Why the run stopped, as its page says it, with the command that continues it."""
    if isinstance(exc, KeyboardInterrupt):
        why = "interrupted (Ctrl-C)"
    elif isinstance(exc, MemrankError):
        why = (str(exc).splitlines() or [type(exc).__name__])[0][:STOP_REASON_CHARS]
    else:
        why = f"{type(exc).__name__}: {exc}"[:STOP_REASON_CHARS]
    return f"{why} -- resume with `{_resume_command(run_id, agent, spec)}`"


def _finished(evaluator: Evaluator, run_id: str, tools: Instruments, live: LiveRun,
              recorder: Recorder, session: runtime.Session) -> RunResult:
    """The run's final result: the saved one on a resume, else recorded and judged here.

    Its verdicts are the ones the event log holds -- graded while the steps ran, or now, for any
    answer still without one (:meth:`GradingPool.finish`). Its total runtime is read once the
    last answer is graded.
    """
    folder = recorder.log.path.parent
    saved = saved_result(folder)
    if saved is not None:
        return saved
    recorded = evaluator.result(run_id)
    write_outputs(folder, recorded)
    tools.trace.recorded = recorded
    recorder.reconcile(recorded)  # answers accepted but never logged, if the runner stopped
    pool = recorder.pool
    if pool is None:
        return _sealed(folder, recorded, session)
    answers = recorded.questions - recorded.failed
    # The judge and whose key it uses were named before the first step (:mod:`header`).
    tools.echo(style.prefixed(f"Judging {answers} answers ..." if pool.grading.model
                              else f"Grading {answers} answers ..."))
    tools.trace.stage = Stage.JUDGING
    live.judging(answers)
    grading = pool.grading
    judged = apply_verdicts(recorded, pool.finish(), judge=grading.identity,
                            judge_samples=grading.samples,
                            judge_prompt_version=grading.prompt_version)
    return _sealed(folder, judged, session)


def _sealed(folder: Path, result: RunResult, session: runtime.Session) -> RunResult:
    """``result`` with the run's total runtime, written to the folder as the final outputs."""
    final = result.model_copy(update={"runtime_ms": session.total_ms()})
    write_outputs(folder, final)
    return final


def _create(evaluator: Evaluator, agent: AgentSpec, options: RunOptions) -> str:
    """A new run's id; its size is said with the rest of its opening lines (:mod:`header`)."""
    created = evaluator.create(RunCreate(evaluation=options.evaluation, agent=agent.ref,
                                         cases=options.cases, questions=options.questions,
                                         seed=options.seed,
                                         judge=options.judge is not None))
    return created.run_id
