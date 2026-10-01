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
"""Every way ``memrank run`` can end, as the one :class:`~memrank.outcome.Outcome` it shows.

The exception that ended a run says what went wrong; the run's trace (:mod:`memrank.loop.trace`)
says how far it got. Together they decide the ending: done, a step for the reader (their agent,
their key, their network, Ctrl-C), or broken -- which is kept for failures inside memrank
itself, so that nothing another system did is ever reported as a bug in memrank.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from memrank.api_client import ApiUnreachable
from memrank.connect.base import AgentError, AgentUnreachable
from memrank.connect.probe import LeakDetected
from memrank.connect.process import AgentStartFailed
from memrank.connect.spec import AgentSpec
from memrank.definitions.commands import CommandFailed
from memrank.errors import ActionRequired, MemrankError
from memrank.loop.explain import NO_CREDIT_REASON as _NO_CREDIT
from memrank.loop.explain import JudgeFailed, account
from memrank.loop.report import achieved, failures
from memrank.loop.run import ProbeIncomplete, RunAborted, UploadIncomplete
from memrank.loop.trace import RunTrace
from memrank.outcome import Details, Kind, Link, Outcome, Step
from memrank.placement.run_api_client import RunApiError

#: Where a bug in memrank is reported.
ISSUES_URL = "https://github.com/ContextCommons/memrank/issues"
#: The exit status of a run stopped with Ctrl-C, as a shell reports SIGINT.
INTERRUPTED_EXIT = 130


@dataclass
class Invocation:
    """What the user ran, as an ending repeats it back.

    Attributes:
        org: The organisation the run is recorded in, once known.
        agent: The agent the run evaluates, once its file was read.
        command: The user's own command line, to run again as it was.
        resume: The command that continues a run, given its id.
    """

    command: str
    resume: Callable[[str], str]
    org: str | None = None
    agent: AgentSpec | None = None


@dataclass
class _Ending:
    """One ending being built: the call, the trace, and what the reader does next."""

    call: Invocation
    trace: RunTrace

    def again(self, text: str) -> Step:
        """The step that picks the run up where it is: resume a run that exists, else rerun."""
        if self.trace.run_id is not None:
            return Step(text, (self.call.resume(self.trace.run_id),))
        return Step(text, (self.call.command,))

    def link(self) -> Link | None:
        return Link("View the run", self.trace.url) if self.trace.url else None

    def saved(self) -> str | None:
        folder = self.trace.folder
        return f"{folder}/" if folder is not None and folder.is_dir() else None

    def title(self) -> str:
        return "The run didn't start" if self.trace.run_id is None else "The run stopped"

    def achieved(self) -> str | None:
        judged = self.trace.judged
        recorded = judged or self.trace.recorded
        return achieved(recorded, judged=judged is not None and judged.judged) \
            if recorded is not None else None

    def note(self) -> str | None:
        recorded = self.trace.recorded
        summary = failures(recorded, self.trace) if recorded is not None else None
        return (f"{summary}. A plain resume keeps them failed; add --retry-failed to ask them "
                "again." if summary else None)

    def outcome(self, kind: Kind, title: str, **parts: object) -> Outcome:
        base: dict[str, object] = {"achieved": self.achieved(), "note": self.note(),
                                   "saved": self.saved(), "link": self.link()}
        return Outcome(kind=kind, title=title, **{**base, **parts})  # type: ignore[arg-type]


def _lead_and_rest(exc: BaseException) -> tuple[str, tuple[str, ...]]:
    """A message's first line, then its indented detail lines (memrank's own message shape)."""
    first, *rest = (str(exc) or type(exc).__name__).splitlines()
    return first, tuple(row.strip() for row in rest if row.strip())


def _judge_steps(end: _Ending, exc: JudgeFailed, org: str) -> tuple[str, tuple[Step, ...]]:
    """What happened when Anthropic would not judge, and the steps before judging again."""
    save = (f"memrank secrets set ANTHROPIC_API_KEY --org {org}",)
    if exc.status is None:
        return ("memrank could not get a verdict from Anthropic: no answer came back. The "
                "details below say what went wrong on the way.",
                (Step("Check that this machine can reach Anthropic (api.anthropic.com)."),))
    if exc.reason == _NO_CREDIT:
        return (f"Anthropic refused to judge the answers: the account behind {org}'s key has "
                "no credit left.",
                (Step("Add credit to that Anthropic account, or save a different key:", save),))
    if exc.status == 429:
        return ("Anthropic kept refusing with rate limits, even after memrank waited, so "
                "judging gave up.", (Step("Wait a few minutes, or raise the account's rate "
                                          "limit."),))
    return (f"Anthropic refused to judge the answers with {org}'s key: {exc.reason}.",
            (Step("Check that key's Anthropic account (its key, permissions and limits), or "
                  "save a different key:", save),))


def _judge(end: _Ending, exc: JudgeFailed) -> Outcome:
    happened, steps = _judge_steps(end, exc, end.call.org or "your organisation")
    return end.outcome(Kind.ACTION, "Judging couldn't run", happened=happened,
                       steps=(*steps, end.again("Then judge the saved answers:")),
                       details=Details("Anthropic", exc.details))


def _agent_check(end: _Ending, exc: AgentError, agent: AgentSpec) -> Outcome:
    told = account(exc, agent)
    happened = ("Before asking any questions, memrank runs a quick check that your agent keeps "
                f"each conversation separate. It couldn't finish: {told.happened}")
    return end.outcome(Kind.ACTION, "Your agent failed the check before the run",
                       happened=happened, details=told.details,
                       steps=(told.check, end.again("Then run it again:")))


def _unreachable(end: _Ending, exc: MemrankError, stopped: bool) -> Outcome:
    lead, rest = _lead_and_rest(exc.__cause__ if stopped and exc.__cause__ else exc)
    what = ("Your agent stopped answering, so the run stopped. Everything done so far is saved."
            if stopped else "memrank could not connect to your agent.")
    return end.outcome(Kind.ACTION, "The run stopped" if stopped else "Your agent isn't running",
                       happened=what,
                       steps=(Step("Start your agent, or check the address in its file."),
                              end.again("Then continue the run:" if stopped
                                        else "Then run it again:")),
                       details=Details("the connection to your agent", (lead, *rest)))


def _leak(end: _Ending, exc: LeakDetected) -> Outcome:
    return end.outcome(Kind.ACTION, "Your agent mixed up two conversations", happened=str(exc),
                       steps=(end.again("Once it keeps them apart, run it again:"),))


def _started(end: _Ending, exc: AgentStartFailed) -> Outcome:
    return end.outcome(Kind.ACTION, "Your agent didn't start", happened=exc.statement,
                       steps=(*exc.steps, end.again("Once it starts, run it again:")),
                       details=Details("your agent's output",
                                       tuple(exc.output.splitlines()[-12:])))


def _upload(end: _Ending, exc: UploadIncomplete) -> Outcome:
    cause = exc.__cause__ or exc
    lead, rest = _lead_and_rest(cause)
    org = end.call.org or "your organisation"
    return end.outcome(Kind.ACTION, f"The run couldn't be saved to {org}'s run history",
                       happened=f"{lead[:1].upper()}{lead[1:]}",
                       steps=(end.again("Once memrank can reach its API again, send it:"),),
                       details=Details("the memrank API", rest) if rest else None)


def _api(end: _Ending, exc: MemrankError) -> Outcome:
    lead, rest = _lead_and_rest(exc)
    broken = isinstance(exc, RunApiError) and (exc.status or 0) >= 500
    steps: tuple[Step, ...] = (end.again("Try again:"),)
    if broken:
        steps += (Step("If it keeps failing, report it with the details below:", (ISSUES_URL,)),)
    return end.outcome(Kind.BROKEN if broken else Kind.ACTION,
                       "memrank's service failed" if broken else end.title(),
                       happened=f"{lead[:1].upper()}{lead[1:]}", steps=steps,
                       details=Details("the memrank API", rest) if rest else None)


def _program(end: _Ending, exc: CommandFailed) -> Outcome:
    """A program the evaluation file runs -- its case command or a grader -- failed."""
    lead, rest = _lead_and_rest(exc)
    return end.outcome(Kind.ACTION, "Your evaluation's program failed", happened=lead,
                       steps=(*exc.steps, end.again("Once it works, run it again:")),
                       details=Details("your program", rest) if rest else None)


def _interrupted(end: _Ending) -> Outcome:
    if end.trace.run_id is None:
        return end.outcome(Kind.ACTION, "Stopped: you pressed Ctrl-C",
                           happened="Nothing was started.")
    return end.outcome(Kind.ACTION, "The run stopped: you pressed Ctrl-C",
                       happened="Everything done so far is saved.",
                       steps=(end.again("To continue where it stopped:"),))


def _step_needed(end: _Ending, exc: ActionRequired) -> Outcome:
    if all(not step.text for step in exc.steps):  # the short form: one sentence, then commands
        return end.outcome(Kind.ACTION, end.title(), steps=(Step(exc.statement, exc.commands),))
    return end.outcome(Kind.ACTION, end.title(), happened=exc.statement, steps=exc.steps)


def _bug(end: _Ending, exc: BaseException) -> Outcome:
    steps = [Step("Report it, with the details below:", (ISSUES_URL,)),
             Step("To include where it happened, run it again with a traceback:",
                  (f"MEMRANK_DEBUG=1 {end.call.command}",))]
    if end.trace.run_id is not None:
        steps.append(end.again("Your run is saved; to continue it:"))
    return end.outcome(Kind.BROKEN, "Something went wrong inside memrank",
                       happened="This is a bug in memrank, not something you did.",
                       steps=tuple(steps),
                       details=Details("memrank", (f"{type(exc).__name__}: {exc}",)))


def _failure(end: _Ending, exc: MemrankError) -> Outcome:
    """Every other failure memrank names itself: its sentence already says what to do."""
    lead, rest = _lead_and_rest(exc)
    return end.outcome(Kind.ACTION, end.title(), happened=f"{lead[:1].upper()}{lead[1:]}",
                       details=Details("memrank", rest) if rest else None)


def ended(exc: BaseException, trace: RunTrace, call: Invocation) -> tuple[Outcome, int]:
    """The ending of a run that stopped on ``exc``, and the exit status it leaves with."""
    end = _Ending(call, trace)
    if isinstance(exc, KeyboardInterrupt):
        return _interrupted(end), INTERRUPTED_EXIT
    return _classify(end, exc), 1


def _classify(end: _Ending, exc: BaseException) -> Outcome:
    agent = end.call.agent
    if isinstance(exc, JudgeFailed):
        return _judge(end, exc)
    if isinstance(exc, ProbeIncomplete) and isinstance(exc.__cause__, AgentError) and agent:
        return _agent_check(end, exc.__cause__, agent)
    if isinstance(exc, RunAborted | AgentUnreachable):
        return _unreachable(end, exc, stopped=isinstance(exc, RunAborted))
    if isinstance(exc, LeakDetected):
        return _leak(end, exc)
    if isinstance(exc, AgentStartFailed):
        return _started(end, exc)
    if isinstance(exc, UploadIncomplete):
        return _upload(end, exc)
    if isinstance(exc, CommandFailed):
        return _program(end, exc)
    if isinstance(exc, ApiUnreachable | RunApiError):
        return _api(end, exc)
    if isinstance(exc, ActionRequired):
        return _step_needed(end, exc)
    if isinstance(exc, MemrankError):
        return _failure(end, exc)
    return _bug(end, exc)
