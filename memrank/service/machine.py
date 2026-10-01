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
"""The run's rules, as a state machine the service enforces and the runner cannot bypass.

Each case goes reset -> feed -> ask, ask, ... in that order. Cases may run side by side: a runner
works through **lanes**, each lane holding one case at a time, so ``--concurrency N`` is N lanes
and one lane is the strictly sequential run. Five rules:

1. Every case starts with a reset.
2. No question of a case is revealed until that case's feed is acknowledged.
3. A step is answered once. Re-posting identical content is a harmless retry; different
   content is refused.
4. A crashed runner resumes from here. Each case has at most one step outstanding, and a lane
   asking for its next step while its case has one outstanding means the runner lost it. A lost
   reset is simply re-issued. A lost feed or ask may or may not have reached the agent, so the
   case restarts from a reset under a fresh session id -- the agent is never fed twice into the
   same memory.
5. A lane takes the lowest-numbered case nobody holds. A case held by a lane the runner no longer
   has (it resumed with fewer lanes) is nobody's, and is taken the same way, its outstanding step
   treated as lost.

A case whose reset or feed failed is finished, and a resume leaves it so; so is a case that
completed with some answers failed. Only an explicit :func:`reopen_failed` (``memrank run
--resume <id> --retry-failed``) runs either again, as a restart: a reset under a fresh session id,
a new feed, and then only the questions still owed -- every question of a failed case from where
it stopped, or just the failed answers of a completed one. Every other case is untouched.

Pure functions over :class:`RunState`: no I/O, no clock, so every transition is testable as a
row in a table.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from memrank.errors import MemrankError
from memrank.service.protocol import Op, StepResult


class ProtocolViolation(MemrankError):
    """A runner did something the rules forbid. The state is left unchanged."""


class UnknownStep(ProtocolViolation):
    """A step id this run never issued."""


class Outstanding(BaseModel):
    step_id: str
    op: Op
    #: The index of the case the step belongs to.
    case: int


class Recorded(BaseModel):
    """An accepted step result, kept so a retry can be recognised as one."""

    op: Op
    case_id: str
    question_index: int | None = None
    result: StepResult


class CaseState(BaseModel):
    """Where one case stands: its next step, and the lane working it."""

    stage: Op = Op.RESET
    question_index: int = 0
    finished: bool = False
    lane: int | None = None
    outstanding: Outstanding | None = None
    #: The question indices a retry still has to ask, in order; empty outside a retry.
    retry: list[int] = Field(default_factory=list)


class RunState(BaseModel):
    """Where a run stands. The plan's shape is fixed at creation: case ids and question counts."""

    case_ids: list[str]
    question_counts: list[int]
    cases: list[CaseState]
    issued: int = 0
    restarts: dict[str, int] = Field(default_factory=dict)
    failed_cases: dict[str, str] = Field(default_factory=dict)
    recorded: dict[str, Recorded] = Field(default_factory=dict)
    #: The digest of the cases the run started on; a resume on other cases is refused.
    cases_digest: str | None = None

    @property
    def done(self) -> bool:
        return all(case.finished for case in self.cases)


def new_state(question_counts: dict[str, int]) -> RunState:
    return RunState(case_ids=list(question_counts), question_counts=list(question_counts.values()),
                    cases=[CaseState() for _ in question_counts])


def next_step(state: RunState, lane: int = 0,
              lanes: int = 1) -> tuple[RunState, Outstanding | None]:
    """The step ``lane`` must carry out now, or None when no case is left for it.

    None is not the end of the run while other lanes still hold cases; the run is finished when
    :attr:`RunState.done`. Returns a new state; the input is not modified.
    """
    if not 0 <= lane < lanes:
        raise ProtocolViolation(f"lane {lane} is not one of the runner's {lanes} lanes")
    state = state.model_copy(deep=True)
    index = _held(state, lane)
    if index is None:
        index = _claimable(state, lanes)
        if index is None:
            return state, None
        state.cases[index].lane = lane
    case = state.cases[index]
    if case.outstanding is not None:
        if case.outstanding.op is Op.RESET:
            return state, case.outstanding
        _restart_case(state, index)
    state.issued += 1
    case.outstanding = Outstanding(step_id=f"s{state.issued}", op=case.stage, case=index)
    return state, case.outstanding


def _held(state: RunState, lane: int) -> int | None:
    return next((i for i, case in enumerate(state.cases)
                 if case.lane == lane and not case.finished), None)


def _claimable(state: RunState, lanes: int) -> int | None:
    """The lowest unfinished case held by no lane the runner has now (rule 5)."""
    return next((i for i, case in enumerate(state.cases) if not case.finished
                 and (case.lane is None or case.lane >= lanes)), None)


def _restart_case(state: RunState, index: int) -> None:
    """Abandon the lost feed or ask: the case begins again from a reset, keeping its answers."""
    case_id = state.case_ids[index]
    state.restarts[case_id] = state.restarts.get(case_id, 0) + 1
    state.cases[index].stage = Op.RESET
    state.cases[index].outstanding = None


def reopen_failed(state: RunState) -> tuple[RunState, list[str]]:
    """Every failed case and every case with a failed answer, reopened as a restart.

    Returns the new state and the reopened case ids, in plan order. Each gets the restart a lost
    step gets (:func:`_restart_case`): it begins from a reset, and its session id moves on so the
    agent is never fed twice into the same memory. A failed case asks its remaining questions and
    its failure is cleared. A complete case with failed answers asks only those questions again;
    its successful answers are kept, and each new answer replaces the failed one. Other cases are
    not touched, and a run with nothing failed is returned unchanged.
    """
    state = state.model_copy(deep=True)
    failed_answers = _failed_answers(state)
    reopened = [case_id for case_id in state.case_ids
                if case_id in state.failed_cases or case_id in failed_answers]
    for case_id in reopened:
        index = state.case_ids.index(case_id)
        case = state.cases[index]
        if case_id in state.failed_cases:
            del state.failed_cases[case_id]
        else:
            case.retry = failed_answers[case_id]
            case.question_index = case.retry[0]
        case.finished = False
        _restart_case(state, index)
    return state, reopened


def _failed_answers(state: RunState) -> dict[str, list[int]]:
    """The question indices whose latest answer is an error, per case that has any."""
    latest: dict[tuple[str, int], bool] = {}
    for recorded in state.recorded.values():  # in step order, so a later answer wins
        if recorded.op is Op.ASK and recorded.question_index is not None:
            latest[(recorded.case_id, recorded.question_index)] = \
                recorded.result.error is not None
    failed: dict[str, list[int]] = {}
    for (case_id, question_index), is_error in sorted(latest.items()):
        if is_error and case_id not in state.failed_cases:
            failed.setdefault(case_id, []).append(question_index)
    return failed


def submit(state: RunState, step_id: str, result: StepResult) -> tuple[RunState, bool]:
    """Record the result of ``step_id``. Returns the new state and whether it was a retry."""
    if step_id in state.recorded:
        if state.recorded[step_id].result != result:
            raise ProtocolViolation(
                f"step {step_id} was already answered with different content; a step is "
                "answered once")
        return state, True
    index = _require_outstanding(state, step_id)
    op = state.cases[index].stage
    _require_kind(op, result)
    state = state.model_copy(deep=True)
    case = state.cases[index]
    question_index = case.question_index if op is Op.ASK else None
    state.recorded[step_id] = Recorded(op=op, case_id=state.case_ids[index],
                                       question_index=question_index, result=result)
    case.outstanding = None
    _advance(state, index, op, result)
    return state, False


def _require_outstanding(state: RunState, step_id: str) -> int:
    if not step_id.startswith("s") or not step_id[1:].isdigit() or int(step_id[1:]) > state.issued:
        raise UnknownStep(f"step {step_id} was never issued by this run")
    index = next((i for i, case in enumerate(state.cases)
                  if case.outstanding is not None and case.outstanding.step_id == step_id), None)
    if index is None:
        raise ProtocolViolation(
            f"step {step_id} is no longer outstanding: its case was restarted after the step "
            "was lost. Ask for the next step.")
    return index


def _require_kind(op: Op, result: StepResult) -> None:
    if op is Op.ASK and result.ok is not None:
        raise ProtocolViolation("an ask step is answered with answer or error, not ok")
    if op is not Op.ASK and result.answer is not None:
        raise ProtocolViolation(f"a {op.value} step is answered with ok or error, not answer")


def _advance(state: RunState, index: int, op: Op, result: StepResult) -> None:
    case = state.cases[index]
    if result.error is not None and op is not Op.ASK:
        state.failed_cases[state.case_ids[index]] = f"{op.value} failed: {result.error}"
        _finish(case)
    elif op is Op.RESET:
        case.stage = Op.FEED
    elif op is Op.FEED:
        case.stage = Op.ASK
        _finish_if_asked(state, index)
    elif case.retry:
        case.retry.pop(0)
        case.question_index = case.retry[0] if case.retry else state.question_counts[index]
        _finish_if_asked(state, index)
    else:
        case.question_index += 1
        _finish_if_asked(state, index)


def _finish_if_asked(state: RunState, index: int) -> None:
    if state.cases[index].question_index >= state.question_counts[index]:
        _finish(state.cases[index])


def _finish(case: CaseState) -> None:
    case.finished, case.lane, case.stage = True, None, Op.RESET


def session_id(run_id: str, state: RunState, index: int) -> str:
    """The agent-facing session key for the current attempt at case ``index``."""
    case_id = state.case_ids[index]
    return f"{run_id[:12]}-{case_id}-{state.restarts.get(case_id, 0)}"
