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

Each case goes reset -> feed -> ask, ask, ... A runner asks for work from one of two **pools**
(:class:`~memrank.service.protocol.Pool`): ``ingest`` hands out a case's reset and then its feed,
``answer`` hands out questions of fed cases. How many of each the runner carries at once is its
own choice (``memrank run --ingest-concurrency`` and ``--answer-concurrency``); the service only
says what may be carried now. Five rules:

1. Every case starts with a reset, and its reset and its feed are each the only step it has
   outstanding. The ingest pool hands them out in plan order: the lowest-numbered case whose
   next step is one of them and which has nothing outstanding.
2. No question of a case is revealed until that case's feed is acknowledged. The answer pool
   hands out the next question of the lowest-numbered fed case that still has one, so questions
   of one case may be outstanding together, spread across cases. ``in_order``
   (``--questions-in-order``) hands a case its next question only once its last is answered.
3. A step is answered once. Re-posting identical content is a harmless retry; different
   content is refused.
4. A crashed runner resumes from here. Within one runner nothing is lost: every step it is
   handed is carried and posted, or the runner stops. So a runner that starts says so
   (:func:`begin`), and every step outstanding then was lost with the runner before it. A lost
   reset is simply handed out again. Every case that was fed and is not finished restarts from
   a reset under a fresh session id, whether or not it had a step outstanding: a feed or ask in
   flight may or may not have reached the agent, and the agent's memory of a conversation is
   never assumed to have survived the stop. The case is fed again -- never twice into the same
   memory -- keeps the answers it has, and asks only the questions still owed, the lost ones
   among them.
5. A pool with nothing to hand out now says so (``None``). The run is finished when every case
   is (:attr:`RunState.done`): a runner holding no step and offered none by either pool has
   nothing left to wait for.

A case whose reset or feed failed is finished, and a resume leaves it so; so is a case that
completed with some answers failed. Only an explicit :func:`reopen_failed` (``memrank run
--resume <id> --retry-failed``) runs either again, as a restart: a reset under a fresh session id,
a new feed, and then only the questions still owed -- every question of a failed case from where
it stopped, or just the failed answers of a completed one. Every other case is untouched.

Every answer is recorded against its own question, so the order answers come back in changes
nothing recorded but the order of ``recorded``.

Pure functions over :class:`RunState`: no I/O, no clock, so every transition is testable as a
row in a table.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator

from memrank.errors import MemrankError
from memrank.service.protocol import Op, Pool, StepResult


class ProtocolViolation(MemrankError):
    """A runner did something the rules forbid. The state is left unchanged."""


class UnknownStep(ProtocolViolation):
    """A step id this run never issued."""


class Outstanding(BaseModel):
    step_id: str
    op: Op
    #: The index of the case the step belongs to.
    case: int
    #: The question an ask asks; None for a reset or a feed.
    question_index: int | None = None


class Recorded(BaseModel):
    """An accepted step result, kept so a retry can be recognised as one."""

    op: Op
    case_id: str
    question_index: int | None = None
    result: StepResult


class CaseState(BaseModel):
    """Where one case stands: its next step, and what it has outstanding.

    A run saved while lanes held cases also has a ``lane``; it is ignored.
    """

    stage: Op = Op.RESET
    #: The next question to hand out; the case's question count once none is left.
    question_index: int = 0
    finished: bool = False
    #: The steps handed out and not yet answered, in the order they were handed out.
    outstanding: list[Outstanding] = Field(default_factory=list)
    #: The question indices a retry still has to hand out, in order; empty outside a retry.
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

    @model_validator(mode="before")
    @classmethod
    def _one_outstanding(cls, data: Any) -> Any:
        """A run saved when each case had at most one step outstanding, read as this shape."""
        if not isinstance(data, dict) or "cases" not in data:
            return data
        counts = data.get("question_counts", [])
        cases = [_as_list(case, counts[i]) if isinstance(case, dict) and i < len(counts)
                 else case for i, case in enumerate(data.get("cases", []))]
        return {**data, "cases": cases}

    @property
    def done(self) -> bool:
        return all(case.finished for case in self.cases)


def _copied(state: RunState) -> RunState:
    """A copy of ``state`` a transition may change, leaving ``state`` as it was.

    Every part a transition changes is copied; the recorded results are shared, because a
    transition only ever adds one. Deep-copying them too would make every step cost as much
    as the whole run so far.
    """
    return state.model_copy(update={
        "cases": [case.model_copy(deep=True) for case in state.cases],
        "restarts": dict(state.restarts), "failed_cases": dict(state.failed_cases),
        "recorded": dict(state.recorded)})


def _as_list(case: dict[str, Any], count: int) -> dict[str, Any]:
    """One saved case's single ``outstanding`` step (or None) as the list it is now.

    Its ``question_index`` was the question its outstanding ask asked, not yet handed out past;
    it moves on here the way handing out an ask moves it now (:func:`_hand_out`).
    """
    if isinstance(case.get("outstanding", []), list):
        return case
    out = case["outstanding"]
    case = {**case, "outstanding": [] if out is None else [dict(out)]}
    if out is not None and out.get("op") == Op.ASK.value:
        moved = CaseState.model_validate({**case, "outstanding": []})
        case["outstanding"][0]["question_index"] = _hand_out(moved, count)
        case.update(question_index=moved.question_index, retry=moved.retry)
    return case


def new_state(question_counts: dict[str, int]) -> RunState:
    return RunState(case_ids=list(question_counts), question_counts=list(question_counts.values()),
                    cases=[CaseState() for _ in question_counts])


def begin(state: RunState) -> tuple[RunState, list[str]]:
    """A runner starts: every step outstanding now was lost with the one before it (rule 4).

    Returns the new state and the ids of the cases restarted, in plan order: every unfinished
    case that was fed or had its feed in flight. A lost reset is dropped and handed out again,
    under the same session id. A run with no such case is returned unchanged.
    """
    state = _copied(state)
    restarted = []
    for index, case in enumerate(state.cases):
        if not case.finished and (case.stage is Op.ASK
                                  or any(out.op is not Op.RESET for out in case.outstanding)):
            _restart_case(state, index)
            restarted.append(state.case_ids[index])
        else:
            case.outstanding = []
    return state, restarted


def next_step(state: RunState, pool: Pool,
              in_order: bool = False) -> tuple[RunState, Outstanding | None]:
    """The step ``pool`` hands out now (rules 1 and 2), or None when it has none now.

    Returns a new state; the input is not modified.
    """
    state = _copied(state)
    index = _ingestible(state) if pool is Pool.INGEST else _answerable(state, in_order)
    if index is None:
        return state, None
    case = state.cases[index]
    state.issued += 1
    question = _hand_out(case, state.question_counts[index]) if case.stage is Op.ASK else None
    out = Outstanding(step_id=f"s{state.issued}", op=case.stage, case=index,
                      question_index=question)
    case.outstanding.append(out)
    return state, out


def _ingestible(state: RunState) -> int | None:
    """The lowest unfinished case whose next step is a reset or a feed, with none outstanding."""
    return next((i for i, case in enumerate(state.cases) if not case.finished
                 and case.stage is not Op.ASK and not case.outstanding), None)


def _answerable(state: RunState, in_order: bool) -> int | None:
    """The lowest fed case with a question still to hand out; in order, with none outstanding."""
    return next((i for i, case in enumerate(state.cases) if not case.finished
                 and case.stage is Op.ASK and _owes(state, i)
                 and not (in_order and case.outstanding)), None)


def _owes(state: RunState, index: int) -> bool:
    """Whether case ``index`` has a question still to hand out."""
    return state.cases[index].question_index < state.question_counts[index]


def _hand_out(case: CaseState, count: int) -> int:
    """The next question of ``case``, moving it on to the one after (``count`` when none)."""
    question = case.question_index
    if case.retry:
        case.retry.pop(0)
        case.question_index = case.retry[0] if case.retry else count
    else:
        case.question_index += 1
    return question


def _restart_case(state: RunState, index: int) -> None:
    """Abandon the lost steps: the case begins again from a reset, keeping its answers, and
    owing again every question it had asked and not had answered."""
    case_id = state.case_ids[index]
    state.restarts[case_id] = state.restarts.get(case_id, 0) + 1
    case = state.cases[index]
    lost = [out.question_index for out in case.outstanding if out.question_index is not None]
    if lost:
        _owe_again(case, lost, state.question_counts[index])
    case.stage = Op.RESET
    case.outstanding = []


def _owe_again(case: CaseState, lost: list[int], count: int) -> None:
    """Put the ``lost`` questions back before the ones ``case`` still has to hand out.

    What is owed is a run to the end of the case when it can be -- the shape a case asked in
    order has -- and a retry list of the questions otherwise.
    """
    owed = sorted(lost + (case.retry or list(range(case.question_index, count))))
    case.retry = [] if owed == list(range(owed[0], count)) else owed
    case.question_index = owed[0]


def reopen_failed(state: RunState) -> tuple[RunState, list[str]]:
    """Every failed case and every case with a failed answer, reopened as a restart.

    Returns the new state and the reopened case ids, in plan order. Each gets the restart a lost
    step gets (:func:`_restart_case`): it begins from a reset, and its session id moves on so the
    agent is never fed twice into the same memory. A failed case asks its remaining questions and
    its failure is cleared. A complete case with failed answers asks only those questions again;
    its successful answers are kept, and each new answer replaces the failed one. Other cases are
    not touched, and a run with nothing failed is returned unchanged.
    """
    state = _copied(state)
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
    index, out = _require_outstanding(state, step_id)
    _require_kind(out.op, result)
    state = _copied(state)
    case = state.cases[index]
    state.recorded[step_id] = Recorded(op=out.op, case_id=state.case_ids[index],
                                       question_index=out.question_index, result=result)
    case.outstanding = [held for held in case.outstanding if held.step_id != step_id]
    _advance(state, index, out.op, result)
    return state, False


def _require_outstanding(state: RunState, step_id: str) -> tuple[int, Outstanding]:
    if not step_id.startswith("s") or not step_id[1:].isdigit() or int(step_id[1:]) > state.issued:
        raise UnknownStep(f"step {step_id} was never issued by this run")
    found = next(((i, out) for i, case in enumerate(state.cases) for out in case.outstanding
                  if out.step_id == step_id), None)
    if found is None:
        raise ProtocolViolation(
            f"step {step_id} is no longer outstanding: a runner that started since treated it "
            "as lost. Ask for the next step.")
    return found


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
    else:
        _finish_if_asked(state, index)


def _finish_if_asked(state: RunState, index: int) -> None:
    """Finish the case once every question is handed out and every answer is in."""
    if not _owes(state, index) and not state.cases[index].outstanding:
        _finish(state.cases[index])


def _finish(case: CaseState) -> None:
    case.finished, case.stage = True, Op.RESET


def session_id(run_id: str, state: RunState, index: int) -> str:
    """The agent-facing session key for the current attempt at case ``index``."""
    case_id = state.case_ids[index]
    return f"{run_id[:12]}-{case_id}-{state.restarts.get(case_id, 0)}"
