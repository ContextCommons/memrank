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
"""The evaluation service as plain methods: create a run, next step, post a step, status, result.

``memrank run`` calls these directly, in its own process, so an ordinary evaluation needs no
server. :mod:`memrank.service.app` exposes the very same object over HTTP for a shared service.
Either way the rules live in :mod:`memrank.service.machine` and nowhere else.

Nothing here judges. The result records every answer ungraded; ``memrank run`` judges it in
its own process with the organisation's key (decision 0037, :mod:`memrank.loop.judge`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from memrank.errors import MemrankError
from memrank.runs.registry import mint_run_id
from memrank.service import machine
from memrank.service.cases import Plan, build_plan, check_unchanged
from memrank.service.machine import Outstanding, RunState
from memrank.service.protocol import (
    Ack,
    Op,
    Progress,
    Reopened,
    RunCreate,
    RunCreated,
    RunResult,
    RunStatus,
    Step,
    StepResult,
)
from memrank.service.results import build_result
from memrank.service.store import RunStore


class RunNotFinished(MemrankError):
    """The result was asked for before every step was answered.

    Refused rather than returned partial: a result lists every question with its reference
    answer, and handing references out mid-run would show them to the runner.
    """


def _step(run_id: str, plan: Plan, state: RunState, out: Outstanding | None) -> Step:
    if out is None:
        return Step(op=Op.DONE)
    case = plan.case(state.case_ids[out.case])
    question_index = state.cases[out.case].question_index
    step = Step(step_id=out.step_id, op=out.op, case_id=case.id,
                session_id=machine.session_id(run_id, state, out.case),
                progress=Progress(case=out.case + 1, cases=len(state.case_ids),
                                  question=question_index + 1, questions=len(case.queries)))
    if out.op is Op.FEED:
        step.sessions = list(case.sessions)
    elif out.op is Op.ASK:
        step.question = case.question(question_index)
    return step


def _plan(run_id: str, request: RunCreate, state: RunState) -> Plan:
    """The run's plan, refused when its cases are not the ones the run started on."""
    plan = build_plan(request)
    check_unchanged(plan, state.cases_digest, run_id)
    return plan


def _status(run_id: str, request: RunCreate, state: RunState) -> RunStatus:
    """How far the run has got, counted from the recorded steps (see :class:`RunStatus`)."""
    fed = {r.case_id: r.result.elapsed_ms for r in state.recorded.values()
           if r.op is Op.FEED and r.result.ok}
    asks = [r.result.elapsed_ms for r in state.recorded.values() if r.op is Op.ASK]
    asked = {(r.case_id, r.question_index) for r in state.recorded.values() if r.op is Op.ASK}
    failed = [i for i, case_id in enumerate(state.case_ids) if case_id in state.failed_cases]
    unasked = sum(state.question_counts[i] - state.cases[i].question_index for i in failed)
    return RunStatus(run_id=run_id, evaluation=_plan(run_id, request, state).evaluation,
                     judge=request.judge, cases=len(state.case_ids),
                     questions=sum(state.question_counts),
                     cases_finished=sum(case.finished for case in state.cases),
                     cases_fed=len(fed) + len(failed), fed_ms=sum(fed.values()),
                     questions_done=len(asked) + unasked, asked_ms=sum(asks), done=state.done)


@dataclass
class EvaluationService:
    """Owns the cases and the step order for every run in ``store``."""

    store: RunStore

    def create(self, request: RunCreate) -> RunCreated:
        plan = build_plan(request)
        run_id = mint_run_id(re.sub(r"[^A-Za-z0-9._-]", "-", plan.evaluation))
        state = machine.new_state({c.id: len(c.queries) for c in plan.cases})
        state.cases_digest = plan.cases_digest
        self.store.create(run_id, request, state)
        return RunCreated(run_id=run_id, cases=len(plan.cases),
                          questions=sum(len(c.queries) for c in plan.cases))

    def next(self, run_id: str, lane: int = 0, lanes: int = 1) -> Step:
        with self.store.transaction(run_id):
            request, state = self.store.load(run_id)
            state, out = machine.next_step(state, lane, lanes)
            self.store.save_state(run_id, state)
        return _step(run_id, _plan(run_id, request, state), state, out)

    def post(self, run_id: str, step_id: str, result: StepResult) -> Ack:
        with self.store.transaction(run_id):
            _, state = self.store.load(run_id)
            state, duplicate = machine.submit(state, step_id, result)
            self.store.save_state(run_id, state)
        return Ack(step_id=step_id, duplicate=duplicate)

    def retry_failed(self, run_id: str) -> Reopened:
        """Run every failed case, and every failed answer, again as a fresh restart.

        Only ever on request (``memrank run --resume <id> --retry-failed``): a plain resume
        leaves a failed case or answer failed.
        """
        with self.store.transaction(run_id):
            request, state = self.store.load(run_id)
            _plan(run_id, request, state)  # refused on other cases, as every resume is
            state, reopened = machine.reopen_failed(state)
            self.store.save_state(run_id, state)
        return Reopened(run_id=run_id, cases=reopened)

    def status(self, run_id: str) -> RunStatus:
        """How far the run has got. Counts only: no answer and no reference."""
        with self.store.transaction(run_id):
            request, state = self.store.load(run_id)
        return _status(run_id, request, state)

    def result(self, run_id: str) -> RunResult:
        """Every answer, failure and latency of a finished run, ungraded."""
        request, state = self.store.load(run_id)
        if not state.done:
            raise RunNotFinished(f"run {run_id} still has steps to answer; finish it first")
        return build_result(run_id, request=request, plan=_plan(run_id, request, state),
                            state=state)
