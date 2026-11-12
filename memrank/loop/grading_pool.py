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
"""Grading each answer of a live run the moment it is final, beside the agent, never in its way.

Decision D1: a pool of judge workers grades answers while the lanes go on asking. A lane hands
an answer over and returns at once -- the queue is unbounded -- so a slow or failing judge never
slows the agent. Each verdict is appended to the run's event log, which drops it if the question
has been answered again meanwhile.

The pool's judging is fixed when it starts and logged then, before any verdict: which judge,
from the run's plan rather than from the answers (:attr:`RunStatus.uses_model`), because no answer
exists yet. A resumed run's pool must start the same judging its log already holds; another is
refused, since a run has one live judging and judging again is its own step (decision 0040).

Decision D5: a grading that raises is not a verdict. The question stays ungraded and the agent
goes on. Once every step is done, every answer still without a verdict is graded once more; if
any is still ungraded the run is not finished, and :class:`RunUngraded` says how many, the first
error, and the ``--resume`` that grades the rest.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, wait

from memrank.definitions.base import Verdict
from memrank.errors import MemrankError
from memrank.evaluation.constants import DEFAULT_JUDGE_WORKERS
from memrank.loop.event_log import EventLog
from memrank.loop.judge import RunGrading
from memrank.loop.live_answers import fold
from memrank.service.live_events import AnswerEvent, JudgingEvent
from memrank.service.live_rows import judging_row


class RunUngraded(MemrankError):
    """Every step is done but some answers have no verdict, so the run is not finished.

    Raised from the first grading error. ``ungraded`` is how many answers are left.
    """

    def __init__(self, message: str, *, ungraded: int) -> None:
        super().__init__(message)
        self.ungraded = ungraded


class JudgingChanged(MemrankError):
    """A run is resumed with a judge other than the one its answers are being graded by."""


class GradingPool:
    """The judge workers of one run, logging into ``log`` under ``grading``'s judge.

    Args:
        log: The run's event log.
        grading: The run's graders and judge model.
        resume: The command that continues the run, for :class:`RunUngraded` to name.
        workers: How many answers are graded at once.
        graded: Called with a count as verdicts are logged, from whichever worker logged them.
        explain: Turns a grading error into the one the run ends on (a provider's refusal named
            as such); the error itself when there is nothing to add.
    """

    def __init__(self, log: EventLog, grading: RunGrading, *, resume: str,
                 workers: int = DEFAULT_JUDGE_WORKERS,
                 graded: Callable[[int], None] = lambda count: None,
                 explain: Callable[[Exception], Exception] = lambda exc: exc) -> None:
        self.log, self.grading, self.resume = log, grading, resume
        self._graded, self._explain = graded, explain
        self._executor = ThreadPoolExecutor(max_workers=max(1, workers),
                                            thread_name_prefix="memrank-judge")
        self._lock = threading.Lock()
        self._futures: list[Future[None]] = []
        self._errors: dict[str, Exception] = {}

    def start(self) -> None:
        """Log the run's judging unless its log holds it, and queue every logged answer that
        has no verdict yet -- on a resume, what the last session left ungraded."""
        events = self.log.read_after(0)
        wanted = judging_row(self.grading.identity, judge_samples=self.grading.samples,
                             judge_prompt_version=self.grading.prompt_version)
        held = next((event for event in events if isinstance(event, JudgingEvent)), None)
        if held is None:
            self.log.judging(self.grading.identity, judge_samples=self.grading.samples,
                             judge_prompt_version=self.grading.prompt_version)
        else:
            started = judging_row(held.identity, judge_samples=held.judge_samples,
                                  judge_prompt_version=held.judge_prompt_version)
            if started != wanted:
                raise JudgingChanged(
                    f"This run's answers are graded by {_named(started.judge)}, not "
                    f"{_named(wanted.judge)}. Resume it with the judge it started with; to "
                    "grade it with another judge, judge the finished run again.")
        logged = fold(events)
        if logged.verdicts:
            self._graded(len(logged.verdicts))
        for answer in logged.ungraded():
            self.submit(answer)

    def submit(self, answer: AnswerEvent) -> None:
        """Queue ``answer`` for grading and return at once; a failed answer has nothing to grade."""
        if answer.outcome.status != "not_judged":
            return
        with self._lock:
            self._futures.append(self._executor.submit(self._grade, answer))

    def _grade(self, answer: AnswerEvent) -> None:
        try:
            verdict = self.grading.grade(answer.outcome)
        except Exception as exc:  # noqa: BLE001 - kept as ungraded, retried at the end (D5)
            with self._lock:
                self._errors[answer.question_id] = exc
            return
        with self._lock:
            self._errors.pop(answer.question_id, None)
        if self.log.verdict(question_id=answer.question_id, answer_ref=answer.answer_ref,
                            fields=verdict) is not None:
            self._graded(1)

    def finish(self) -> dict[str, Verdict]:
        """Every current answer's verdict, once each queued grading is done and every answer
        still without one has been graded once more; :class:`RunUngraded` if any is left."""
        self._drain()
        for answer in fold(self.log.read_after(0)).ungraded():
            self.submit(answer)  # the one retry (D5)
        self._drain()
        logged = fold(self.log.read_after(0))
        left = logged.ungraded()
        if left:
            raise self._ungraded(left)
        return logged.verdicts

    def close(self, *, wait_for_gradings: bool = True) -> None:
        """Stop the workers; a run that is stopping does not wait for gradings in flight."""
        self._executor.shutdown(wait=wait_for_gradings, cancel_futures=True)

    def _drain(self) -> None:
        while True:
            with self._lock:
                pending = [future for future in self._futures if not future.done()]
                self._futures = pending
            if not pending:
                return
            wait(pending)

    def _ungraded(self, left: list[AnswerEvent]) -> RunUngraded:
        with self._lock:
            errors = [self._errors[a.question_id] for a in left if a.question_id in self._errors]
        first = self._explain(errors[0]) if errors else None
        why = (str(first).splitlines() or [type(first).__name__])[0] if first else "no verdict"
        count = f"{len(left)} answer{'' if len(left) == 1 else 's'}"
        failure = RunUngraded(
            f"{count} could not be graded, so the run is not finished: grading failed, and so "
            f"did its retry. First error: {why}\nEvery answer is saved. Once the judge works "
            f"again, `{self.resume}` grades the rest and finishes the run.", ungraded=len(left))
        failure.__cause__ = first
        return failure


def _named(judge: str | None) -> str:
    return judge or "its own graders, with no judge model"
