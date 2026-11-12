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
"""A live run's answers in its event log, and what the log says is graded.

Each accepted step's :class:`~memrank.service.protocol.Ack` names the questions it made final;
:class:`Recorder` logs each as an :class:`~memrank.service.live_events.AnswerEvent` under the
step's id and hands it to the grading pool. The log, not memory, is what the run's verdicts are
read back from (:func:`fold`), so a resumed run picks up exactly where its log stops.

A step can be accepted and its answer never logged -- the runner stopped in between, or the run
was recorded before runs had a log. :meth:`Recorder.reconcile` closes that gap once every step is
done: it compares the log with the result the service builds and logs whatever the log lacks or
holds differently. Such an answer has no step id to name it, so it is named by its content
(:func:`content_ref`), which also keeps a later, different answer distinct from it.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from memrank.definitions.base import Verdict
from memrank.loop.event_log import EventLog
from memrank.service.live_events import AnswerEvent, LiveEvent, VerdictEvent
from memrank.service.protocol import Ack, QuestionOutcome, RunResult

if TYPE_CHECKING:
    from memrank.loop.grading_pool import GradingPool

#: The verdict fields a :class:`VerdictEvent` carries, as :func:`apply_verdicts` reads them.
VERDICT_FIELDS = ("status", "score", "passed", "rationale", "error", "probability", "decision")

#: The statuses of a question whose answer is final and not yet graded, as a result records it.
FINAL = ("not_judged", "failed")


@dataclass
class Logged:
    """What a run's log holds: each question's current answer, and the verdicts on them."""

    answers: dict[str, AnswerEvent] = field(default_factory=dict)
    verdicts: dict[str, Verdict] = field(default_factory=dict)

    def ungraded(self) -> list[AnswerEvent]:
        """Every current answer the agent gave that has no verdict, in log order."""
        waiting = [event for event in self.answers.values()
                   if event.outcome.status == "not_judged"
                   and event.question_id not in self.verdicts]
        return sorted(waiting, key=lambda event: event.seq)


def fold(events: Iterable[LiveEvent]) -> Logged:
    """The current answers and their verdicts, as the log's events leave them.

    A later answer for a question replaces the earlier one and ungrades it; a verdict counts
    only for the answer it names, which the log guarantees is current when it is appended.
    """
    logged = Logged()
    for event in events:
        if isinstance(event, AnswerEvent):
            logged.answers[event.question_id] = event
            logged.verdicts.pop(event.question_id, None)
        elif isinstance(event, VerdictEvent):
            current = logged.answers.get(event.question_id)
            if current is not None and current.answer_ref == event.answer_ref:
                logged.verdicts[event.question_id] = event.model_dump(include=set(VERDICT_FIELDS))
    return logged


def content_ref(outcome: QuestionOutcome) -> str:
    """A name for an answer that has no step id: a digest of the question as recorded."""
    digest = hashlib.sha256(outcome.model_dump_json().encode("utf-8")).hexdigest()
    return f"result-{digest[:16]}"


@dataclass
class Recorder:
    """Where a live run's final answers go: its event log, and its grading pool when judged."""

    log: EventLog
    pool: GradingPool | None = None

    def answered(self, ack: Ack) -> None:
        """Log every question ``ack``'s step made final, and queue its answer for grading."""
        for final in ack.outcomes:
            self._log(final.position, ack.step_id, final.outcome)

    def reconcile(self, result: RunResult) -> int:
        """Log every final question ``result`` holds that the log lacks or holds differently.

        Returns how many were logged. Each is queued for grading like any other answer.
        """
        logged = fold(self.log.read_after(0)).answers
        questions = [q for case in result.cases for q in case.questions]
        missing = [(position, q) for position, q in enumerate(questions)
                   if q.status in FINAL and (q.question_id not in logged
                                             or logged[q.question_id].outcome != q)]
        for position, question in missing:
            self._log(position, content_ref(question), question)
        return len(missing)

    def _log(self, position: int, answer_ref: str, outcome: QuestionOutcome) -> None:
        event = self.log.answer(position=position, answer_ref=answer_ref, outcome=outcome)
        if self.pool is not None:
            self.pool.submit(event)
