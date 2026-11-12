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
"""A run's live events, appended to ``events.jsonl`` in its folder and never rewritten whole.

The outbox of a live run (:mod:`memrank.service.live_events`): every answer as it becomes final
and every verdict as it lands, numbered from 1 without gaps. It has its own lock -- never the
run store's -- so an append never waits on a ``run.json`` rewrite. A crash can leave the last
line torn; reading ignores it, and opening the log cuts it off so the next append starts a
whole line. A torn line anywhere else is corruption and is refused.

The log also knows each question's current answer (``answer_ref``), so a verdict on an answer
that has since been replaced is dropped here, under the same lock, and never logged.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from pydantic import TypeAdapter, ValidationError

from memrank.service.live_events import AnswerEvent, JudgingEvent, LiveEvent, VerdictEvent
from memrank.service.protocol import JudgeIdentity, QuestionOutcome

#: The log's file name in a run folder.
EVENTS_FILE = "events.jsonl"

_EVENT: TypeAdapter[LiveEvent] = TypeAdapter(LiveEvent)

_E = TypeVar("_E", AnswerEvent, JudgingEvent, VerdictEvent)

#: Called with the new event's ``seq`` after each append, outside the log's lock.
AppendListener = Callable[[int], None]


def _parse(lines: list[bytes], path: Path, start: int = 0) -> list[LiveEvent]:
    """Every whole event in ``lines``, the file's lines from index ``start`` on; a torn last
    line is left out, any other is refused."""
    events: list[LiveEvent] = []
    for index, line in enumerate(lines, start=start):
        last = index == start + len(lines) - 1
        if last and not line.endswith(b"\n"):
            break  # a write the process did not finish
        try:
            events.append(_EVENT.validate_json(line))
        except ValidationError as exc:
            raise ValueError(f"{path} line {index + 1} is not an event: {exc}") from exc
    return events


class EventLog:
    """The append-only event log of one run folder."""

    def __init__(self, folder: Path) -> None:
        self.path = folder / EVENTS_FILE
        self._lock = threading.Lock()
        self._listeners: list[AppendListener] = []
        self._seq = 0
        self._current: dict[str, str] = {}
        with self._lock:
            self._recover()

    def _recover(self) -> None:
        """Resume numbering and current answers from what is on disk; cut a torn last line."""
        if not self.path.exists():
            return
        data = self.path.read_bytes()
        lines = data.splitlines(keepends=True)
        events = _parse(lines, self.path)
        if lines and not lines[-1].endswith(b"\n"):
            with self.path.open("r+b") as handle:
                handle.truncate(len(data) - len(lines[-1]))
        for seq, event in enumerate(events, start=1):
            if event.seq != seq:
                raise ValueError(f"{self.path} holds event {event.seq} where {seq} belongs")
            if isinstance(event, AnswerEvent):
                self._current[event.question_id] = event.answer_ref
        self._seq = len(events)

    @property
    def through_seq(self) -> int:
        """The last event's number; 0 when the log is empty."""
        with self._lock:
            return self._seq

    def on_append(self, listener: AppendListener) -> None:
        """Call ``listener`` with each new event's number -- the uploader's wake-up."""
        self._listeners.append(listener)

    def answer(self, *, position: int, answer_ref: str, outcome: QuestionOutcome) -> AnswerEvent:
        """Log that ``outcome``'s answer is final; it becomes the question's current answer."""
        event = self._append(lambda seq: AnswerEvent(
            seq=seq, question_id=outcome.question_id, case_id=outcome.case_id,
            position=position, answer_ref=answer_ref, outcome=outcome))
        assert event is not None  # an answer is always logged
        return event

    def judging(self, identity: JudgeIdentity | None, *, judge_samples: int,
                judge_prompt_version: str | None) -> JudgingEvent:
        """Log that the judge pool started under this judge (None: no grader calls a model)."""
        event = self._append(lambda seq: JudgingEvent(
            seq=seq, identity=identity, judge_samples=judge_samples,
            judge_prompt_version=judge_prompt_version))
        assert event is not None  # a judging is always logged
        return event

    def verdict(self, *, question_id: str, answer_ref: str,
                fields: dict[str, Any]) -> VerdictEvent | None:
        """Log a verdict on ``answer_ref``; None, logging nothing, when it is no longer current.

        Args:
            question_id: The question graded.
            answer_ref: The answer the judge graded.
            fields: The verdict: ``status`` and any of ``score``, ``passed``, ``rationale``,
                ``error``, ``probability``, ``decision``.
        """
        def build(seq: int) -> VerdictEvent | None:
            if self._current.get(question_id) != answer_ref:
                return None  # the question was answered again while this one was judged
            return VerdictEvent(seq=seq, question_id=question_id, answer_ref=answer_ref,
                                **fields)
        return self._append(build)

    def read_after(self, seq: int) -> list[LiveEvent]:
        """Every whole event numbered above ``seq``, in order.

        Event ``n`` is line ``n`` -- numbering and appending keep it so -- so only the lines
        after ``seq`` are parsed: the uploader reads the log's tail on every send.
        """
        with self._lock:
            if not self.path.exists():
                return []
            lines = self.path.read_bytes().splitlines(keepends=True)
        start = max(0, seq)
        return _parse(lines[start:], self.path, start)

    def _append(self, build: Callable[[int], _E | None]) -> _E | None:
        """Build the next event under the lock, write it whole, then tell the listeners."""
        with self._lock:
            event = build(self._seq + 1)
            if event is None:
                return None
            with self.path.open("ab") as handle:
                handle.write(event.model_dump_json().encode("utf-8") + b"\n")
                handle.flush()
                os.fsync(handle.fileno())
            self._seq = event.seq
            if isinstance(event, AnswerEvent):
                self._current[event.question_id] = event.answer_ref
        for listener in list(self._listeners):
            listener(event.seq)
        return event
