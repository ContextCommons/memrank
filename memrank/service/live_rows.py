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
"""A run's live results as rows, and the one mapping between those rows and ``answers.json``.

The rows mirror the accounts tables ``run_answers``, ``run_judgings`` (the ``live`` one) and
``run_verdicts``, minus what the database owns (ids, ``run_id``, ``updated_at``). Rows carry no
event sequence (decided 2026-11-11): the server applies one contiguous range at a time, so an
event simply upserts its row. Every writer and reader goes through this module -- the API's
dual write, the backfill, the read route -- so "one-to-one with ``answers.json``" is the
round trip ``cases_of(rows_of(cases, ...), shells) == cases``, tested over the fixtures.

Two facts about a case live in the run's record, not in rows: its ``status``/``error`` (whether
its reset or feed failed) and its ``restarts``. The record's summary keeps every case without
its questions (:func:`~memrank.service.protocol.split_answers`); :func:`cases_of` takes those
as ``shells``. Without them -- a live run, before its record is sealed -- a case reads
``pending``.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from memrank.service.live_events import (
    AnswerEvent,
    JudgingEvent,
    LiveEvent,
    VerdictEvent,
)
from memrank.service.protocol import (
    CaseOutcome,
    Identity,
    JudgeIdentity,
    JudgeSource,
    QuestionOutcome,
)
from memrank.service.scoring import Provisional

_VERDICT_FIELDS = ("score", "passed", "rationale", "error", "probability", "decision")


class AnswerRow(BaseModel):
    """``accounts.run_answers``: one question's final answer, or its failure."""

    model_config = ConfigDict(frozen=True)

    question_id: str
    case_id: str
    position: int = Field(ge=0)
    question: str
    reference: list[str]
    grading: dict[str, Any]
    category: str | None
    status: Literal["answered", "failed"]
    answer: str | None
    error: str | None
    elapsed_ms: float | None
    retrieval_ms: float | None


class JudgingRow(BaseModel):
    """``accounts.run_judgings``: a judging's identity. ``judge`` is None for a model-free one."""

    model_config = ConfigDict(frozen=True)

    origin: Literal["live", "added"] = "live"
    judge: str | None
    judge_provider: str | None
    judge_model: str | None
    judge_endpoint: str | None
    judge_settings: dict[str, Any] | None
    judge_prompt_version: str | None
    judge_samples: int = Field(ge=0)
    #: Where the identity came from (``run``, or a backfill's attribution); ``run`` when
    #: model-free, since then nothing was attributed.
    judge_source: JudgeSource = "run"


class VerdictRow(BaseModel):
    """``accounts.run_verdicts``: the live judging's verdict on one question's current answer."""

    model_config = ConfigDict(frozen=True)

    question_id: str
    status: Literal["judged", "unjudged"]
    score: float | None
    passed: bool | None
    rationale: str | None
    error: str | None
    probability: float | None
    decision: dict[str, Any] | None


class LiveRows(BaseModel):
    """One run's rows: answers in plan order, its live judging, and that judging's verdicts."""

    model_config = ConfigDict(frozen=True)

    answers: list[AnswerRow] = Field(default_factory=list)
    judging: JudgingRow | None = None
    verdicts: list[VerdictRow] = Field(default_factory=list)


class LiveAnswers(BaseModel):
    """``GET .../answers``: every case from rows; ``provisional`` only while unsealed.

    ``judged`` says whether the run has a live judging, so a reader can tell an answer waiting
    for its verdict from one that will never get one (a ``--no-judge`` run).
    """

    cases: list[CaseOutcome]
    judged: bool
    provisional: Provisional | None = None


def judging_row(identity: JudgeIdentity | None, *, judge_samples: int,
                judge_prompt_version: str | None) -> JudgingRow:
    """The ``live`` judging row of a judge identity; all-None judge fields when model-free."""
    named = identity.model_dump() if identity is not None else {}
    return JudgingRow(
        judge=named.get("judge"), judge_provider=named.get("judge_provider"),
        judge_model=named.get("judge_model"), judge_endpoint=named.get("judge_endpoint"),
        judge_settings=named.get("judge_settings"), judge_prompt_version=judge_prompt_version,
        judge_samples=judge_samples, judge_source=named.get("judge_source", "run"))


def answer_row(outcome: QuestionOutcome, position: int) -> AnswerRow:
    """The answer row of a question whose answer is final (``not_judged``, ``judged``, ...)."""
    if outcome.status == "pending":
        raise ValueError(f"question {outcome.question_id!r} has no final answer; a pending "
                         f"question has no row")
    return AnswerRow(
        question_id=outcome.question_id, case_id=outcome.case_id, position=position,
        question=outcome.question, reference=outcome.reference, grading=outcome.grading,
        category=outcome.category, status="failed" if outcome.status == "failed" else "answered",
        answer=outcome.answer, error=None if outcome.status != "failed" else outcome.error,
        elapsed_ms=outcome.elapsed_ms, retrieval_ms=outcome.retrieval_ms)


def _verdict_row(outcome: QuestionOutcome) -> VerdictRow | None:
    if outcome.status not in ("judged", "unjudged"):
        return None
    status: Literal["judged", "unjudged"] = "judged" if outcome.status == "judged" else "unjudged"
    return VerdictRow(question_id=outcome.question_id, status=status,
                      **{name: getattr(outcome, name) for name in _VERDICT_FIELDS})


def rows_of(cases: Iterable[CaseOutcome], identity: Identity, *, judged: bool) -> LiveRows:
    """The rows of a sealed run's cases (``answers.json``, or questions inline in a record).

    Args:
        cases: Every case with all of its questions, in plan order.
        identity: The run's identity, whose judge fields name the live judging.
        judged: Whether the run was judged -- false for ``--no-judge`` and for a run still
            awaiting its judge, which therefore have no judging row and no verdicts.
    """
    questions = [q for case in cases for q in case.questions]
    answers = [answer_row(q, position) for position, q in enumerate(questions)]
    if not judged:
        return LiveRows(answers=answers)
    judging = judging_row(identity.judge_identity(), judge_samples=identity.judge_samples,
                          judge_prompt_version=identity.judge_prompt_version)
    verdicts = [row for row in map(_verdict_row, questions) if row is not None]
    return LiveRows(answers=answers, judging=judging, verdicts=verdicts)


def _question(answer: AnswerRow, verdict: VerdictRow | None, *, judged: bool) -> QuestionOutcome:
    """The question as :func:`memrank.service.scoring.apply_verdicts` would have left it."""
    outcome = QuestionOutcome(
        case_id=answer.case_id, question_id=answer.question_id, question=answer.question,
        reference=answer.reference, category=answer.category, grading=answer.grading,
        status="failed" if answer.status == "failed" else "not_judged", answer=answer.answer,
        error=answer.error, elapsed_ms=answer.elapsed_ms, retrieval_ms=answer.retrieval_ms)
    if answer.status == "failed":
        # a failure is the agent's outcome: it scores 0 in a judged run, and has no verdict
        return outcome.model_copy(update={"score": 0.0 if judged else None})
    if verdict is None:
        return outcome
    return outcome.model_copy(update={
        "status": verdict.status, **{name: getattr(verdict, name) for name in _VERDICT_FIELDS}})


def cases_of(rows: LiveRows, shells: Iterable[CaseOutcome] | None = None) -> list[CaseOutcome]:
    """The cases ``rows`` hold, each with its questions in plan order.

    Args:
        rows: The run's rows.
        shells: The record summary's cases (status, error, restarts; questions ignored), in plan
            order. None for a live run, whose cases read ``pending`` in order of first answer.
    """
    judged = rows.judging is not None
    verdicts = {v.question_id: v for v in rows.verdicts}
    by_case: dict[str, list[QuestionOutcome]] = {}
    for answer in sorted(rows.answers, key=lambda a: a.position):
        by_case.setdefault(answer.case_id, []).append(
            _question(answer, verdicts.get(answer.question_id), judged=judged))
    if shells is None:
        return [CaseOutcome(case_id=case_id, status="pending", questions=questions)
                for case_id, questions in by_case.items()]
    return [shell.model_copy(update={"questions": by_case.get(shell.case_id, [])})
            for shell in shells]


def apply(events: Iterable[LiveEvent], rows: LiveRows | None = None) -> LiveRows:
    """``rows`` with ``events`` folded in, in order.

    An answer event upserts its question's row and deletes the live verdict on it, so a
    re-answered question is ungraded until its new verdict arrives. A judging event creates the
    live judging (again, identically, on a resume). A verdict event upserts its question's
    verdict. Anything a database constraint would refuse is refused here, loudly.
    """
    current = rows or LiveRows()
    answers = {a.question_id: a for a in current.answers}
    verdicts = {v.question_id: v for v in current.verdicts}
    judging = current.judging
    for event in events:
        if isinstance(event, AnswerEvent):
            answers[event.question_id] = answer_row(event.outcome, event.position)
            verdicts.pop(event.question_id, None)
        elif isinstance(event, JudgingEvent):
            judging = _one_judging(judging, event)
        else:
            verdicts[event.question_id] = _checked_verdict(event, answers, judging)
    ordered = sorted(answers.values(), key=lambda a: a.position)
    return LiveRows(answers=ordered, judging=judging,
                    verdicts=[verdicts[a.question_id] for a in ordered
                              if a.question_id in verdicts])


def _one_judging(existing: JudgingRow | None, event: JudgingEvent) -> JudgingRow:
    started = judging_row(event.identity, judge_samples=event.judge_samples,
                          judge_prompt_version=event.judge_prompt_version)
    if existing is not None and existing != started:
        raise ValueError(f"the run's live judging is {existing.judge!r}; event {event.seq} "
                         f"starts a different one, {started.judge!r}")
    return started


def _checked_verdict(event: VerdictEvent, answers: dict[str, AnswerRow],
                     judging: JudgingRow | None) -> VerdictRow:
    if judging is None:
        raise ValueError(f"verdict event {event.seq} precedes the run's judging event")
    answer = answers.get(event.question_id)
    if answer is None or answer.status != "answered":
        raise ValueError(f"verdict event {event.seq} grades {event.question_id!r}, which has "
                         f"no answer to grade")
    return VerdictRow(question_id=event.question_id, status=event.status,
                      **{name: getattr(event, name) for name in _VERDICT_FIELDS})
