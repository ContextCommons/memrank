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
"""The events a live ``memrank run`` logs and uploads, and the wire contract that carries them.

A run's local folder is its source of truth (decision 0037). Beside ``run.json`` the runner
keeps ``events.jsonl`` (:class:`memrank.loop.event_log.EventLog`): one numbered event per line,
appended as each answer becomes final and each verdict lands. An uploader ships numbered ranges
of it, and the API folds each range into rows (:func:`memrank.service.live_rows.apply`). Three
kinds of event:

- :class:`AnswerEvent` -- a question's answer is final: the question as the service computed it
  (:class:`~memrank.service.protocol.QuestionOutcome`, ``not_judged`` or ``failed``). A later
  answer event for the same question replaces it, and ungrades it until its new verdict lands.
- :class:`JudgingEvent` -- the judge pool started, creating the run's ``live`` judging. Its
  identity is None when no grader calls a model (a model-free judging).
- :class:`VerdictEvent` -- a verdict on the question's answer named by ``answer_ref``. The log
  appends one only while that is still the question's current answer, so the server never
  receives a stale verdict.

Wire contract (the API implements it; nothing here talks HTTP):

- ``POST /orgs/{org}/runs/{id}/events`` with an :class:`Upload`, authenticated as the member
  who registered the run (``runs:write``). Answers :class:`UploadAck`. The server accepts only
  the next contiguous range: ``from_seq`` must be its ``through_seq + 1``, else ``409 seq_gap``
  carrying the server's ``through_seq``. A range wholly at or below ``through_seq`` is
  acknowledged as already held, changing nothing. Rows are upserted and
  ``runs.live_through_seq`` advanced in one transaction by a conditional update
  (``where live_through_seq = <expected>``); losing it to another writer is
  ``409 seq_conflict``. A body over :data:`MAX_UPLOAD_BYTES` is ``413 upload_too_large``; an
  event that contradicts the rows refuses the whole range, ``422 bad_upload``; a finished run
  takes none, ``409 run_sealed``. ``retry_after_s``, on the ack or on
  ``429 too_many_uploads``, is how long the server asks the uploader to wait.
- ``GET /orgs/{org}/runs/{id}/answers`` -> :class:`~memrank.service.live_rows.LiveAnswers`:
  every case from the rows, whether the run has a judging (``judged``), and for an unsealed
  run the provisional score (:func:`memrank.service.scoring.provisional`), computed on that
  read and never stored.
- The run registration answer (``PUT /orgs/{org}/runs/{id}``) gains ``live``
  (:class:`LiveEndpoint`) for an agent run. A runner that gets none refuses before its first
  step, naming the API it needs: there is no end-only fallback.
- Each accepted upload publishes :data:`RUN_GRADED` ``{run_id, org_id}`` -- a notification with
  no payload; a watcher refetches the answers.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from memrank.service.protocol import JudgeIdentity, QuestionOutcome

#: The ceiling on one upload's body. A size, not a time (decision R3): events appended while a
#: send is in flight ride the next one, split only here.
MAX_UPLOAD_BYTES = 1_048_576

#: The run-domain event an accepted upload publishes.
RUN_GRADED = "run.graded"

#: The judging the run's own judge pool produces, as against one added afterwards (0040).
LIVE_ORIGIN = "live"

#: The question statuses an answer event may carry: final, and not yet graded.
ANSWER_STATUSES = ("not_judged", "failed")


class AnswerEvent(BaseModel):
    """A question's answer is final; ``answer_ref`` is the step that produced it."""

    model_config = ConfigDict(frozen=True)

    seq: int = Field(ge=1)
    kind: Literal["answer"] = "answer"
    question_id: str
    case_id: str
    #: The question's place in the run's plan order, across every case.
    position: int = Field(ge=0)
    answer_ref: str = Field(min_length=1)
    outcome: QuestionOutcome

    @model_validator(mode="after")
    def _matches_outcome(self) -> AnswerEvent:
        if (self.outcome.question_id, self.outcome.case_id) != (self.question_id, self.case_id):
            raise ValueError(f"answer event for {self.question_id!r} carries the outcome of "
                             f"{self.outcome.question_id!r}")
        if self.outcome.status not in ANSWER_STATUSES:
            raise ValueError(f"an answer event's outcome is one of {ANSWER_STATUSES}, "
                             f"not {self.outcome.status!r}")
        return self


class JudgingEvent(BaseModel):
    """The judge pool started; the run's ``live`` judging is this one."""

    model_config = ConfigDict(frozen=True)

    seq: int = Field(ge=1)
    kind: Literal["judging"] = "judging"
    #: None when no grader calls a model (decided 2026-11-11: model-free judgings are stored).
    identity: JudgeIdentity | None
    judge_samples: int = Field(ge=0)
    judge_prompt_version: str | None = None

    @model_validator(mode="after")
    def _model_free_takes_no_samples(self) -> JudgingEvent:
        if self.identity is None and (self.judge_samples or self.judge_prompt_version):
            raise ValueError("a model-free judging has no samples and no prompt version")
        return self


class VerdictEvent(BaseModel):
    """A verdict on the answer ``answer_ref`` names, under the run's live judging."""

    model_config = ConfigDict(frozen=True)

    seq: int = Field(ge=1)
    kind: Literal["verdict"] = "verdict"
    question_id: str
    answer_ref: str = Field(min_length=1)
    status: Literal["judged", "unjudged"]
    score: float | None = None
    passed: bool | None = None
    rationale: str | None = None
    error: str | None = None
    probability: float | None = None
    decision: dict[str, Any] | None = None


LiveEvent = Annotated[AnswerEvent | JudgingEvent | VerdictEvent, Field(discriminator="kind")]


class Upload(BaseModel):
    """A contiguous range of a run's events, ``from_seq`` through ``to_seq``."""

    model_config = ConfigDict(frozen=True)

    from_seq: int = Field(ge=1)
    to_seq: int = Field(ge=1)
    events: list[LiveEvent] = Field(min_length=1)

    @model_validator(mode="after")
    def _contiguous(self) -> Upload:
        seqs = [event.seq for event in self.events]
        if seqs != list(range(self.from_seq, self.to_seq + 1)):
            raise ValueError(f"an upload carries events {self.from_seq}..{self.to_seq} in "
                             f"order, each once; got seqs {seqs}")
        return self


class UploadAck(BaseModel):
    """The server holds every event through ``through_seq``; ``state`` is the run's status."""

    through_seq: int = Field(ge=0)
    state: str
    retry_after_s: float | None = Field(default=None, ge=0)


class LiveEndpoint(BaseModel):
    """Where a registered agent run's events go, and how many the server already holds."""

    url: str
    through_seq: int = Field(ge=0)
