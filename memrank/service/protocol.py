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
"""The wire shapes between the evaluation service and a runner.

Plain pydantic models with no web framework behind them, so the runner imports this module
without installing the service. Six calls make up the whole protocol:

- ``POST /v1/runs`` with :class:`RunCreate` -> :class:`RunCreated`
- ``POST /v1/runs/{id}/next?lane=L&lanes=N`` -> :class:`Step` (lane 0 of 1 when omitted)
- ``POST /v1/runs/{id}/steps/{step_id}`` with :class:`StepResult` -> :class:`Ack`
- ``POST /v1/runs/{id}/retry-failed`` -> :class:`Reopened` (only when asked: ``--retry-failed``)
- ``GET /v1/runs/{id}/status`` -> :class:`RunStatus`
- ``GET /v1/runs/{id}/result`` -> :class:`RunResult`
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

#: The shipped evaluations the service can put to an agent: the answer-judged ones. Anything
#: else it runs is an evaluation file (:mod:`memrank.definitions`).
EVALUATIONS = ("locomo", "longmemeval", "beam")


class Op(str, Enum):
    """What a step asks the runner to do."""

    RESET = "reset"
    FEED = "feed"
    ASK = "ask"
    DONE = "done"


class AgentRef(BaseModel):
    """How the run names the agent under test. Recorded, never interpreted."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1)
    version: str | None = None
    #: The agent file the run was started from, and the sha256 of its bytes, so a result says
    #: exactly which spec produced it and a resume can refuse a spec that changed.
    spec_path: str | None = None
    spec_sha256: str | None = None


class RunCreate(BaseModel):
    """A request to evaluate one agent on one evaluation."""

    model_config = ConfigDict(frozen=True)

    evaluation: str = Field(min_length=1, description="An eval ref, e.g. locomo or beam:100k, "
                                                      "or the path to an evaluation file")
    agent: AgentRef
    cases: int | None = Field(default=None, ge=1, description="Sample this many cases")
    questions: int | None = Field(default=None, ge=1,
                                  description="Sample at most this many questions per case")
    seed: int = 0
    judge: bool = Field(default=True, description="False records answers and never scores them")


class RunCreated(BaseModel):
    run_id: str
    cases: int
    questions: int


class Turn(BaseModel):
    """One utterance in a past conversation."""

    role: str
    speaker: str
    text: str
    timestamp: str | None = None


class Session(BaseModel):
    """One past conversation, in order."""

    id: str
    timestamp: str | None = None
    turns: list[Turn]


class Question(BaseModel):
    """What the agent is asked. Never the reference answer or the category."""

    id: str
    text: str
    timestamp: str | None = None


class Progress(BaseModel):
    """Where this step sits in the run, for the runner's progress line."""

    case: int
    cases: int
    question: int
    questions: int


class Step(BaseModel):
    """The next thing the runner must do.

    ``session_id`` is unique per run and per attempt at a case, so an agent that scopes memory
    by it starts every attempt clean even when it has no reset endpoint.
    """

    step_id: str | None = None
    op: Op
    case_id: str | None = None
    session_id: str | None = None
    sessions: list[Session] | None = None
    question: Question | None = None
    progress: Progress | None = None


class StepResult(BaseModel):
    """What happened when the runner carried out a step: exactly one of the three outcomes."""

    model_config = ConfigDict(frozen=True)

    ok: bool | None = None
    answer: str | None = None
    error: str | None = None
    elapsed_ms: float = Field(ge=0)

    @model_validator(mode="after")
    def _exactly_one(self) -> StepResult:
        given = [name for name in ("ok", "answer", "error") if getattr(self, name) is not None]
        if len(given) != 1:
            raise ValueError(f"a step result carries exactly one of ok, answer, error; got {given}")
        if self.ok is False:
            raise ValueError("report a failed step as error, not ok=false")
        return self


class Ack(BaseModel):
    """The service accepted a step result (or had already accepted this exact one)."""

    step_id: str
    duplicate: bool = False


class Reopened(BaseModel):
    """The cases a run was told to try again -- failed, or complete with failed answers -- each
    restarting from a reset."""

    run_id: str
    cases: list[str]


class RunStatus(BaseModel):
    """How far a run has got, for a progress bar: counts and summed step time, nothing more.

    A failed case counts as fed and its unasked questions as done: nothing more will happen to
    them, so a bar that waited for them would never reach its end.
    """

    run_id: str
    #: The canonical eval ref, and whether the run is to be judged -- what a resumed runner
    #: needs to register the run without being told its original arguments again.
    evaluation: str
    judge: bool
    cases: int
    questions: int
    cases_finished: int
    cases_fed: int
    fed_ms: float
    questions_done: int
    asked_ms: float
    done: bool


class Interval(BaseModel):
    """A mean with its 95% bootstrap interval, resampling whole cases."""

    mean: float | None
    ci95: tuple[float | None, float | None]
    n: int


class QuestionOutcome(BaseModel):
    case_id: str
    question_id: str
    question: str
    #: What the judge grades against: the gold answer(s), or BEAM's rubric criteria.
    reference: list[str]
    category: str | None
    #: What the benchmark's judge shape reads besides the text and category (gold answers or
    #: rubric, kind, judge prompt key), so the result alone is enough to judge the answer.
    grading: dict[str, Any] = Field(default_factory=dict)
    status: str = Field(description="judged | not_judged | failed | unjudged | pending")
    answer: str | None = None
    error: str | None = None
    elapsed_ms: float | None = None
    score: float | None = None
    passed: bool | None = None
    rationale: str | None = None


class CaseOutcome(BaseModel):
    case_id: str
    status: str = Field(description="complete | failed | pending")
    error: str | None = None
    restarts: int = 0
    questions: list[QuestionOutcome]


class Identity(BaseModel):
    """What was run: enough to rerun it, and to tell two results apart."""

    #: The evaluation's name, as runs are compared under (``locomo``, ``support-faq@2``).
    evaluation: str
    dataset_version: str
    task_version: int
    seed: int
    cases_requested: int | None
    questions_per_case: int | None
    case_ids: list[str]
    agent: AgentRef
    judge_requested: bool
    judge_model: str | None
    judge_samples: int
    #: The judge prompts' version (``JUDGE_PROMPT_VERSION``) when the judge model graded; with
    #: ``judge_model`` it names this judging (decision 0040). None on runs recorded before it.
    judge_prompt_version: str | None = None
    memrank_version: str
    #: The evaluation's own version, when it declares one (an evaluation file's ``version``).
    evaluation_version: str | None = None
    #: ``sha256:...`` over everything that defines the evaluation -- for a file, the file and
    #: every file its commands name -- so two runs can tell they ran the same one.
    evaluation_fingerprint: str | None = None
    #: What finds the evaluation again: a shipped ref, or the evaluation file's path.
    evaluation_source: str | None = None


class RunResult(BaseModel):
    """Scores, their variability, and every case's outcome."""

    run_id: str
    evaluation: str
    agent: AgentRef
    identity: Identity
    status: str = Field(description="complete")
    judged: bool
    #: Set whenever the scores below are not a quality measurement, saying why.
    notice: str | None = None
    score: Interval
    per_category: dict[str, Interval]
    failure_rate: float
    questions: int
    failed: int
    unjudged: int
    latency_ms: dict[str, float | None]
    cases: list[CaseOutcome]


#: The artifact an uploaded run's every question is stored in, beside the summary record that
#: points at it (ATO-2343): the record is bounded, and a run of any size is not.
ANSWERS_FILE = "answers.json"


def split_answers(result: RunResult) -> tuple[RunResult, list[CaseOutcome]]:
    """``result`` as its summary -- every case kept, none carrying its questions -- and the
    cases with their questions, which travel as :data:`ANSWERS_FILE`."""
    summary = result.model_copy(update={
        "cases": [case.model_copy(update={"questions": []}) for case in result.cases]})
    return summary, result.cases
