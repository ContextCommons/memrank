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
from typing import Any, Literal

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
    #: The grader's probability that the answer is correct, when it gives one (a Jev verdict);
    #: None for a text judge, whose verdict is its rationale.
    probability: float | None = None


class CaseOutcome(BaseModel):
    case_id: str
    status: str = Field(description="complete | failed | pending")
    error: str | None = None
    restarts: int = 0
    questions: list[QuestionOutcome]


#: How a record's judge fields were filled: by the client that judged (``run``), or afterwards
#: -- attributed from the router translator's log, or by model alone with no provider claimed.
JudgeSource = Literal["run", "backfill:translator-log", "backfill:unverified"]

#: The provider a record names when nothing recorded which provider judged. A record may say
#: it; a run may not ask for it (``unknown`` is a reserved provider name).
UNKNOWN_PROVIDER = "unknown"


class JudgeIdentity(BaseModel):
    """Who judged: the judge as typed, what it resolved to, and how that is known.

    ``judge`` is ``provider:model`` -- ``anthropic:claude-haiku-4-5``,
    ``router:bedrock/claude-haiku-4-5`` -- or ``unknown:<model>`` on a record whose provider
    was never recorded, where ``judge_provider`` is None. ``judge_endpoint`` is a named
    endpoint's fingerprint (``<id>@<revision>``, ``local:<name>``), never its URL.
    """

    model_config = ConfigDict(frozen=True)

    judge: str = Field(min_length=3)
    judge_provider: str | None
    judge_model: str = Field(min_length=1)
    judge_endpoint: str | None = None
    #: What was sent besides the prompt: ``max_tokens`` (and the field it went under) and any
    #: sampling parameter; None where nothing recorded them.
    judge_settings: dict[str, Any] | None = None
    judge_source: JudgeSource = "run"

    @model_validator(mode="after")
    def _consistent(self) -> JudgeIdentity:
        prefix, sep, model = self.judge.partition(":")
        if not sep or not prefix or model != self.judge_model:
            raise ValueError(f"judge {self.judge!r} is not <provider>:{self.judge_model}")
        if (prefix == UNKNOWN_PROVIDER) != (self.judge_provider is None):
            raise ValueError("an unknown: judge, and only one, has no recorded provider")
        return self

    @classmethod
    def unverified(cls, judge_model: str) -> JudgeIdentity:
        """A judge recorded by model alone: its provider was never recorded, so none is claimed."""
        return cls(judge=f"{UNKNOWN_PROVIDER}:{judge_model}", judge_provider=None,
                   judge_model=judge_model, judge_source="backfill:unverified")

    def label(self) -> str:
        """The judge as every surface shows it, saying how it is known when not by the run."""
        if self.judge_source == "backfill:unverified":
            return f"{self.judge_model} (provider not recorded)"
        if self.judge_source == "backfill:translator-log":
            return f"{self.judge} (attributed from the translator log)"
        return self.judge


#: The judge fields a record carries, flat beside the rest of its identity.
JUDGE_FIELDS = tuple(JudgeIdentity.model_fields)


def judge_fields(record: dict[str, Any]) -> dict[str, Any]:
    """``record`` (an identity or a judging) with an old judge -- a ``judge_model`` and no
    ``judge`` -- read as :meth:`JudgeIdentity.unverified`. Never as ``anthropic:``: nothing
    recorded which provider served that model."""
    if "judge" in record or not record.get("judge_model"):
        return record
    return {**record, **JudgeIdentity.unverified(record["judge_model"]).model_dump()}


def recorded_judge_label(fields: dict[str, Any]) -> str | None:
    """The judge that flat judge fields record -- an identity's, or a receipt ``config``'s -- as
    every surface shows it (:meth:`JudgeIdentity.label`); None when no judge model graded."""
    # A null provider is the record of an unknown one; any other null field is left unstated.
    named = judge_fields({name: fields.get(name) for name in JUDGE_FIELDS
                          if fields.get(name) is not None or name == "judge_provider"})
    if not named.get("judge"):
        return None
    return JudgeIdentity.model_validate(named).label()


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
    #: The :class:`JudgeIdentity` fields, flat; all None when no judge model graded. A record
    #: from before they existed is read by :func:`judge_fields`.
    judge: str | None = None
    judge_provider: str | None = None
    judge_model: str | None
    judge_endpoint: str | None = None
    judge_settings: dict[str, Any] | None = None
    judge_source: JudgeSource | None = None
    judge_samples: int
    #: The judge prompts' version (``JUDGE_PROMPT_VERSION``) when the judge model graded; with
    #: ``judge`` it names this judging (decision 0040). None on runs recorded before it.
    judge_prompt_version: str | None = None
    memrank_version: str
    #: The evaluation's own version, when it declares one (an evaluation file's ``version``).
    evaluation_version: str | None = None
    #: ``sha256:...`` over everything that defines the evaluation -- for a file, the file and
    #: every file its commands name -- so two runs can tell they ran the same one.
    evaluation_fingerprint: str | None = None
    #: What finds the evaluation again: a shipped ref, or the evaluation file's path.
    evaluation_source: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _old_judge(cls, data: Any) -> Any:
        return judge_fields(data) if isinstance(data, dict) else data

    @model_validator(mode="after")
    def _one_judge(self) -> Identity:
        if self.judge is None and self.judge_model is not None:
            raise ValueError(f"judge_model {self.judge_model!r} is recorded with no judge")
        self.judge_identity()  # a judge, when named, is a consistent one
        return self

    def judge_identity(self) -> JudgeIdentity | None:
        """The judge that graded this run, or None when no judge model did."""
        if self.judge is None:
            return None
        return JudgeIdentity.model_validate(
            {name: getattr(self, name) for name in JUDGE_FIELDS})

    def judge_label(self) -> str | None:
        """The judge as every surface shows it (:meth:`JudgeIdentity.label`), or None."""
        judge = self.judge_identity()
        return judge.label() if judge is not None else None


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
