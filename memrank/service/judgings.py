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
"""One judging of a finished run's recorded answers, as it is stored beside the run.

Decision 0040: judging is a step taken on a finished run, and every judging is kept side by
side. A run's own judging is its ``result`` (:class:`~memrank.service.protocol.RunResult`); each
judging added afterwards is a :class:`Judging` -- the scores over the same answers and every
answer's verdict, named by its judge (as typed, with its endpoint) and judge prompt version. The runs API stores the
verdicts as an artifact of the run and lists the scores in the run's record
(:mod:`memrank.api.agent_judgings`); nothing here depends on the API.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel, Field, model_validator

from memrank.service.protocol import JUDGE_FIELDS, Interval, JudgeIdentity, JudgeSource, RunResult

#: The alphabet of a judge prompt version, which becomes part of an artifact's name. The judge
#: does not: it may hold ``:`` and ``/``, so the artifact is named by its hash.
_NAME = r"^[A-Za-z0-9._-]+$"


class QuestionVerdict(BaseModel):
    """One recorded answer's grade under this judging."""

    case_id: str
    question_id: str
    status: str = Field(description="judged | failed | unjudged")
    score: float | None = None
    passed: bool | None = None
    rationale: str | None = None
    error: str | None = None
    probability: float | None = None


class Judging(BaseModel):
    """The scores and every verdict one judge gave a run's recorded answers."""

    #: The :class:`~memrank.service.protocol.JudgeIdentity` fields, flat as in a run's identity.
    judge: str
    judge_provider: str | None
    judge_model: str
    judge_endpoint: str | None = None
    judge_settings: dict[str, Any] | None = None
    judge_source: JudgeSource = "run"
    judge_prompt_version: str = Field(pattern=_NAME)
    judge_samples: int = Field(ge=1)
    notice: str | None = None
    score: Interval
    per_category: dict[str, Interval]
    failure_rate: float
    questions: int
    failed: int
    unjudged: int
    verdicts: list[QuestionVerdict]

    @model_validator(mode="after")
    def _a_judge(self) -> Judging:
        self.identity()  # the same consistency a run's judge is held to
        return self

    def identity(self) -> JudgeIdentity:
        """The judge that gave this judging."""
        return JudgeIdentity.model_validate({name: getattr(self, name) for name in JUDGE_FIELDS})

    def key(self) -> dict[str, str | None]:
        """What names this judging among a run's others: the judge as typed, the endpoint it
        reached, and the prompts' version."""
        return {"judge": self.judge, "judge_endpoint": self.judge_endpoint,
                "judge_prompt_version": self.judge_prompt_version}

    def judge_hash(self) -> str:
        """A file-name-safe name for the judge and its endpoint."""
        named = f"{self.judge}|{self.judge_endpoint or ''}"
        return hashlib.sha256(named.encode("utf-8")).hexdigest()[:16]

    def digest(self) -> str:
        """``sha256`` over the canonical JSON, so the same judging sent twice is recognised."""
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def summary(self) -> dict:
        """What the run's record lists: everything but the verdicts."""
        return self.model_dump(mode="json", exclude={"verdicts"})


def judging_of(judged: RunResult) -> Judging:
    """``judged`` -- a run's recorded answers graded again -- as a judging to add to that run.

    Raises:
        ValueError: when no judge model graded it, so nothing names the judging.
    """
    identity = judged.identity
    judge = identity.judge_identity()
    if not judged.judged or judge is None or not identity.judge_prompt_version:
        raise ValueError("a judging is named by the judge and prompt version that graded it; "
                         "this result was not graded by a judge model")
    verdicts = [QuestionVerdict(case_id=q.case_id, question_id=q.question_id, status=q.status,
                                score=q.score, passed=q.passed, rationale=q.rationale,
                                error=q.error, probability=q.probability)
                for case in judged.cases for q in case.questions]
    return Judging(**judge.model_dump(),
                   judge_prompt_version=identity.judge_prompt_version,
                   judge_samples=identity.judge_samples, notice=judged.notice,
                   score=judged.score, per_category=judged.per_category,
                   failure_rate=judged.failure_rate, questions=judged.questions,
                   failed=judged.failed, unjudged=judged.unjudged, verdicts=verdicts)
