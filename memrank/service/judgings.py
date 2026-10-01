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
answer's verdict, named by its judge model and judge prompt version. The runs API stores the
verdicts as an artifact of the run and lists the scores in the run's record
(:mod:`memrank.api.agent_judgings`); nothing here depends on the API.
"""

from __future__ import annotations

import hashlib
import json

from pydantic import BaseModel, Field

from memrank.service.protocol import Interval, RunResult

#: The alphabet of a judge model and prompt version: both become part of an artifact's name.
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


class Judging(BaseModel):
    """The scores and every verdict one judge gave a run's recorded answers."""

    judge_model: str = Field(pattern=_NAME)
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

    def key(self) -> tuple[str, str]:
        """What names this judging among a run's others."""
        return self.judge_model, self.judge_prompt_version

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
    if not judged.judged or not identity.judge_model or not identity.judge_prompt_version:
        raise ValueError("a judging is named by the judge model and prompt version that graded "
                         "it; this result was not graded by a judge model")
    verdicts = [QuestionVerdict(case_id=q.case_id, question_id=q.question_id, status=q.status,
                                score=q.score, passed=q.passed, rationale=q.rationale,
                                error=q.error)
                for case in judged.cases for q in case.questions]
    return Judging(judge_model=identity.judge_model,
                   judge_prompt_version=identity.judge_prompt_version,
                   judge_samples=identity.judge_samples, notice=judged.notice,
                   score=judged.score, per_category=judged.per_category,
                   failure_rate=judged.failure_rate, questions=judged.questions,
                   failed=judged.failed, unjudged=judged.unjudged, verdicts=verdicts)
