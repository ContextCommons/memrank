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
"""What every evaluation is to ``memrank run``: a name, its cases, and how each answer is graded.

A shipped benchmark (:mod:`memrank.definitions.shipped`) and a person's own evaluation file
(:mod:`memrank.definitions.file`) are both a :class:`Definition`, so the service plans, steps
and records them on one code path, and the runner grades them on one code path too.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from memrank.judging.judge import Completer, JudgeConfig
from memrank.service.protocol import Question, QuestionOutcome, Session

#: A grading's outcome: ``{"status": "judged", "score", "passed", "rationale"}``, or
#: ``{"status": "unjudged", "error"}`` when a model judge never gave a readable verdict.
Verdict = dict[str, Any]


@dataclass(frozen=True)
class Case:
    """One unit: the sessions an agent is fed, and the queries it is asked (with references)."""

    id: str
    sessions: tuple[Session, ...]
    queries: tuple[dict[str, Any], ...]

    def question(self, index: int) -> Question:
        query = self.queries[index]
        return Question(id=query["id"], text=query["text"],
                        timestamp=query.get("query_timestamp"))


@dataclass(frozen=True)
class ModelJudge:
    """The model a grader may call: the organisation-keyed completer and its configuration."""

    complete: Completer
    cfg: JudgeConfig


class Grader(ABC):
    """Grades one recorded answer. ``uses_model`` says whether it needs the judge model."""

    uses_model: bool = False

    @abstractmethod
    def grade(self, question: QuestionOutcome, model: ModelJudge | None) -> Verdict:
        """The verdict on ``question.answer``; ``model`` is set whenever ``uses_model``."""


class Definition(ABC):
    """An evaluation, as a run names it, loads its cases and grades its answers.

    Attributes:
        evaluation: The name runs are recorded and compared under (``locomo``,
            ``support-faq@2``).
        version: The evaluation's own version, when it declares one.
        source: What finds it again on a resume: a shipped ref, or the file's absolute path.
        fingerprint: ``sha256:...`` over everything that defines it, so two runs can tell they
            ran the same evaluation.
        dataset_version: The dataset's version, or None where the cases' digest stands in.
        task_version: The version of the task around the data.
    """

    evaluation: str
    version: str | None
    source: str
    fingerprint: str
    dataset_version: str | None
    task_version: int

    @abstractmethod
    def load_cases(self, seed: int) -> list[Case]:
        """Every case with at least one gradeable question, holding only those questions."""

    @abstractmethod
    def grader_for(self, question: QuestionOutcome) -> Grader:
        """The grader for one recorded question."""

    @abstractmethod
    def uses_model(self, seed: int) -> bool:
        """Whether any question ``seed`` selects is graded by the judge model (needs a key)."""
