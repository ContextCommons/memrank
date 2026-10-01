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
"""Judging one recorded answer with its benchmark's own judge shape.

``memrank run`` calls this for each answer of a finished run (:mod:`memrank.loop.judge`). Everything the shape reads comes
from the question itself (:attr:`~memrank.service.protocol.QuestionOutcome.grading`), so the
dataset is never loaded to judge.
"""

from __future__ import annotations

from typing import Any

from memrank.judging.judge import Completer, JudgeConfig, UnparseableVerdict
from memrank.judging.shape import JudgeShape
from memrank.service.protocol import QuestionOutcome


def query_of(question: QuestionOutcome) -> dict[str, Any]:
    """The query a judge shape expects, rebuilt from a recorded question."""
    return {"id": question.question_id, "text": question.question,
            "category": question.category, **question.grading}


def verdict_for(shape: JudgeShape, complete: Completer, cfg: JudgeConfig,
                question: QuestionOutcome) -> dict[str, Any]:
    """One answer's verdict. A reply the judge never makes parseable is recorded as unjudged."""
    try:
        grade = shape.grade_answer(complete, cfg, query=query_of(question),
                                   answer=question.answer or "")
    except UnparseableVerdict as exc:
        return {"status": "unjudged", "error": str(exc)}
    return {"status": "judged", "score": grade.score, "passed": grade.correctness.passed,
            "rationale": grade.correctness.rationale}
