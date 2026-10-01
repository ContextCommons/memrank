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
"""The benchmarks memrank ships, as definitions: LoCoMo, LongMemEval and BEAM.

Loading reuses the benchmark loaders unchanged, so the dataset digests and the question text
they verify are exactly what the in-process harness uses, and grading is each benchmark's own
judge shape -- the scoring these benchmarks always had.
"""

from __future__ import annotations

import hashlib
import json

from memrank.benchmarks import resolve_eval
from memrank.benchmarks.refs import list_eval_refs
from memrank.core import BenchmarkUnit, Document
from memrank.definitions.base import Case, Definition, Grader, ModelJudge, Verdict
from memrank.errors import MemrankError
from memrank.judging.prompts import JUDGE_PROMPT_VERSION
from memrank.judging.shape import JudgeShape
from memrank.service.grading import verdict_for
from memrank.service.protocol import EVALUATIONS, QuestionOutcome, Session, Turn


class UnknownEvaluation(MemrankError):
    """A name that is neither a shipped evaluation nor an evaluation file."""


def _turn_text(message: dict, speaker: str) -> str:
    """The utterance without the ``"Speaker: "`` prefix loaders put into ``content``."""
    content = str(message.get("content", ""))
    prefix = f"{speaker}: "
    return content[len(prefix):] if message.get("speaker") and content.startswith(prefix) else content


def session_of(doc: Document) -> Session:
    """One benchmark document as a session of attributed, timestamped turns."""
    messages = doc.messages or [{"role": "user", "content": doc.content}]
    turns = []
    for message in messages:
        speaker = str(message.get("speaker") or message.get("role") or "user")
        turns.append(Turn(role=str(message.get("role") or "user"), speaker=speaker,
                          text=_turn_text(message, speaker), timestamp=doc.timestamp))
    return Session(id=doc.id, timestamp=doc.timestamp, turns=turns)


def case_of(unit: BenchmarkUnit, shape: JudgeShape) -> Case:
    """A unit as a case holding only the questions its judge shape can grade."""
    return Case(id=unit.unit_id, sessions=tuple(session_of(d) for d in unit.documents),
                queries=tuple(q for q in unit.queries if shape.is_judgeable(q)))


def shipped_refs() -> list[str]:
    """Every shipped evaluation ``memrank run`` takes, canonically spelled."""
    return [ref for ref in list_eval_refs() if ref.split(":")[0] in EVALUATIONS]


class ShapeGrader(Grader):
    """A benchmark's own judge shape, grading with the judge model."""

    uses_model = True

    def __init__(self, shape: JudgeShape) -> None:
        self.shape = shape

    def grade(self, question: QuestionOutcome, model: ModelJudge | None) -> Verdict:
        assert model is not None  # uses_model: the runner always hands one over
        return verdict_for(self.shape, model.complete, model.cfg, question)


class BenchmarkDefinition(Definition):
    """A shipped benchmark variant (``locomo``, ``beam:100k``) as an evaluation."""

    task_version: int

    def __init__(self, ref: str) -> None:
        try:
            bench, canonical = resolve_eval(ref)
        except MemrankError as exc:
            raise UnknownEvaluation(
                f"{exc} -- choose one of {', '.join(EVALUATIONS)}, or pass an evaluation file "
                "(`memrank evals new my-eval.yaml` writes a starter)") from exc
        if bench.name not in EVALUATIONS:
            raise UnknownEvaluation(f"evaluation {bench.name!r} cannot be put to an agent; "
                                    f"choose one of {', '.join(EVALUATIONS)}")
        self.bench, self.shape = bench, bench.judge_shape()
        self.evaluation = self.source = canonical
        self.version = None
        self.dataset_version = bench.dataset_version
        self.task_version = bench.VERSION
        self._grader = ShapeGrader(self.shape)
        identity = {"ref": canonical, "dataset_version": bench.dataset_version,
                    "task_version": bench.VERSION, "judge_prompts": JUDGE_PROMPT_VERSION}
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        self.fingerprint = f"sha256:{digest}"

    def load_cases(self, seed: int) -> list[Case]:
        units = [case_of(u, self.shape) for u in self.bench.load()]
        return [case for case in units if case.queries]

    def grader_for(self, question: QuestionOutcome) -> Grader:
        return self._grader

    def uses_model(self, seed: int) -> bool:
        return True
