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
"""The graders an evaluation file can name, and running a grader that is your own program.

- ``exact``: the answer equals a reference once both are normalised -- lowercased, punctuation
  and the articles a/an/the removed, whitespace collapsed.
- ``choice``: the answer picks the right one of the question's ``choices``, by its letter
  (``B``, ``(B)``, ``B.``) or by its text.
- ``numeric``: the last number in the answer is within ``tolerance`` of a reference.
- ``judge``: the judge model decides whether the answer matches the reference.
- ``rubric``: the judge model scores each rubric criterion 0, 0.5 or 1; the score is the mean.
- a command: your program reads ``{question, answer, reference, ...}`` as JSON on stdin and
  prints ``{"score": <0..1>, "why": "..."}``.

The built-in ones score 1 or 0 (``rubric`` and a command may give partial credit); ``passed``
means a full score.
"""

from __future__ import annotations

import json
import re
import string
from pathlib import Path
from typing import Any

from memrank.definitions import commands
from memrank.definitions.base import Grader, ModelJudge, Verdict
from memrank.definitions.schema import DEFAULT_TIMEOUT_S, GraderSpec
from memrank.judging.judge import UnparseableVerdict, judge_answer
from memrank.judging.shape import NuggetJudgeShape
from memrank.outcome import Step
from memrank.service.grading import verdict_for
from memrank.service.protocol import QuestionOutcome

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCTUATION = str.maketrans("", "", string.punctuation)
_LETTER = re.compile(r"^\s*\(?([A-Za-z])\)?(?:[.):]|\s*$)")
_NUMBER = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?|[-+]?\.\d+")


def normalise(text: str) -> str:
    """Lowercase, no punctuation, no articles, single spaces."""
    lowered = _ARTICLES.sub(" ", text.lower().translate(_PUNCTUATION))
    return " ".join(lowered.split())


def letter(index: int) -> str:
    return string.ascii_uppercase[index]


def choice_index(text: str, choices: list[str]) -> int | None:
    """Which of ``choices`` ``text`` picks: by letter, by exact text, or by naming only one."""
    match = _LETTER.match(text)
    if match and string.ascii_uppercase.find(match.group(1).upper()) < len(choices):
        return string.ascii_uppercase.index(match.group(1).upper())
    said = normalise(text)
    normalised = [normalise(choice) for choice in choices]
    if said in normalised:
        return normalised.index(said)
    named = [i for i, choice in enumerate(normalised) if choice and choice in said]
    return named[0] if len(named) == 1 else None


def parse_number(text: str) -> float | None:
    """The last number written in ``text``, thousands separators allowed."""
    found = _NUMBER.findall(text)
    return float(found[-1].replace(",", "")) if found else None


def _verdict(score: float, why: str) -> Verdict:
    return {"status": "judged", "score": score, "passed": score == 1.0, "rationale": why}


def _references(question: QuestionOutcome) -> list[str]:
    return [str(r) for r in question.grading.get("gold_answers") or []]


class Exact(Grader):
    def grade(self, question: QuestionOutcome, model: ModelJudge | None) -> Verdict:
        said = normalise(question.answer or "")
        hit = next((r for r in _references(question) if normalise(r) == said), None)
        return _verdict(1.0 if hit is not None else 0.0,
                        f"matches {hit!r}" if hit is not None else
                        "matches no reference after normalising")


class Choice(Grader):
    def grade(self, question: QuestionOutcome, model: ModelJudge | None) -> Verdict:
        choices = list(question.grading.get("choices") or [])
        right = choices.index(_references(question)[0])
        picked = choice_index(question.answer or "", choices)
        if picked is None:
            return _verdict(0.0, "picks none of the choices")
        return _verdict(1.0 if picked == right else 0.0,
                        f"picks {letter(picked)}; the answer is {letter(right)}")


class Numeric(Grader):
    def __init__(self, tolerance: float) -> None:
        self.tolerance = tolerance

    def grade(self, question: QuestionOutcome, model: ModelJudge | None) -> Verdict:
        value = parse_number(question.answer or "")
        if value is None:
            return _verdict(0.0, "has no number")
        references = [float(r) for r in _references(question)]
        hit = any(abs(value - r) <= self.tolerance for r in references)
        return _verdict(1.0 if hit else 0.0, f"reads {value:g}; reference "
                        f"{' or '.join(f'{r:g}' for r in references)} +/- {self.tolerance:g}")


class Judge(Grader):
    """The judge model against the reference answer(s)."""

    uses_model = True

    def grade(self, question: QuestionOutcome, model: ModelJudge | None) -> Verdict:
        assert model is not None  # uses_model: the runner always hands one over
        references = _references(question)
        gold = references[0] if len(references) == 1 else "any of: " + " | ".join(references)
        try:
            verdict = judge_answer(model.complete, question=question.question,
                                   answer=question.answer or "", gold=gold,
                                   model=model.cfg.judge_model, samples=model.cfg.samples)
        except UnparseableVerdict as exc:
            return {"status": "unjudged", "error": str(exc)}
        return _verdict(1.0 if verdict.passed else 0.0, verdict.rationale)


class Rubric(Grader):
    """BEAM's rubric grading: every criterion scored by the judge model, then averaged."""

    uses_model = True

    def grade(self, question: QuestionOutcome, model: ModelJudge | None) -> Verdict:
        assert model is not None  # uses_model: the runner always hands one over
        return verdict_for(NuggetJudgeShape(), model.complete, model.cfg, question)


class GraderFailed(commands.CommandFailed):
    """A command grader failed or printed something other than a score."""


class Program(Grader):
    """Your grader: JSON in on stdin, ``{"score", "why"}`` out on stdout."""

    def __init__(self, command: list[str], folder: Path, timeout_s: float) -> None:
        self.command, self.folder, self.timeout_s = command, folder, timeout_s

    def grade(self, question: QuestionOutcome, model: ModelJudge | None) -> Verdict:
        references = _references(question)
        given = {"case_id": question.case_id, "question_id": question.question_id,
                 "question": question.question, "answer": question.answer or "",
                 "reference": references[0] if references else None,
                 "references": references, "category": question.category,
                 **{k: question.grading[k] for k in ("choices", "rubric")
                    if k in question.grading}}
        out = commands.run(self.command, self.folder, stdin=json.dumps(given),
                           timeout_s=self.timeout_s, what="The grader command")
        return self._read(out, question)

    def _read(self, out: str, question: QuestionOutcome) -> Verdict:
        try:
            printed: Any = json.loads(out)
            score = float(printed["score"])
            why = str(printed.get("why") or "")
        except (ValueError, KeyError, TypeError) as exc:
            raise self._refused(question, f"it printed {out.strip()[:200]!r}") from exc
        if not 0.0 <= score <= 1.0:
            raise self._refused(question, f"its score {score:g} is not between 0 and 1")
        return _verdict(score, why)

    def _refused(self, question: QuestionOutcome, what: str) -> GraderFailed:
        return GraderFailed(
            f"The grader command {' '.join(self.command)} did not grade question "
            f"{question.question_id}: {what}.",
            steps=(Step('It must print one JSON object: {"score": <0 to 1>, "why": "..."}.'),))


def grader_of(spec: GraderSpec, folder: Path) -> Grader:
    """The grader ``spec`` names; a command runs from the evaluation file's ``folder``."""
    if spec.command is not None:
        return Program(spec.command, folder, spec.timeout_s or DEFAULT_TIMEOUT_S)
    built_in: dict[str, Grader] = {"exact": Exact(), "choice": Choice(), "judge": Judge(),
                                   "rubric": Rubric(), "numeric": Numeric(spec.tolerance or 0.0)}
    return built_in[spec.name]
