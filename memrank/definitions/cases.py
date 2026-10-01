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
"""An evaluation file's cases, checked one by one and turned into the cases a run asks.

Each raw case -- a YAML list item, a JSONL line, a line a case command printed -- is checked on
its own so that every problem names its case, and then against the rest (ids are unique). A
question's grader is its own ``grade``; else ``rubric`` when it has a rubric and ``choice`` when
it has choices; else the file's ``grade``; else ``judge``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from memrank.definitions.base import Case
from memrank.definitions.graders import choice_index, letter
from memrank.definitions.problems import problems_of
from memrank.definitions.schema import CaseSpec, GraderSpec, QuestionSpec, SessionSpec
from memrank.service.protocol import Session, Turn

#: What each built-in grader needs a question to carry.
_NEEDS = {"exact": ("answer",), "numeric": ("answer",), "judge": ("answer",),
          "choice": ("answer", "choices"), "rubric": ("rubric",)}


@dataclass
class Checked:
    """Cases read so far, and every problem found in them."""

    cases: list[Case] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    case_ids: set[str] = field(default_factory=set)
    question_ids: set[str] = field(default_factory=set)


def grader_for(question: QuestionSpec, default: GraderSpec | None) -> GraderSpec:
    """The question's own ``grade``; else ``rubric`` or ``choice`` when it carries a rubric or
    choices; else the file's ``grade``; else ``judge``."""
    if question.grade is not None:
        return question.grade
    if question.rubric or question.choices:
        return GraderSpec(kind="rubric" if question.rubric else "choice")
    return default or GraderSpec(kind="judge")


def right_choice(answer: str, choices: list[str]) -> int | None:
    """Which choice a reference names: its exact text first, else as an answer would."""
    exact = [i for i, c in enumerate(choices) if c.strip().lower() == answer.strip().lower()]
    return exact[0] if exact else choice_index(answer, choices)


def _choice_problems(question: QuestionSpec) -> list[str]:
    choices, answers = question.choices or [], question.answer or []
    if len({c.strip().lower() for c in choices}) != len(choices):
        return ["`choices`: two choices read the same"]
    if len(answers) != 1:
        return ["`answer`: the choice grader takes exactly one answer, the right choice"]
    if right_choice(answers[0], choices) is None:
        return [f"`answer`: {answers[0]!r} is neither one of the choices nor a letter "
                f"A-{letter(len(choices) - 1)}"]
    return []


def question_problems(question: QuestionSpec, grader: GraderSpec) -> list[str]:
    """What ``question`` lacks for ``grader``, each naming the field."""
    missing = [f"`{name}`: is required by the {grader.name} grader"
               for name in _NEEDS.get(grader.name, ()) if not getattr(question, name)]
    if missing:
        return missing
    if grader.name == "choice":
        return _choice_problems(question)
    if grader.name == "numeric":
        return [f"`answer`: {a!r} is not a number" for a in question.answer or []
                if not _is_number(a)]
    return []


def _is_number(text: str) -> bool:
    try:
        float(text.replace(",", ""))
    except ValueError:
        return False
    return True


def _session(spec: SessionSpec, case_id: str, index: int) -> Session:
    turns = [Turn(role=t.role, speaker=t.speaker or t.role, text=t.text,
                  timestamp=t.timestamp or spec.timestamp) for t in spec.turns]
    return Session(id=spec.id or f"{case_id}.s{index + 1}", timestamp=spec.timestamp,
                   turns=turns)


def _text(question: QuestionSpec) -> str:
    """What the agent is asked: the question, then its choices, lettered, when it has them."""
    if not question.choices:
        return question.question
    lines = [f"{letter(i)}. {choice}" for i, choice in enumerate(question.choices)]
    return question.question + "\n\n" + "\n".join(lines)


def query_of(question: QuestionSpec, qid: str, grader: GraderSpec) -> dict[str, Any]:
    """A question as the query the service keeps: its reference and grader stay server-side."""
    answers = list(question.answer or [])
    if grader.name == "choice" and question.choices:
        right = right_choice(answers[0], question.choices)
        assert right is not None  # question_problems refused the file otherwise
        answers = [question.choices[right]]
    query: dict[str, Any] = {"id": qid, "text": _text(question), "category": question.category,
                             "grader": grader.model_dump(exclude_none=True)}
    optional = {"gold_answers": answers, "rubric": question.rubric, "choices": question.choices,
                "query_timestamp": question.timestamp}
    return {**query, **{k: v for k, v in optional.items() if v}}


def _as_case(raw: Any) -> Any:
    """A question written on its own is a case with that one question and no history."""
    if isinstance(raw, dict) and "questions" not in raw and "question" in raw:
        return {"id": raw.get("id"), "questions": [raw]}
    return raw


def _read_question(raw: Any, spot: str, qid: str, default: GraderSpec | None,
                   checked: Checked) -> dict[str, Any] | None:
    """One question as a query, or None with its problems added to ``checked``."""
    try:
        question = QuestionSpec.model_validate(raw)
    except ValidationError as exc:
        checked.problems += problems_of(exc, spot, raw)
        return None
    grader = grader_for(question, default)
    found = [f"{spot}, {p}" for p in question_problems(question, grader)]
    qid = question.id or qid
    if qid in checked.question_ids:
        found.append(f"{spot}: question id {qid!r} is used twice")
    checked.question_ids.add(qid)
    checked.problems += found
    return None if found else query_of(question, qid, grader)


def read_case(raw: Any, where: str, index: int, default: GraderSpec | None,
              checked: Checked) -> None:
    """Check one raw case, question by question, and add it or its problems to ``checked``."""
    raw = _as_case(raw)
    if isinstance(raw, dict) and raw.get("id") is not None:
        where = f"{where} ({raw['id']})"
    try:
        spec = CaseSpec.model_validate(raw)
    except ValidationError as exc:
        checked.problems += problems_of(exc, where, raw)
        return
    case_id = spec.id or f"case-{index + 1}"
    if case_id in checked.case_ids:
        checked.problems.append(f"{where}: `id` {case_id!r} is used by another case")
    checked.case_ids.add(case_id)
    queries = []
    for number, question in enumerate(spec.questions):
        named = question.get("id") if isinstance(question, dict) else None
        spot = f"{where}, question {number + 1}" + (f" ({named})" if named is not None else "")
        queries.append(_read_question(question, spot, f"{case_id}.q{number + 1}", default,
                                      checked))
    sessions = tuple(_session(s, case_id, i) for i, s in enumerate(spec.history))
    if all(query is not None for query in queries):
        checked.cases.append(Case(id=case_id, sessions=sessions,
                                  queries=tuple(q for q in queries if q is not None)))
