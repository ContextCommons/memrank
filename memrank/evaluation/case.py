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
"""One case's row, typed -- the READ form of a ``per_query`` entry.

The artifact's ``per_query`` is a ``list[dict[str, Any]]`` whose keys are learned from
``aggregate._drill_unit``'s source, and four of them -- ``generated_answer``,
``sufficiency``, ``correctness``, ``answered_without_context`` -- are written by
``judge_stage._judge_one_query`` and by nothing else, so they are ABSENT entirely on an
unjudged run, which is the shape most people run first. Reading one failing case therefore
meant opening two modules to find out which keys could be there.

:class:`CaseRow` is a view over that same dict. It is DERIVED, never stored: the dict stays
exactly as it was, ``EvalResult.to_dict()`` is untouched, and the two publication allowlists
(``analysis.compare._whitelist_query``, ``leaderboard.safe``) keep selecting the same keys
from the same dict. Nothing here can put a key into a bundle, because nothing here writes a
key anywhere.

Two rules the read form holds that the dict does not:

- **Nothing a reader needs is absent.** An unjudged run produces no answer, so
  :attr:`CaseRow.answered` is an :class:`Answer` that SAYS no answer was produced and why --
  :data:`NOT_JUDGED`. An absent field a reader could take for a failure is exactly what the
  target's T13 forbids.
- **A marking is a named subscore with a decider, not a bare number.** Each entry in
  :attr:`CaseRow.subscores` carries who decided it (:data:`DECIDED_BY_MEMRANK` or
  :data:`DECIDED_BY_JUDGE`), why, and what was missing. A judged and an unjudged run differ
  in WHICH subscores are present -- never in whether the reader can tell what decided one.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

#: memrank's own deterministic marking decided this -- the substring proxy, no model involved.
DECIDED_BY_MEMRANK = "memrank"
#: An LLM judge decided this. What the judge is worth is a separate question the result does
#: not yet answer (the plan leaves marker evidence open); this says only who decided.
DECIDED_BY_JUDGE = "judge"

#: Why an unjudged run carries no answer. Stated as a value rather than left as an absence:
#: no answer was produced BECAUSE this run shape produces none, which is a different thing
#: from an engine that was asked and returned nothing.
NOT_JUDGED = ("no answer was produced: this run was not judged, and only a judged run "
              "produces an answer from what was recalled")

#: Subscore key for memrank's deterministic marking: was the evidence this case required
#: present in what the engine returned? Named for what it decides on ONE case -- the rate
#: over many cases is the composite, which the result names ``quality_metric``.
EVIDENCE_FOUND = "evidence_found"
#: Subscore key for the judge's verdict on the answer against the dataset's reference.
CORRECTNESS = "correctness"
#: Subscore key for the judge's verdict on whether what was recalled could support an answer
#: at all. Distinguishes "the engine recalled nothing useful" from "the answer was wrong".
SUFFICIENCY = "sufficiency"


@dataclass(frozen=True)
class Passage:
    """One piece of material the engine returned, as it returned it."""

    text: str
    id: str | None = None
    #: Position in the engine's own ordering, 0-based. Result order is the engine's ranking.
    rank: int | None = None
    #: The engine's own relevance score where it reported one. ``None`` means it reported
    #: none, never zero.
    score: float | None = None


@dataclass(frozen=True)
class Answer:
    """What the run produced as an answer for one case, or that it produced none, and why.

    Never ``None``: a reader asking "what answer was produced" gets an object that answers
    either way. :attr:`was_produced` is the discriminator.
    """

    text: str | None = None
    produced_by: str | None = None
    #: Set exactly when no answer exists. :data:`NOT_JUDGED` on an unjudged run.
    not_produced_because: str | None = None

    @property
    def was_produced(self) -> bool:
        return self.not_produced_because is None

    @classmethod
    def none_produced(cls, because: str) -> Answer:
        return cls(not_produced_because=because)


@dataclass(frozen=True)
class Subscore:
    """One named criterion's verdict on one case, with who decided it and why.

    A mapping of these rather than a float, because a float cannot say who decided it, and
    a case that scored zero under a proxy and a case a judge marked wrong are not the same
    finding.
    """

    value: float | bool
    decided_by: str
    why: str | None = None
    #: What the case required and did not get. Empty when nothing was missing, or when the
    #: decider does not report at that granularity.
    missing: tuple[str, ...] = ()


@dataclass(frozen=True)
class CaseRow:
    """One case, complete: asked, recalled, answered, marked -- judged run or not."""

    case_id: str
    asked: str
    subscores: Mapping[str, Subscore]
    answered: Answer
    #: The unit this case was asked against. The haystack, in the benchmark's own vocabulary.
    unit_id: str | None = None
    category: str | None = None
    recalled: tuple[Passage, ...] = ()
    #: Tokens of context the recalled material amounted to, and whether the budget cut it.
    context_tokens: int | None = None
    context_truncated: bool | None = None
    #: Judged runs only: the judge answered from no context at all and still got it right,
    #: which makes the case say nothing about memory. ``None`` when no judge ran.
    answered_without_context: bool | None = None
    #: One entry per rubric criterion, keyed by the criterion's own text, for the judge
    #: shapes that grade against a rubric. Empty otherwise.
    rubric: Mapping[str, Subscore] = field(default_factory=dict)

    @classmethod
    def from_artifact(cls, row: Mapping[str, Any]) -> CaseRow:
        """Derive the read form from one ``per_query`` dict. Reads it; never writes it."""
        return cls(
            case_id=row["query_id"],
            asked=row.get("text", ""),
            subscores=_subscores(row),
            answered=_answer(row),
            unit_id=row.get("unit_id"),
            category=row.get("category"),
            recalled=_passages(row.get("retrieved") or ()),
            context_tokens=row.get("context_tokens_sent", row.get("context_tokens")),
            context_truncated=row.get("context_truncated"),
            answered_without_context=row.get("answered_without_context"),
            rubric=_rubric(row.get("nuggets") or ()),
        )


def rows_of(per_query: Sequence[Mapping[str, Any]]) -> tuple[CaseRow, ...]:
    """Every case in a cell, in the order the cell measured them."""
    return tuple(CaseRow.from_artifact(row) for row in per_query)


def _passages(retrieved: Sequence[Mapping[str, Any]]) -> tuple[Passage, ...]:
    return tuple(
        Passage(text=doc.get("content", ""), id=doc.get("id"), rank=rank,
                score=doc.get("score"))
        for rank, doc in enumerate(retrieved))


def _answer(row: Mapping[str, Any]) -> Answer:
    """The judge's answer, or the statement that this run shape produced none."""
    if "generated_answer" not in row:
        return Answer.none_produced(NOT_JUDGED)
    return Answer(text=row["generated_answer"], produced_by=DECIDED_BY_JUDGE)


def _subscores(row: Mapping[str, Any]) -> dict[str, Subscore]:
    """memrank's deterministic verdict always; the judge's two when a judge ran."""
    scores = {EVIDENCE_FOUND: _evidence_subscore(row)}
    correctness = row.get("correctness")
    if correctness is not None:
        scores[CORRECTNESS] = _verdict_subscore(correctness)
    sufficiency = row.get("sufficiency")
    if sufficiency is not None:
        scores[SUFFICIENCY] = _verdict_subscore(sufficiency)
    return scores


def _evidence_subscore(row: Mapping[str, Any]) -> Subscore:
    """``hit`` plus the span that matched, or the spans that did not."""
    hit = bool(row.get("hit", False))
    missing = tuple(row.get("missed_required_spans") or ())
    if hit:
        why = _matched_why(row)
    elif missing:
        why = f"none of the required spans appeared in what the engine returned: {list(missing)}"
    else:
        why = "the required evidence did not appear in what the engine returned"
    return Subscore(value=hit, decided_by=DECIDED_BY_MEMRANK, why=why, missing=missing)


def _matched_why(row: Mapping[str, Any]) -> str:
    span = row.get("matched_span")
    doc_id = row.get("matched_doc_id")
    where = f" in document {doc_id!r}" if doc_id is not None else ""
    return f"matched {span!r}{where}" if span is not None else f"matched{where}"


def _verdict_subscore(verdict: Mapping[str, Any]) -> Subscore:
    """A judge verdict dict (``{passed, rationale, samples}``) as a subscore."""
    return Subscore(value=bool(verdict.get("passed", False)),
                    decided_by=DECIDED_BY_JUDGE,
                    why=verdict.get("rationale"))


def _rubric(nuggets: Sequence[Mapping[str, Any]]) -> dict[str, Subscore]:
    return {n["nugget"]: Subscore(value=n["score"], decided_by=DECIDED_BY_JUDGE,
                                  why=n.get("rationale"))
            for n in nuggets}

