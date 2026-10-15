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
"""Grading an answer with Jev: one request of typed questions, decided in code.

Jev writes no text, so it is not a :data:`~memrank.judging.judge.Completer`: it grades the shape a
benchmark declares (:mod:`memrank.judging.shape`) with its own questions
(:mod:`~memrank.judging.jev.questions`) and rules (:mod:`~memrank.judging.jev.rules`). It gives no
rationale either, so every verdict carries a ``decision`` record in its place: the exact state, the
question set's version and a digest of the questions, every answer with its probabilities, the
rule and threshold that decided, and the model asked for beside the model that answered.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

from memrank.judging.jev import questions as jq
from memrank.judging.jev import rules
from memrank.judging.jev.client import Decision, JevClient
from memrank.judging.judge import UnparseableVerdict, gold_answer
from memrank.judging.prompts import LME_PROMPTS
from memrank.judging.shape import (
    EVENT_ORDERING,
    BeamJudgeShape,
    BinaryJudgeShape,
    JudgeShape,
    NuggetJudgeShape,
    is_negative,
)

#: What a verdict's rationale says in place of an explanation Jev does not give.
NO_RATIONALE = "Jev gives no rationale; the decision record holds its probabilities."

#: How an extraction-free ordering score is flagged wherever it is recorded.
EXPERIMENTAL_ORDERING = (
    "experimental: event_ordering under Jev is scored from per-event and per-pair judgements "
    "without extracting the answer's events; it is not the same measurement as the "
    "extraction-based score")


def _digest(questions: dict[str, Any]) -> str:
    canonical = json.dumps(questions, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()


def _verdict(score: float, passed: bool, rationale: str, record: dict[str, Any],
             probability: float | None = None) -> dict[str, Any]:
    return {"status": "judged", "score": score, "passed": passed, "rationale": rationale,
            "probability": probability, "decision": record}


@dataclass(frozen=True)
class JevJudge:
    """Grades with one System One model at a fixed threshold."""

    client: JevClient
    threshold: float = rules.THRESHOLD

    def _ask(self, state: Any, questions: dict[str, Any]) -> tuple[Decision, dict[str, Any]]:
        decision = self.client.decide(state, questions, question_set=jq.QUESTION_SET)
        record = {"rationale": "none: " + NO_RATIONALE, "question_set": jq.QUESTION_SET,
                  "questions_sha256": _digest(questions), "threshold": self.threshold,
                  "model_requested": self.client.target.model, "model": decision.model,
                  "request_id": decision.request_id, "usage": decision.usage,
                  "state": state, "answers": decision.answers}
        return decision, record

    def verdict(self, shape: JudgeShape, query: dict[str, Any], answer: str) -> dict[str, Any]:
        """The verdict on ``answer`` under ``shape``, as a grader returns one.

        Raises:
            ValueError: ``shape`` is one Jev has no questions for.
        """
        try:
            if isinstance(shape, BeamJudgeShape) and query.get("category") == EVENT_ORDERING:
                return self._ordering(query, answer)
            if isinstance(shape, NuggetJudgeShape):
                return self._rubric(query, answer)
            if isinstance(shape, BinaryJudgeShape):
                return self._binary(query, answer, _rules_for(shape, query))
        except UnparseableVerdict as exc:
            return {"status": "unjudged", "error": str(exc)}
        raise ValueError(f"Jev has no questions for {type(shape).__name__}")

    def _binary(self, query: dict[str, Any], answer: str,
                kind: jq.BinaryRules) -> dict[str, Any]:
        """States the reference, contradicts nothing, does not decline -- or, for an abstention
        question, declines."""
        state = jq.binary_state(kind, question=query["text"], reference=gold_answer(query),
                                answer=answer)
        decision, record = self._ask(state, jq.binary_questions(kind))
        outcome = rules.binary(kind, decision.answers, self.threshold)
        record.update(rule=outcome.rule, passing=outcome.passing)
        said = ", ".join(f"{name} {value:.2f}" for name, value in outcome.passing.items())
        return _verdict(1.0 if outcome.passed else 0.0, outcome.passed,
                        f"{NO_RATIONALE} P(passing side): {said}; {outcome.rule}.", record,
                        probability=outcome.probability)

    def _rubric(self, query: dict[str, Any], answer: str) -> dict[str, Any]:
        """Each criterion's Score level mapped to 0 / 0.5 / 1, averaged; passed when all are 1."""
        criteria = list(query.get("rubric") or [])
        state = {"question": query["text"], "response": answer}
        decision, record = self._ask(state, jq.rubric_questions(criteria))
        outcomes = rules.rubric(criteria, decision.answers)
        record["rule"] = "each expected level rounded half up and mapped to 0, 0.5, 1; the mean"
        score = sum(o.score for o in outcomes) / len(outcomes) if outcomes else 0.0
        full = sum(1 for o in outcomes if o.score == 1.0)
        partial = sum(1 for o in outcomes if o.score == 0.5)
        record["nuggets"] = [asdict(o) for o in outcomes]
        return _verdict(score, score == 1.0,
                        f"{NO_RATIONALE} {full}/{len(outcomes)} criteria fully met, "
                        f"{partial} partly.", record)

    def _ordering(self, query: dict[str, Any], answer: str) -> dict[str, Any]:
        """Coverage, pairwise order and extras of the reference events (experimental)."""
        events = list(query.get("rubric") or [])
        state = {"question": query["text"], "answer": answer}
        decision, record = self._ask(state, jq.ordering_questions(events))
        record["rule"] = f"memrank.metrics.ordering.score_relations, Nouls at {self.threshold:g}"
        outcome = rules.ordering(len(events), decision.answers, self.threshold)
        scored = outcome.scored
        record.update(experimental=EXPERIMENTAL_ORDERING,
                      ordering={**asdict(scored), "present": outcome.present,
                                "relations": outcome.relations, "extras": outcome.extras})
        return _verdict(scored.score, scored.score == 1.0,
                        f"{NO_RATIONALE} {EXPERIMENTAL_ORDERING}. Order {scored.tau_norm:.2f} x "
                        f"coverage {scored.f1:.2f}; {len(outcome.present)}/{len(events)} "
                        f"events present.", record)


def _rules_for(shape: BinaryJudgeShape, query: dict[str, Any]) -> jq.BinaryRules:
    """The binary rules ``query`` is graded under: its benchmark's per-type ones, or generic.

    A negative question with no per-type rules has no Jev rule, and is refused rather than
    graded as if it were positive.
    """
    if shape.prompts is not None and shape.prompts is not LME_PROMPTS:
        raise ValueError("Jev has questions for LongMemEval's per-type prompts only")
    if shape.prompts is None:
        if is_negative(query):
            raise ValueError("Jev has no rule for a negative question without per-type rules")
        return jq.GENERIC
    key = query.get("judge_prompt_key") or query.get("category", "uncategorized")
    if key not in jq.LONGMEMEVAL:
        raise ValueError(f"Jev has no questions for judge prompt {key!r} "
                         f"(it has: {sorted(jq.LONGMEMEVAL)})")
    return jq.LONGMEMEVAL[key]
