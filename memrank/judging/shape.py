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
"""How a benchmark's answers are graded -- the shape of judging, not its prompts.

Every benchmark used one grading shape: generate an answer from the retrieved context, ask a
judge one question about it, get one boolean. That is right for LoCoMo and LongMemEval, whose
questions each have a single gold answer.

It is not what BEAM specifies. BEAM scores against a `rubric` of atomic nuggets -- one judge call
PER NUGGET, each scored {0, 0.5, 1}, averaged within the question. That is a different shape,
not a different prompt: N calls where there was one, and a mean where there was a boolean.
There was nowhere to say it. The runner's judge path is benchmark-agnostic by construction, so
the only way to express it was to branch on benchmark name inside the runner -- which is the
thing this seam exists to prevent.

`docs/evaluations.md` told benchmark authors to accept an injectable `judge` callable, and
`memrank/benchmarks/beam.py` promised the same. Neither existed. This is that promise, made real
and made narrower: a benchmark declares its shape, and the runner asks the shape both what a query
COSTS and what grading one YIELDS. The cost half matters as much as the grading half -- a preflight
that cannot size the shape it is about to run quotes a budget the run does not honour.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any

from memrank.judging.judge import (
    Completer,
    JudgeConfig,
    JudgedQuery,
    JudgeVerdict,
    NuggetScore,
    extract_events,
    generate_answer,
    gold_answer,
    grade_nugget,
    judge_answer,
    judge_equivalent,
    judge_query,
    judge_rubric_sufficiency,
    no_context_calls,
    with_context_calls,
)
from memrank.metrics.ordering import OrderingScore, score_sequence


def is_negative(query: dict[str, Any]) -> bool:
    """Whether a query expects the system NOT to answer affirmatively.

    One reading of `kind`, shared by the grader and the budget so the two cannot disagree about
    what a query costs -- a negative skips sufficiency, and skipping it is worth `samples` calls.
    """
    return query.get("kind") == "negative"


@dataclass(frozen=True)
class AnswerGrade:
    """One answer graded against its reference -- the correctness half of grading, alone.

    What an agent evaluation needs: the agent answered for itself, so there is no context to
    judge for sufficiency and no reader to run. :meth:`JudgeShape.grade` composes this with those
    stages when memrank produced the answer from retrieved context.
    """

    correctness: JudgeVerdict
    score: float
    nuggets: list[NuggetScore] | None = None
    ordering: dict[str, Any] | None = None


class JudgeShape(ABC):
    """How one benchmark's queries are graded, and what grading one costs.

    Cost is declared as TWO halves rather than one total, because the two are billed differently and
    `memrank.judging.judge.required_calls` depends on the distinction. The control half is built
    only from the question, the gold answer and the model, so two engines in one sweep send byte-
    identical requests and the second is a cache hit -- a sweep pays it once. The context half
    carries the engine's own retrieval, so nothing can share it and it multiplies by targets.
    Collapsing them into one number would over-cost every sweep by the control's share. """

    @abstractmethod
    def unjudged_reason(self, query: dict[str, Any]) -> str | None:
        """Why ``query`` cannot be graded, or None when it can.

        The reason, not just a boolean, because `_judge_metrics` attributes unjudged queries to a
        cause and "no gold answer" and "this shape does not score that category" are different
        problems -- only one is worth re-running to fix.
        """

    def is_judgeable(self, query: dict[str, Any]) -> bool:
        """Whether ``query`` yields a graded score under this shape.

        Derived from :meth:`unjudged_reason` rather than declared separately, so the predicate the
        coverage gate uses cannot drift from the reason the metrics report. Four things ask this --
        the pre-run coverage gate, the budget estimate, the progress denominator and the judging
        loop -- and they must not disagree about what the run is going to do.
        """
        return self.unjudged_reason(query) is None

    @abstractmethod
    def control_calls(self, cfg: JudgeConfig, query: dict[str, Any]) -> int:
        """Engine-INDEPENDENT calls for ``query`` -- the no-context control. Shared across a sweep."""

    @abstractmethod
    def context_calls(self, cfg: JudgeConfig, query: dict[str, Any]) -> int:
        """Engine-SPECIFIC calls for ``query``. Multiplies by the number of targets."""

    @abstractmethod
    def grade(self, complete: Completer, cfg: JudgeConfig, *,
              query: dict[str, Any], context: str) -> JudgedQuery:
        """Grade one query against what the engine retrieved for it."""

    def grade_answer(self, complete: Completer, cfg: JudgeConfig, *,
                     query: dict[str, Any], answer: str) -> AnswerGrade:
        """Grade an answer the system under test produced itself, with the judge model only.

        Not abstract, so a shape written only for retrieval grading keeps constructing; asking it
        to grade an agent's answer is what fails, and says which shape could not.
        """
        raise NotImplementedError(
            f"{type(self).__name__} cannot grade an agent's own answer; implement grade_answer")

    def aggregates(self, per_category: dict[str, list[float]],
                   per_prompt_key: dict[str, list[float]]) -> dict[str, Any]:
        """Extra NAMED metrics this protocol publishes, beyond the micro-mean every shape emits.

        Opt-in and empty by default, so LoCoMo and BEAM keep exactly the metrics they had.

        It exists because a benchmark's headline is part of its protocol, not a presentation
        choice. LongMemEval publishes three numbers -- micro over all questions, an unweighted
        macro over its six types, and abstention separately -- and half the disputes in its
        literature are one of the first two reported as the other. Deriving the macro from the
        per-category cells rather than recomputing it from raw verdicts is deliberate: two
        computations of the same quantity can drift, one cannot.
        """
        return {}

    def calls_per_query(self, cfg: JudgeConfig, query: dict[str, Any]) -> int:
        """Total completer calls :meth:`grade` will make for ``query``.

        Must match :meth:`grade` exactly. `tests/judging/test_judge_budget.py` runs both against a
        counting completer and fails if they drift, because a preflight built on a stale count
        refuses runs that would have fitted or admits ones that die part-way.
        """
        return self.control_calls(cfg, query) + self.context_calls(cfg, query)


#: Categories for benchmarks that never declared their own vocabulary (the demo benchmark and
#: ad-hoc units). A benchmark with a real category taxonomy declares it in its own
#: ``judge_shape()`` override -- LoCoMo and LongMemEval do -- so this set never grows to cover a
#: real benchmark. Growing it here is the failure mode the per-benchmark declaration replaced:
#: one global allowlist, keyed on strings, where a loader's labelling bug silently became a
#: scoring bug (audit F2: LoCoMo judged 39.2% of itself for weeks).
GENERIC_BINARY_CATEGORIES = frozenset({
    "single-hop", "temporal", "preference", "abstention", "uncategorized",
})


class BinaryJudgeShape(JudgeShape):
    """One answer, one verdict, one boolean -- the shape every benchmark used before this existed.

    Delegates rather than reimplements: `judge_query`, `no_context_calls` and `with_context_calls`
    are unchanged and remain the single description of this sequence. The class is a seam, not a
    rewrite.

    ``categories`` is the benchmark's own declaration of its category vocabulary -- required, no
    default, so a benchmark cannot inherit an allowlist written for a different benchmark. A
    query whose category is not declared RAISES rather than skips: for a binary benchmark every
    category is gradeable, so an unknown label is a loader defect upstream of scoring, and a
    skip is how such a defect ran silently for weeks (audit F2).
    """

    def __init__(self, categories: frozenset[str],
                 prompts: dict[str, tuple[str, str]] | None = None) -> None:
        self.categories = categories
        #: Optional per-question-type grading prompts, keyed by the query's `judge_prompt_key`
        #: (which defaults to its category). A benchmark supplies these when its protocol defines
        #: type-conditional grading rules -- LongMemEval ships six, and they carry the only
        #: published human-agreement figure in this repo. When None, every query is graded by the
        #: generic pair and the bytes are exactly what they were before this parameter existed.
        self.prompts = prompts

    def _prompt_for(self, query: dict[str, Any]) -> tuple[str, str] | None:
        """The (system, gold-label) pair for ``query``, or None to use the generic pair.

        Raises on an unknown key for the same reason an unknown category raises: a benchmark that
        declares per-type prompts and then emits a key none of them covers has a labelling bug
        upstream of scoring, and silently grading it with a different prompt is how 12% of this
        benchmark was graded against the wrong object for months.
        """
        if self.prompts is None:
            return None
        key = query.get("judge_prompt_key") or query.get("category", "uncategorized")
        if key not in self.prompts:
            raise ValueError(
                f"query declares judge prompt {key!r}, which its shape does not define "
                f"(declared: {sorted(self.prompts)})")
        return self.prompts[key]

    def unjudged_reason(self, query: dict[str, Any]) -> str | None:
        # Gold before category, matching the order `_apply_judge` has always checked them in, so
        # the two reason counts stay attributed exactly as before.
        if not gold_answer(query):
            return "no_gold"
        category = query.get("category", "uncategorized")
        if category not in self.categories:
            raise ValueError(
                f"loader emitted category {category!r} unknown to its judge shape "
                f"(declared: {sorted(self.categories)}) -- a label bug upstream of scoring, "
                "not a skippable query")
        return None

    def control_calls(self, cfg: JudgeConfig, query: dict[str, Any]) -> int:
        return no_context_calls(cfg)

    def context_calls(self, cfg: JudgeConfig, query: dict[str, Any]) -> int:
        return with_context_calls(cfg, negative=is_negative(query))

    def grade(self, complete: Completer, cfg: JudgeConfig, *,
              query: dict[str, Any], context: str) -> JudgedQuery:
        return judge_query(complete, question=query["text"], context=context,
                           gold=gold_answer(query), cfg=cfg, negative=is_negative(query),
                           query_date=query.get("query_timestamp"),
                           prompt=self._prompt_for(query))

    def grade_answer(self, complete: Completer, cfg: JudgeConfig, *,
                     query: dict[str, Any], answer: str) -> AnswerGrade:
        verdict = judge_answer(complete, question=query["text"], answer=answer,
                               gold=gold_answer(query), model=cfg.judge_model,
                               samples=cfg.samples, negative=is_negative(query),
                               prompt=self._prompt_for(query))
        return AnswerGrade(verdict, score=1.0 if verdict.passed else 0.0)


#: Abilities the plain rubric grader cannot score. `event_ordering` needs rank correlation over an
#: ordered gold list, which `BeamJudgeShape` below supplies -- so BEAM passes an empty set and this
#: default exists for any other benchmark adopting the rubric grader without an ordering path.
#: Everything else BEAM ships, summarization included, is scored by exactly this grader: BEAM
#: treats summarization no differently from information extraction, and the "partial-credit
#: rubric" it was once said to need IS the rubric.
UNSCORED_BY_RUBRIC = frozenset({"event_ordering"})


class NuggetJudgeShape(JudgeShape):
    """BEAM's protocol: one judge call per rubric criterion, {0, 0.5, 1}, averaged.

    No gold answer is used or required. The rubric is the reference, which is what makes
    `instruction_following` and `preference_following` gradeable at all -- BEAM ships no answer for
    either, only a description of the behaviour expected, and grading a response against that
    description as though it were an answer grades the wrong object.
    """

    def __init__(self, unscored: frozenset[str] = UNSCORED_BY_RUBRIC) -> None:
        self.unscored = unscored

    def unjudged_reason(self, query: dict[str, Any]) -> str | None:
        if query.get("category", "uncategorized") in self.unscored:
            return "unsupported_category"
        if not query.get("rubric"):
            return "no_rubric"
        return None

    def control_calls(self, cfg: JudgeConfig, query: dict[str, Any]) -> int:
        if not cfg.no_context_control:
            return 0
        return 1 + cfg.samples * len(query.get("rubric") or [])

    def context_calls(self, cfg: JudgeConfig, query: dict[str, Any]) -> int:
        sufficiency = 0 if is_negative(query) else cfg.samples
        return sufficiency + 1 + cfg.samples * len(query.get("rubric") or [])

    def _score(self, complete: Completer, cfg: JudgeConfig, *, question: str, answer: str,
               rubric: list[str]) -> list[NuggetScore]:
        return [grade_nugget(complete, question=question, answer=answer, nugget=item,
                             model=cfg.judge_model, samples=cfg.samples) for item in rubric]

    def grade(self, complete: Completer, cfg: JudgeConfig, *,
              query: dict[str, Any], context: str) -> JudgedQuery:
        question, rubric = query["text"], list(query.get("rubric") or [])
        answered_without_context = False
        if cfg.no_context_control:
            # Graded the same way as the real answer, so the two are on one scale. This half is
            # engine-independent and therefore cache-shared across a sweep -- it is paid once.
            blind = generate_answer(complete, question=question, context="",
                                    model=cfg.answer_model, prompt=cfg.answer_prompt)
            answered_without_context = _mean(
                self._score(complete, cfg, question=question, answer=blind, rubric=rubric)) == 1.0
        sufficiency = None
        if not is_negative(query):
            # Still skipped for abstention: BEAM builds those probes to be unanswerable, so asking
            # whether the snippets suffice can only fail and would measure nothing.
            sufficiency = judge_rubric_sufficiency(complete, question=question, context=context,
                                                   rubric=rubric, model=cfg.judge_model,
                                                   samples=cfg.samples)
        answer = generate_answer(complete, question=question, context=context,
                                 model=cfg.answer_model, prompt=cfg.answer_prompt)
        graded = self.grade_answer(complete, cfg, query=query, answer=answer)
        return JudgedQuery(answer, sufficiency, graded.correctness, answered_without_context,
                           score=graded.score, nuggets=graded.nuggets)

    def grade_answer(self, complete: Completer, cfg: JudgeConfig, *,
                     query: dict[str, Any], answer: str) -> AnswerGrade:
        rubric = list(query.get("rubric") or [])
        scores = self._score(complete, cfg, question=query["text"], answer=answer, rubric=rubric)
        score = _mean(scores)
        full = sum(1 for s in scores if s.score == 1.0)
        partial = sum(1 for s in scores if 0.0 < s.score < 1.0)
        # `passed` means the WHOLE rubric was satisfied. Any other cut would be a threshold we
        # invented; the graded value rides in `score`, which is what the metric actually uses.
        correctness = JudgeVerdict(
            passed=score == 1.0,
            rationale=f"{full}/{len(scores)} criteria fully satisfied, {partial} partially",
            samples=cfg.samples)
        return AnswerGrade(correctness, score=score, nuggets=scores)


#: The ability BEAM scores by rank correlation rather than by averaging criteria.
EVENT_ORDERING = "event_ordering"

#: How many predicted events to align, as a multiple of the reference count.
#:
#: Alignment is quadratic -- every predicted event is compared against the unmatched reference
#: events until one matches -- so an answer that lists fifty items would spend hundreds of judge
#: calls on a response already destined to score near zero on precision. BEAM does not cap,
#: because splitting on newlines caps them implicitly; extracting properly removes that accident,
#: so the bound has to be deliberate. Twice the reference length is generous: a correct answer
#: lists exactly as many events as the rubric names.
_MAX_PREDICTED_FACTOR = 2


class BeamJudgeShape(NuggetJudgeShape):
    """BEAM's grader, which is two metrics: rubric nuggets, and rank correlation for one ability.

    Per-category dispatch because BEAM's own harness dispatches per ability. `event_ordering` asks
    for events IN ORDER, so its rubric is an ordered list and averaging it would score a reversed
    answer identically to a correct one.
    """

    def __init__(self) -> None:
        super().__init__(unscored=frozenset())

    @staticmethod
    def _is_ordering(query: dict[str, Any]) -> bool:
        return query.get("category") == EVENT_ORDERING

    def _alignment_calls(self, cfg: JudgeConfig, query: dict[str, Any]) -> int:
        """Extraction plus the WORST CASE alignment cost.

        An upper bound, deliberately, and the reason `calls_per_query` is not an equality for this
        ability: how many events the model lists is not knowable before it answers. `required_calls`
        is already documented as an upper bound for the same class of reason, and erring high
        refuses a run that would have fitted rather than admitting one that dies part-way.
        """
        gold = len(query.get("rubric") or [])
        return 1 + cfg.samples * (_MAX_PREDICTED_FACTOR * gold) * gold

    def control_calls(self, cfg: JudgeConfig, query: dict[str, Any]) -> int:
        if not self._is_ordering(query):
            return super().control_calls(cfg, query)
        return 0 if not cfg.no_context_control else 1 + self._alignment_calls(cfg, query)

    def context_calls(self, cfg: JudgeConfig, query: dict[str, Any]) -> int:
        if not self._is_ordering(query):
            return super().context_calls(cfg, query)
        sufficiency = 0 if is_negative(query) else cfg.samples
        return sufficiency + 1 + self._alignment_calls(cfg, query)

    def _align(self, complete: Completer, cfg: JudgeConfig, *, question: str, answer: str,
               gold: list[str]) -> tuple[OrderingScore, list[dict[str, Any]]]:
        """Extract the answer's events, align them to `gold`, and score the aligned sequence."""
        predicted = extract_events(complete, question=question, answer=answer,
                                   model=cfg.judge_model)[:_MAX_PREDICTED_FACTOR * len(gold)]
        unused = list(range(len(gold)))
        canonical: list[str] = []
        pairs: list[dict[str, Any]] = []
        for event in predicted:
            match = next(
                (i for i in unused
                 if judge_equivalent(complete, reference=gold[i], candidate=event,
                                     model=cfg.judge_model, samples=cfg.samples).passed),
                None)
            if match is None:
                canonical.append(event)
            else:
                canonical.append(gold[match])
                unused.remove(match)
            pairs.append({"predicted": event,
                          "matched_reference": None if match is None else gold[match]})
        return score_sequence(gold, canonical), pairs

    def grade(self, complete: Completer, cfg: JudgeConfig, *,
              query: dict[str, Any], context: str) -> JudgedQuery:
        if not self._is_ordering(query):
            return super().grade(complete, cfg, query=query, context=context)
        question, gold = query["text"], list(query.get("rubric") or [])
        answered_without_context = False
        if cfg.no_context_control:
            blind = generate_answer(complete, question=question, context="",
                                    model=cfg.answer_model, prompt=cfg.answer_prompt)
            answered_without_context = self._align(
                complete, cfg, question=question, answer=blind, gold=gold)[0].score == 1.0
        sufficiency = None
        if not is_negative(query):
            sufficiency = judge_rubric_sufficiency(complete, question=question, context=context,
                                                   rubric=gold, model=cfg.judge_model,
                                                   samples=cfg.samples)
        answer = generate_answer(complete, question=question, context=context,
                                 model=cfg.answer_model, prompt=cfg.answer_prompt)
        graded = self.grade_answer(complete, cfg, query=query, answer=answer)
        return JudgedQuery(answer, sufficiency, graded.correctness, answered_without_context,
                           score=graded.score, ordering=graded.ordering)

    def grade_answer(self, complete: Completer, cfg: JudgeConfig, *,
                     query: dict[str, Any], answer: str) -> AnswerGrade:
        if not self._is_ordering(query):
            return super().grade_answer(complete, cfg, query=query, answer=answer)
        gold = list(query.get("rubric") or [])
        scored, pairs = self._align(complete, cfg, question=query["text"], answer=answer,
                                    gold=gold)
        correctness = JudgeVerdict(
            passed=scored.score == 1.0,
            rationale=(f"order {scored.tau_norm:.2f} x coverage {scored.f1:.2f}; "
                       f"{sum(1 for p in pairs if p['matched_reference'])}/{len(gold)} matched"),
            samples=cfg.samples)
        return AnswerGrade(correctness, score=scored.score,
                           ordering={**asdict(scored), "alignment": pairs})


def _mean(scores: list[NuggetScore]) -> float:
    """Mean nugget score, or 0.0 for an empty rubric -- which judgeability already excludes."""
    return sum(s.score for s in scores) / len(scores) if scores else 0.0
