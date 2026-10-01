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
"""Either half of an evaluation may be the person's, and never the half they did not bring.

The three runs step 8 exists for, written the way a person would write them, plus the seam that
keeps them safe: both halves declare what they produce and expect, and a disagreement is refused
before any item runs rather than averaged.
"""

from __future__ import annotations

from typing import Any

import pytest

# `memrank.run` is now the typed run over the seven (memrank/instrument/); the cell run
# that produces an `EvalResult` is imported from its own module, which is where the CLI
# and the cloud reach it too.
from memrank import (
    AdapterResponse,
    BenchmarkUnit,
    ComposedEvaluation,
    CriteriaMismatch,
    Document,
    Scorer,
    SpanRecall,
)
from memrank.benchmarks.demo import DemoBenchmark
from memrank.evaluation.api import run as run_cell

# These tests exercise the deprecated in-process route on purpose; its notice is expected here.
pytestmark = pytest.mark.filterwarnings(
    "ignore:.*is deprecated and will be removed:DeprecationWarning")

MY_UNITS = [BenchmarkUnit(
    unit_id="acme", isolation_id="acme",
    documents=[Document(id="d1", user_id="acme",
                        content="Acme upgraded to the enterprise plan in March."),
               Document(id="d2", user_id="acme",
                        content="The outage was traced to an expired webhook secret.")],
    queries=[{"id": "q1", "text": "What plan is Acme on?",
              "required_spans": ["enterprise"], "category": "single-hop"},
             {"id": "q2", "text": "What caused the outage?",
              "required_spans": ["webhook secret"], "category": "single-hop"}])]


class MyScorer(Scorer):
    """My criteria, my names for them, my reasons. The questions are not mine."""

    criterion_names = ("composite", "said_it_briefly")
    identity = "my-own-rules/v1"
    quality_metric = "judged_answer_correctness"

    def score(self, unit: BenchmarkUnit,
              responses: list[AdapterResponse]) -> dict[str, Any]:
        seen = [doc.content for response in responses for doc in response.documents]
        return {"composite": 1.0 if seen else 0.0,
                "said_it_briefly": all(len(text) < 200 for text in seen),
                "per_category": {}, "n_queries": len(unit.queries)}


def test_my_questions_with_memranks_scorer() -> None:
    """The scoring half is not supplied, and is not written either."""
    evaluation = ComposedEvaluation(name="mine", questions=MY_UNITS)

    result = run_cell("word-overlap", evaluation, repeats=1)

    assert isinstance(evaluation.scorer, SpanRecall)
    assert result.composite == 1.0
    assert result.per_unit[0]["metric"] == SpanRecall.METRIC_LABEL


def test_memranks_questions_with_my_scorer() -> None:
    """The question half is not supplied, and no loader is written."""
    evaluation = ComposedEvaluation(name="demo+mine", questions=DemoBenchmark(),
                                   scorer=MyScorer())

    result = run_cell("word-overlap", evaluation, repeats=1)

    assert result.composite == 1.0
    assert result.per_unit[0]["said_it_briefly"] is True
    # The questions' own declarations survive the swap; the scorer decides only the KIND.
    assert evaluation.dataset_version == "memrank-demo@v1"
    assert evaluation.is_synthetic is True
    assert evaluation.quality_metric == "judged_answer_correctness"


def test_both_halves_mine() -> None:
    evaluation = ComposedEvaluation(name="mine", questions=MY_UNITS, scorer=MyScorer())

    result = run_cell("word-overlap", evaluation, repeats=1)

    assert result.composite == 1.0
    assert result.benchmark == "mine"


def test_the_three_runs_stay_distinguishable_in_the_receipt() -> None:
    """Two scorers are not one reproducible run, so the receipt names which decided."""
    mine = ComposedEvaluation(name="demo+mine", questions=DemoBenchmark(), scorer=MyScorer())
    theirs = ComposedEvaluation(name="demo+mine", questions=DemoBenchmark())

    assert mine.config_for_receipt()["scorer"] == "my-own-rules/v1"
    assert theirs.config_for_receipt()["scorer"] == "span-recall"
    assert mine.config_for_receipt()["questions"] == "demo"
    assert mine.config_for_receipt() != theirs.config_for_receipt()


# ---- the criterion-name check -----------------------------------------------------------


class OtherNames(Scorer):
    """A scorer with its own vocabulary. Nothing is wrong with it; it is simply not the same."""

    criterion_names = ("says_the_thing",)
    identity = "other-names/v1"

    def score(self, unit: BenchmarkUnit,
              responses: list[AdapterResponse]) -> dict[str, Any]:
        return {"says_the_thing": 1.0, "composite": 1.0}


class LiteralQuestions(DemoBenchmark):
    """Questions whose own aggregation names its criterion literally -- so it declares it."""

    name = "literal"
    criterion_names = ("composite",)


def test_a_mismatch_is_refused_at_composition_naming_both_lists() -> None:
    with pytest.raises(CriteriaMismatch) as raised:
        ComposedEvaluation(name="clash", questions=LiteralQuestions(), scorer=OtherNames())

    message = str(raised.value)
    assert "composite" in message and "says_the_thing" in message
    assert "only in the questions ['composite']" in message
    assert "only in the scorer ['says_the_thing']" in message


def test_the_refusal_happens_before_any_item_runs() -> None:
    """A scorer swapped after construction is still refused before the engine is touched."""
    evaluation = ComposedEvaluation(name="clash", questions=LiteralQuestions())
    evaluation.scorer = OtherNames()

    with pytest.raises(CriteriaMismatch):
        evaluation.load()


def test_silence_on_either_side_is_not_a_mismatch() -> None:
    """The five registered benchmarks declare no criteria, so someone else's are accepted."""
    assert DemoBenchmark.criterion_names == ()
    evaluation = ComposedEvaluation(name="demo+other", questions=DemoBenchmark(),
                                   scorer=OtherNames())

    assert evaluation.load()[0].unit_id == DemoBenchmark().load()[0].unit_id


def test_a_declared_criterion_that_never_arrives_is_refused_loudly() -> None:
    class Forgetful(Scorer):
        criterion_names = ("composite", "grounded")

        def score(self, unit, responses):  # type: ignore[no-untyped-def]
            return {"composite": 1.0}

    evaluation = ComposedEvaluation(name="forgetful", questions=MY_UNITS, scorer=Forgetful())

    with pytest.raises(CriteriaMismatch, match="grounded"):
        evaluation.score(MY_UNITS[0], [])


# ---- the judge half of the same check ---------------------------------------------------
#
# `judge_shape()` is the only place a benchmark's grading rules and its loader's labels are held
# together: `BinaryJudgeShape._prompt_for` (memrank/judging/shape.py:170) raises on a query whose
# declared prompt key the shape does not define, because a silent skip "is how 12% of this
# benchmark was graded against the wrong object for months". Separating the halves removes the
# class that held them together, so the composition compares the two declarations instead.


class _JudgingScorer(Scorer):
    """A scorer that grades, carrying its own per-type prompts."""

    def __init__(self, prompt_keys: tuple[str, ...]) -> None:
        self._prompt_keys = prompt_keys

    def judge_shape(self):  # type: ignore[no-untyped-def]
        from memrank.judging.shape import GENERIC_BINARY_CATEGORIES, BinaryJudgeShape

        return BinaryJudgeShape(
            GENERIC_BINARY_CATEGORIES,
            prompts=dict.fromkeys(self._prompt_keys, ("system", "gold")))

    def score(self, unit: BenchmarkUnit,
              responses: list[AdapterResponse]) -> dict[str, Any]:
        return {"composite": 1.0}


def _longmemeval_prompt_keys() -> tuple[str, ...]:
    from memrank.judging.prompts import LME_PROMPTS

    return tuple(LME_PROMPTS)


def test_a_scorer_declaring_a_prompt_key_the_questions_do_not_carry_is_refused() -> None:
    from memrank.benchmarks.longmemeval import LongMemEvalBenchmark

    with pytest.raises(CriteriaMismatch) as raised:
        ComposedEvaluation(name="lme+mine", questions=LongMemEvalBenchmark(),
                          scorer=_JudgingScorer(("invented-type",)))

    message = str(raised.value)
    assert "invented-type" in message
    assert "abstention" in message           # one of the seven the questions do carry
    assert "only in the scorer ['invented-type']" in message
    assert "12% of a benchmark was graded against the wrong object" in message
    assert "BinaryJudgeShape._prompt_for" in message


def test_a_scorer_whose_prompts_match_the_questions_composes() -> None:
    from memrank.benchmarks.longmemeval import LongMemEvalBenchmark

    evaluation = ComposedEvaluation(name="lme+mine", questions=LongMemEvalBenchmark(),
                                   scorer=_JudgingScorer(_longmemeval_prompt_keys()))

    assert set(evaluation.judge_shape().prompts) == set(_longmemeval_prompt_keys())


def test_a_scorer_with_no_shape_leaves_the_questions_own_grading_in_force() -> None:
    from memrank.benchmarks.longmemeval import LongMemEvalBenchmark

    evaluation = ComposedEvaluation(name="lme+span", questions=LongMemEvalBenchmark())

    assert set(evaluation.judge_shape().prompts) == set(_longmemeval_prompt_keys())
