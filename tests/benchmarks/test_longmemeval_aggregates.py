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
"""LongMemEval publishes THREE numbers; memrank published one micro-mean.

`print_qa_metrics.py` emits Overall Accuracy (micro over 500, abstention inside the
denominator), Task-averaged Accuracy (unweighted macro over the six types), and Abstention
Accuracy (separate, n=30). Half the disputes in the field survey are one of the first two
reported as the other: Mastra's 84.23 is macro presented unqualified, Supermemory's "95%" is a
macro of six ROUNDED INTEGERS, Redis prints 86.14 and 85.0 side by side.

The two differ most on `single-session-preference`, which is 6% of the micro denominator and
16.7% of the macro one -- and is simultaneously the type with the weakest judge-human agreement
(0.90). That is why the ADR makes micro the headline and prints the other two beside it.

The 30 abstention items are counted TWICE by design: inside their parent type bucket, and again
in their own readout. A reader who "corrects" that double-count is wrong.
"""
from memrank.benchmarks.longmemeval import LongMemEvalBenchmark


def _cells(**kw):
    return {t: list(scores) for t, scores in kw.items()}


def test_task_averaged_is_an_unweighted_macro_over_types():
    """Unweighted: a 30-question type counts as much as a 133-question one."""
    shape = LongMemEvalBenchmark().judge_shape()

    metrics = shape.aggregates(
        per_category={"multi-session": [1.0] * 100, "single-session-preference": [0.0] * 10},
        per_prompt_key={})

    # Micro would be 100/110 = 0.909; macro is (1.0 + 0.0) / 2.
    assert metrics["answer_correctness_task_averaged"] == 0.5


def test_abstention_is_reported_separately_with_its_n():
    shape = LongMemEvalBenchmark().judge_shape()

    metrics = shape.aggregates(
        per_category={"multi-session": [1.0, 1.0, 0.0]},
        per_prompt_key={"abstention": [1.0, 0.0]})

    assert metrics["abstention_accuracy"] == 0.5
    assert metrics["n_abstention"] == 2


def test_abstention_items_stay_inside_their_parent_type():
    """The double-count is the protocol's construction, not a bug to remove.

    An abstention item contributes to its type's cell AND to the abstention readout, so the two
    denominators overlap on purpose.
    """
    shape = LongMemEvalBenchmark().judge_shape()

    metrics = shape.aggregates(
        per_category={"temporal-reasoning": [1.0, 0.0]},   # one of these IS the abstention item
        per_prompt_key={"abstention": [0.0]})

    assert metrics["answer_correctness_task_averaged"] == 0.5
    assert metrics["n_abstention"] == 1


def test_absent_abstention_reports_none_not_zero():
    """A slice with no `_abs` item has no abstention accuracy -- which is not 0%."""
    shape = LongMemEvalBenchmark().judge_shape()

    metrics = shape.aggregates(per_category={"multi-session": [1.0]}, per_prompt_key={})

    assert metrics["abstention_accuracy"] is None
    assert metrics["n_abstention"] == 0


def test_no_cells_yields_no_aggregates():
    shape = LongMemEvalBenchmark().judge_shape()

    metrics = shape.aggregates(per_category={}, per_prompt_key={})

    assert metrics["answer_correctness_task_averaged"] is None


def test_other_benchmarks_declare_no_extra_aggregates():
    """LoCoMo and BEAM keep exactly the metrics they had; this hook is opt-in."""
    from memrank.benchmarks.locomo import LoCoMoBenchmark
    from memrank.judging.shape import BeamJudgeShape

    assert LoCoMoBenchmark().judge_shape().aggregates({"temporal": [1.0]}, {}) == {}
    assert BeamJudgeShape().aggregates({"summarization": [1.0]}, {}) == {}
