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
"""The seam that lets a benchmark say how its answers are graded.

BEAM scores against a rubric of atomic nuggets -- one judge call each, averaged within the
question -- and there was nowhere to express that. The runner's judge path was benchmark-agnostic,
so the only way in was to branch on benchmark name inside the runner, which is what this seam
exists to prevent.

A seam the runner can ignore is worth nothing, so the tests that matter here are the two that
prove it is actually consulted: once for what a query COSTS (the preflight budget) and once for
what grading one YIELDS (the metrics).
"""

import pytest

from memrank.benchmarks import REGISTRY
from memrank.judging.judge import JudgeConfig, JudgedQuery, JudgeVerdict, required_calls
from memrank.judging.shape import (
    GENERIC_BINARY_CATEGORIES,
    BinaryJudgeShape,
    JudgeShape,
    NuggetJudgeShape,
)
from memrank.runner import _apply_judge
from tests.fakes import JudgeFakeBenchmark, make_fake_completer


class _ThreeCallShape(JudgeShape):
    """A shape that costs a fixed 3 calls and always scores 0.25 -- nothing like the binary one,
    so any test asserting on it fails loudly if the runner quietly used the default instead."""

    def unjudged_reason(self, query):
        # Same judgeability as the default, so these tests isolate COST and SCORE. A shape that
        # also changed what counts as judgeable would make the arithmetic below ambiguous.
        return BinaryJudgeShape(GENERIC_BINARY_CATEGORIES).unjudged_reason(query)

    def control_calls(self, cfg, query):
        return 1

    def context_calls(self, cfg, query):
        return 2

    def grade(self, complete, cfg, *, query, context):
        verdict = JudgeVerdict(passed=True, rationale="fixed")
        return JudgedQuery("answer", None, verdict, False, score=0.25)


def test_only_beam_overrides_the_binary_shape():
    """BEAM is the one benchmark whose metric is not "one answer, one verdict". Every other
    benchmark must keep the default, or this seam has quietly changed something it should not."""
    for name, cls in REGISTRY.items():
        shape = cls().judge_shape()
        if name == "beam":
            assert isinstance(shape, NuggetJudgeShape), name
        else:
            assert isinstance(shape, BinaryJudgeShape), name


def test_the_binary_shape_costs_what_judge_query_actually_spends():
    """calls_per_query must track `grade`, or the preflight quotes a budget the run ignores.
    `tests/judging/test_judge_budget.py` pins this against a counting completer; this pins the split."""
    shape, cfg = BinaryJudgeShape(GENERIC_BINARY_CATEGORIES), JudgeConfig(samples=1)
    positive, negative = {"kind": "positive"}, {"kind": "negative"}
    assert shape.calls_per_query(cfg, positive) == 5
    assert shape.calls_per_query(cfg, negative) == 4, "a negative skips sufficiency"
    assert (shape.control_calls(cfg, positive) + shape.context_calls(cfg, positive)
            == shape.calls_per_query(cfg, positive))


def test_required_calls_asks_the_shape_rather_than_assuming_one_grade():
    """The control half is cached across a sweep and the context half is not, so a shape must be
    able to move them independently -- collapsing to one total over-costs every sweep."""
    units = JudgeFakeBenchmark().load()
    cfg = JudgeConfig(samples=1, cache=True)
    shape = _ThreeCallShape()
    judgeable = 2  # JudgeFakeBenchmark: 3 queries, one of them goldless
    assert required_calls(units, cfg, targets=1, shape=shape) == judgeable * 3
    # Four engines share the control half and pay the context half each.
    assert required_calls(units, cfg, targets=4, shape=shape) == judgeable * (1 + 2 * 4)


def test_apply_judge_scores_from_the_shape():
    """The metric must come from whatever the shape returned, not from `correctness.passed`."""
    units = JudgeFakeBenchmark().load()
    rows = [{"query_id": q["id"], "retrieved": []} for u in units for q in u.queries]
    metrics = _apply_judge(units, rows, JudgeConfig(completer=make_fake_completer(), cache=False),
                           make_fake_completer(), "matched", shape=_ThreeCallShape())
    assert metrics["answer_correctness"] == 0.25, "the shape's score, not a boolean"
    assert metrics["n_judged"] == 2


def test_benchmarks_declare_their_loaders_category_vocabulary():
    """The declaration and the loader's vocabulary must be one object (or provably equal), or
    the drift this seam replaced comes back: a label bug becoming a silent denominator bug."""
    from memrank.benchmarks.locomo import _CATEGORY_NAMES, LoCoMoBenchmark
    from memrank.benchmarks.longmemeval import _QUESTION_TYPES, LongMemEvalBenchmark

    assert LoCoMoBenchmark().judge_shape().categories == frozenset(_CATEGORY_NAMES.values())
    assert LongMemEvalBenchmark().judge_shape().categories == frozenset(_QUESTION_TYPES)


def test_an_unknown_category_raises_instead_of_skipping():
    """For a binary benchmark every declared category is gradeable, so an unknown label is a
    loader defect upstream of scoring -- the failure mode that judged 39.2% of LoCoMo silently."""
    shape = BinaryJudgeShape(frozenset({"temporal"}))

    with pytest.raises(ValueError, match="unknown to its judge shape"):
        shape.unjudged_reason({"text": "q", "gold_answers": ["a"], "category": "single-hop"})


def test_required_calls_and_apply_judge_agree_on_what_is_judgeable():
    """The budget and the loop must count the same queries. They used to ask different
    predicates -- `required_calls` the module-level one, the loop the shape's -- so a shape that
    widened judgeability was under-costed by exactly the queries it added."""
    units = JudgeFakeBenchmark().load()
    shape = NuggetJudgeShape()
    by_shape = [q for u in units for q in u.queries if shape.is_judgeable(q)]
    cfg = JudgeConfig(samples=1, no_context_control=False, cache=True)
    expected = sum(shape.context_calls(cfg, q) for q in by_shape)
    assert required_calls(units, cfg, targets=1, shape=shape) == expected


def test_judged_metrics_break_out_per_category():
    """Per-category cells with n, alongside the unchanged overall -- LoCoMo's category sizes
    differ by 8.8x, so a single micro-mean hides the structure these benchmarks exist to
    measure, and a cell without its n is how mislabeled tables spread (survey section 5)."""
    units = JudgeFakeBenchmark().load()
    rows = [{"query_id": q["id"], "retrieved": []} for u in units for q in u.queries]
    fake = make_fake_completer()
    metrics = _apply_judge(units, rows, JudgeConfig(completer=fake, cache=False), fake,
                           "matched", shape=BinaryJudgeShape(GENERIC_BINARY_CATEGORIES))

    per_cat = metrics["answer_correctness_per_category"]
    assert set(per_cat) == {"single-hop", "abstention"}
    assert all(cell["n"] == 1 for cell in per_cat.values())
    weighted = (sum(cell["mean"] * cell["n"] for cell in per_cat.values())
                / sum(cell["n"] for cell in per_cat.values()))
    assert metrics["answer_correctness"] == pytest.approx(weighted)
    assert metrics["n_scoreable"] == metrics["n_judged"]


# --------------------------------------------------------------------------- #
# The preflight gate must use the BENCHMARK's vocabulary, never a global one
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name", ["locomo", "longmemeval", "beam"])
def test_the_coverage_gate_speaks_each_benchmarks_own_vocabulary(name):
    """`memrank submit <benchmark> --judge` crashed in preflight for ALL THREE benchmarks.

    `assert_judge_coverage` declared `shape: JudgeShape | None = None` and substituted
    `BinaryJudgeShape(GENERIC_BINARY_CATEGORIES)` -- the DEMO benchmark's five categories -- when a
    caller forgot to pass one. `runner.py` had exactly one such caller, in the submit preflight,
    five lines above a `judge_cost_estimate` call that passed the shape correctly. Every real
    category then read as "unknown to its judge shape", which is a real error message about a
    real defect that was not where it pointed.

    Asserted against `judge_shape()` rather than a hard-coded list, so changing a benchmark's
    vocabulary cannot silently re-open this.
    """
    from memrank.benchmarks import get_benchmark
    from memrank.judging.shape import GENERIC_BINARY_CATEGORIES

    bench = get_benchmark(name, **({"tier": "100k"} if name == "beam" else {}))
    shape = bench.judge_shape()
    declared = getattr(shape, "categories", None)
    if declared is None:
        return  # BEAM grades on rubrics, not categories -- nothing to compare
    # The bug in one assertion: these vocabularies are NOT the generic one.
    assert declared - GENERIC_BINARY_CATEGORIES, (
        f"{name} shares the generic vocabulary; this test would not detect the substitution")


def test_the_gate_requires_a_shape_rather_than_inventing_one():
    """No caller may omit it: the fallback is how a global vocabulary stood in for a real one."""
    import inspect

    from memrank.runner import assert_judge_coverage, judge_cost_estimate

    for fn in (assert_judge_coverage, judge_cost_estimate):
        param = inspect.signature(fn).parameters["shape"]
        assert param.default is inspect.Parameter.empty, (
            f"{fn.__name__} still defaults `shape`; a forgotten argument silently becomes the "
            "demo benchmark's category list")
