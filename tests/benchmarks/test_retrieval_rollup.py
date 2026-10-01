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
"""Per-unit retrieval metrics must become a run-level number, or nobody can read them.

`score()` returns LongMemEval's `recall_all@k` / `ndcg_any@k` per unit and LoCoMo's
`evidence_recall` per unit; `_aggregate_cell` stored both under `per_unit` and `_mean_composite`
only ever means `composite`, which is None for both benchmarks by design. So a 500-unit run
produced 500 recall values and no headline -- the metric existed and was unreadable.

The rollup is a WEIGHTED micro-mean over each unit's own denominator, not a mean of unit means.
For LongMemEval each unit holds exactly one query so the weight is 0 or 1, but writing it
weighted is what keeps it correct if that changes and what makes the real denominator (419, not
500 -- abstention and assistant-side questions are excluded from retrieval only) explicit.
"""
from memrank.benchmarks.locomo import LoCoMoBenchmark
from memrank.benchmarks.longmemeval import LongMemEvalBenchmark


def _lme_unit(recall5, recall10, n=1, ndcg5=0.0, ndcg10=0.0):
    return {"composite": None, "retrieval": {
        "recall_all@5": recall5, "recall_all@10": recall10,
        "ndcg_any@5": ndcg5, "ndcg_any@10": ndcg10, "n_retrieval_scoreable": n}}


def test_rollup_is_a_weighted_micro_mean():
    out = LongMemEvalBenchmark().rollup([
        _lme_unit(1.0, 1.0), _lme_unit(0.0, 1.0), _lme_unit(0.0, 0.0), _lme_unit(1.0, 1.0)])

    assert out["recall_all@5"] == 0.5
    assert out["recall_all@10"] == 0.75
    assert out["n_retrieval_scoreable"] == 4


def test_units_excluded_from_retrieval_do_not_dilute_the_mean():
    """An abstention unit contributes n=0 and must not count as a zero."""
    out = LongMemEvalBenchmark().rollup([
        _lme_unit(1.0, 1.0, n=1),
        _lme_unit(None, None, n=0),   # abstention: excluded from retrieval only
    ])

    assert out["recall_all@5"] == 1.0
    assert out["n_retrieval_scoreable"] == 1


def test_nothing_scoreable_reports_none_not_zero():
    out = LongMemEvalBenchmark().rollup([_lme_unit(None, None, n=0)])

    assert out["recall_all@5"] is None
    assert out["ndcg_any@10"] is None
    assert out["n_retrieval_scoreable"] == 0


def test_rollup_of_no_units_is_empty_not_zero():
    assert LongMemEvalBenchmark().rollup([])["recall_all@5"] is None


def test_weighting_uses_each_unit_denominator():
    """A unit covering three scoreable queries outweighs one covering a single query."""
    out = LongMemEvalBenchmark().rollup([
        _lme_unit(1.0, 1.0, n=3),   # three queries, all hit
        _lme_unit(0.0, 0.0, n=1),   # one query, missed
    ])

    assert out["recall_all@5"] == 0.75


def test_locomo_evidence_recall_rolls_up_too():
    """The identical gap; fixing one and leaving the other is how the second gets forgotten."""
    out = LoCoMoBenchmark().rollup([
        {"evidence_recall": 1.0, "n_queries_with_evidence": 10},
        {"evidence_recall": 0.5, "n_queries_with_evidence": 10},
    ])

    assert out["evidence_recall"] == 0.75
    assert out["n_queries_with_evidence"] == 20


def test_locomo_rollup_reports_none_when_no_unit_had_evidence():
    out = LoCoMoBenchmark().rollup([{"evidence_recall": None, "n_queries_with_evidence": 0}])

    assert out["evidence_recall"] is None


def test_beam_declares_no_rollup():
    """Opt-in: a benchmark that declares none keeps exactly the artifact it had."""
    from memrank.benchmarks.beam import BEAMBenchmark

    assert BEAMBenchmark().rollup([{"composite": 1.0}]) == {}
