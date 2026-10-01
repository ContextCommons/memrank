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
"""LongMemEval's two official retrieval metrics -- and the three traps in reimplementing them.

1. `recall_all@k` is ALL-evidence-in-top-k, a per-question 0/1 indicator. Fractional recall
   under that name inflates the number substantially.
2. Gold must come from `answer_session_ids`, not the `has_answer` turn flags. They disagree on
   62 of 500 questions and `has_answer` is always the strict SUBSET, so building on it makes the
   metric easier -- one fewer document that has to appear in top-k.
3. The 30 abstention items and the 51 assistant-side questions are excluded from RETRIEVAL ONLY.
   Issue #16 records the author confirming they must not reach the QA denominator, which stays
   500; several publications made exactly that error.
"""
from memrank.benchmarks.longmemeval import LongMemEvalBenchmark
from memrank.core import AdapterResponse, Document
from memrank.metrics.retrieval import ndcg_any_at_k, recall_all_at_k, retrieval_metrics
from tests.benchmarks.test_longmemeval_methodology import _item, _write_fixture


def test_recall_all_is_all_not_fractional():
    """Two of three gold documents inside k scores ZERO. This is the whole metric."""
    assert recall_all_at_k({"a": 0, "b": 1}, n_gold=3, k=10) == 0.0
    assert recall_all_at_k({"a": 0, "b": 1, "c": 2}, n_gold=3, k=10) == 1.0


def test_recall_all_respects_the_cutoff():
    """A gold document matched at rank 12 is a miss at k=10 and a hit at k=20."""
    ranks = {"a": 0, "b": 12}
    assert recall_all_at_k(ranks, n_gold=2, k=10) == 0.0
    assert recall_all_at_k(ranks, n_gold=2, k=20) == 1.0


def test_ndcg_rewards_ranking_evidence_high():
    perfect = ndcg_any_at_k({"a": 0, "b": 1}, n_gold=2, k=10)
    buried = ndcg_any_at_k({"a": 8, "b": 9}, n_gold=2, k=10)

    assert perfect == 1.0
    assert 0.0 < buried < perfect


def test_ndcg_and_recall_disagree_by_design():
    """Both gold inside k, but ranked last: recall_all is 1.0 while nDCG is not."""
    ranks = {"a": 8, "b": 9}
    assert recall_all_at_k(ranks, n_gold=2, k=10) == 1.0
    assert ndcg_any_at_k(ranks, n_gold=2, k=10) < 1.0


def _unit_with(retrieved_texts, tmp_path, monkeypatch, qid="q1", qtype="multi-session"):
    _write_fixture(tmp_path, monkeypatch, [_item(qid, qtype, n_sessions=4)])
    unit = LongMemEvalBenchmark().load()[0]
    docs = [Document(id=f"engine-{i}", content=t) for i, t in enumerate(retrieved_texts)]
    return unit, [AdapterResponse(query_id=qid, documents=docs)]


def test_gold_comes_from_answer_session_ids(tmp_path, monkeypatch):
    """The fixture marks session 0 as evidence via BOTH labellings, so the handle must match."""
    _write_fixture(tmp_path, monkeypatch, [_item("q1", "multi-session", n_sessions=3)])
    unit = LongMemEvalBenchmark().load()[0]

    assert unit.queries[0]["evidence_doc_ids"] == ["q1_s000"]


def test_a_retrieved_session_is_matched_by_content(tmp_path, monkeypatch):
    _write_fixture(tmp_path, monkeypatch, [_item("q1", "multi-session", n_sessions=3)])
    unit = LongMemEvalBenchmark().load()[0]
    gold_doc = next(d for d in unit.documents if d.id == "q1_s000")

    responses = [AdapterResponse(query_id="q1",
                                 documents=[Document(id="opaque", content=gold_doc.content)])]
    out = retrieval_metrics(unit, responses)

    assert out["recall_all@5"] == 1.0
    assert out["n_retrieval_scoreable"] == 1


def test_retrieving_nothing_scores_zero_not_none(tmp_path, monkeypatch):
    _write_fixture(tmp_path, monkeypatch, [_item("q1", "multi-session", n_sessions=3)])
    unit = LongMemEvalBenchmark().load()[0]

    out = retrieval_metrics(unit, [AdapterResponse(query_id="q1", documents=[])])

    assert out["recall_all@5"] == 0.0
    assert out["n_retrieval_scoreable"] == 1


def test_abstention_items_are_excluded_from_retrieval_only(tmp_path, monkeypatch):
    _write_fixture(tmp_path, monkeypatch, [_item("q1_abs", "multi-session", n_sessions=3)])
    unit = LongMemEvalBenchmark().load()[0]

    out = retrieval_metrics(unit, [AdapterResponse(query_id="q1_abs", documents=[])])

    assert out["n_retrieval_scoreable"] == 0
    assert out["recall_all@5"] is None      # not measured, which is not zero
    # ...but the question is still judged. The QA denominator is untouched.
    assert LongMemEvalBenchmark().judge_shape().is_judgeable(unit.queries[0])


def test_assistant_side_evidence_is_excluded_from_retrieval(tmp_path, monkeypatch):
    """51 questions on the canonical file, all single-session-assistant: the paper indexes
    user-side utterances only, so there is nothing for retrieval to target."""
    item = _item("q1", "single-session-assistant", n_sessions=2)
    for turn in item["haystack_sessions"][0]:
        turn["has_answer"] = turn["role"] == "assistant"

    _write_fixture(tmp_path, monkeypatch, [item])
    unit = LongMemEvalBenchmark().load()[0]

    assert unit.queries[0]["retrieval_scoreable"] is False
    assert LongMemEvalBenchmark().judge_shape().is_judgeable(unit.queries[0])


def test_score_reports_retrieval_beside_the_counts_and_never_as_a_composite(tmp_path,
                                                                           monkeypatch):
    """Publishing a retrieval proxy as a quality score is the field's commonest category error."""
    _write_fixture(tmp_path, monkeypatch, [_item("q1", "multi-session", n_sessions=3)])
    bench = LongMemEvalBenchmark()
    unit = bench.load()[0]

    out = bench.score(unit, [AdapterResponse(query_id="q1", documents=[])])

    assert out["composite"] is None
    assert bench.composite_rankable is False
    assert "recall_all@10" in out["retrieval"]
