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
"""LongMemEval reports counts, not a quality score -- its metric is the judge's.

Two proxies were published from `score` and both were constants rather than measurements. The
first asked whether a gold DOCUMENT ID appeared among the retrieved ids; an engine returns its
own ids, so the comparison was `e47becba_answer_280352e9` against `0226ceb6-999e-...` and every run
by every engine scored exactly 0.0. The second asked whether the reference answer appeared as a
verbatim span -- which is the exact matching LongMemEval's paper rejects ("correct answers can
take flexible forms"), and which scored every abstention item 0 for every engine, an abstention
having no answer text to find. The published metric is a judged QA accuracy.
"""
from __future__ import annotations

from memrank.benchmarks.longmemeval import LongMemEvalBenchmark
from memrank.core import AdapterResponse, BenchmarkUnit, Document

#: The shape that produced the first constant: our id namespace left, the engine's right.
#: Our side is an OPAQUE POSITIONAL HANDLE since 2026-08-14. It used to be
#: "e47becba_answer_280352e9" -- the dataset's own session id, which prefixes every evidence
#: session with `answer` and so told any engine that indexed ids exactly where the answer was
#: (audit F1). That the old constant read naturally here is a fair measure of how invisible the
#: leak was: it sat in our own test suite, in a file about scoring, for months.
GOLD_DOC_ID = "e47becba_s004"
ENGINE_MEMORY_ID = "0226ceb6-999e-37e3-7734-000000000000"


def _unit(category: str = "single-session-user") -> BenchmarkUnit:
    return BenchmarkUnit(
        unit_id="e47becba", isolation_id="e47becba", documents=[],
        queries=[{"id": "e47becba", "text": "what did I study?", "user_id": "e47becba",
                  "category": category, "gold_ids": [GOLD_DOC_ID],
                  "gold_answers": ["Business Administration"]}],
    )


def _responses(content: str) -> list[AdapterResponse]:
    return [AdapterResponse(query_id="e47becba",
                            documents=[Document(id=ENGINE_MEMORY_ID, content=content)])]


def test_longmemeval_publishes_no_composite_however_retrieval_went():
    """Neither the right answer nor the wrong one produces a number here."""
    right = LongMemEvalBenchmark().score(_unit(), _responses("They studied Business Administration"))
    wrong = LongMemEvalBenchmark().score(_unit(), _responses("They enjoy hiking on weekends"))

    assert right["composite"] is None
    assert wrong["composite"] is None


def test_a_query_that_retrieved_nothing_is_counted():
    """Dropping it would shrink a denominator nobody reports -- how the last two proxies hid."""
    score = LongMemEvalBenchmark().score(_unit(), [])

    assert score["n_queries"] == 1
    assert score["n_retrieved_nothing"] == 1


def test_the_per_question_type_counts_stay_visible():
    score = LongMemEvalBenchmark().score(_unit("temporal-reasoning"), _responses("class of 2019"))

    assert score["per_category_counts"] == {"temporal-reasoning": 1}


def test_the_metric_names_the_judge():
    score = LongMemEvalBenchmark().score(_unit(), _responses("Business Administration"))

    assert "judge required" in score["metric"]


def test_every_longmemeval_question_type_is_judgeable():
    """LongMemEval was fully unjudgeable: 0 of 500 queries passed the judge's category gate.

    Its six `question_type` values come from the dataset file itself, and none was in
    JUDGE_VALID_CATEGORIES -- which was built for LoCoMo, demo and BEAM. Two failed on punctuation
    alone (`temporal-reasoning` vs BEAM's `temporal_reasoning`). Nothing detected it because
    judging LongMemEval had never been attempted; the retrieval metrics never read `category`.

    Read from the loader rather than hard-coded, so a dataset that gains a type fails here instead
    of silently dropping those questions out of judged coverage.
    """
    from memrank.benchmarks.longmemeval import LongMemEvalBenchmark

    bench = LongMemEvalBenchmark(slice="mini")
    shape = bench.judge_shape()
    queries = [q for unit in bench.load() for q in unit.queries]
    assert queries, "fixture slice produced no queries"

    unsupported = {q["category"] for q in queries if q.get("category") not in shape.categories}
    assert not unsupported, f"unjudgeable LongMemEval categories: {sorted(unsupported)}"
    assert all(shape.is_judgeable(q) for q in queries)


def test_longmemeval_categories_are_not_folded_into_beams():
    """`temporal-reasoning` (LongMemEval) and `temporal_reasoning` (BEAM) are different
    benchmarks' categories. Per-benchmark declarations keep them apart structurally: the
    hyphenated literals are LongMemEval's, and BEAM's underscored names must NOT appear in its
    set -- pooling them is what the old global allowlist did."""
    from memrank.benchmarks.longmemeval import LongMemEvalBenchmark

    categories = LongMemEvalBenchmark().judge_shape().categories
    for hyphenated, underscored in (("temporal-reasoning", "temporal_reasoning"),
                                    ("knowledge-update", "knowledge_update")):
        assert hyphenated in categories
        assert underscored not in categories
