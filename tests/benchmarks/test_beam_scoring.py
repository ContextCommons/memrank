"""BEAM reports counts, not a quality score -- its metric is the rubric judge's.

These tests used to pin a substring proxy the module simultaneously declared structurally
invalid (`substring_recall_supported = False`): BEAM's golds are prose `ideal_response`, not
verbatim spans. Computing it anyway distorted the data it touched -- abstention probes had to be
dropped from its denominator, because a negative carrying no forbidden spans scores an
unconditional hit. See `judge_shape()` for how BEAM is actually graded.
"""
from memrank.benchmarks.beam import BEAMBenchmark
from memrank.core import AdapterResponse, BenchmarkUnit, Document


def _unit():
    return BenchmarkUnit(
        unit_id="conv1", isolation_id="conv1", documents=[],
        queries=[{"id": "conv1_information_extraction_0",
                  "text": "favorite drink?", "user_id": "conv1",
                  "category": "information_extraction",
                  "gold_answers": ["green tea"], "evidence_doc_ids": ["conv1"]},
                 {"id": "conv1_abstention_0", "text": "what is my cat's name?",
                  "user_id": "conv1", "category": "abstention", "kind": "negative",
                  "gold_answers": [""], "evidence_doc_ids": ["conv1"]}],
    )


def _response(content: str) -> list[AdapterResponse]:
    return [AdapterResponse(query_id="conv1_information_extraction_0",
                            documents=[Document(id="m1", content=content,
                                                metadata={"doc_id": "conv1"})])]


def test_beam_publishes_no_composite_however_retrieval_went():
    assert BEAMBenchmark(tier="100k").score(
        _unit(), _response("loves green tea daily"))["composite"] is None
    assert BEAMBenchmark(tier="100k").score(
        _unit(), _response("we discussed the rain"))["composite"] is None


def test_beam_names_the_rubric_judge_as_its_metric():
    assert "judge required" in BEAMBenchmark(tier="100k").score(_unit(), [])["metric"]


def test_abstention_is_counted_rather_than_excluded():
    """The proxy had to drop abstention from its denominator to avoid awarding it a free 1.0.

    With no proxy there is nothing to distort, so the ability is simply part of the question
    counts like any other -- and the count stays reported so its size is never a surprise.
    """
    score = BEAMBenchmark(tier="100k").score(_unit(), [])

    assert score["n_queries"] == 2
    assert score["n_abstention"] == 1
    assert score["per_ability_counts"] == {"information_extraction": 1, "abstention": 1}


def test_a_query_that_retrieved_nothing_is_counted():
    score = BEAMBenchmark(tier="100k").score(_unit(), _response("loves green tea daily"))

    assert score["n_retrieved_nothing"] == 1  # the abstention probe, which got no response
