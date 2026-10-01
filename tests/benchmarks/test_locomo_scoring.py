"""LoCoMo reports counts, not a quality score -- its metric is the judge's.

These tests used to pin a substring proxy: does the reference answer appear verbatim inside a
retrieved document? LoCoMo's published protocol is token-F1 or an LLM-judge verdict on the
GENERATED answer, and its derived answers ("3 times", "2022") cannot appear as spans at any
retrieval quality, so the proxy ranked engines by how literally they stored transcripts. It is
gone; what remains here is the guarantee that nothing quality-shaped is published without a judge.
"""
from memrank.benchmarks.locomo import LoCoMoBenchmark
from memrank.core import AdapterResponse, BenchmarkUnit, Document


def _unit():
    return BenchmarkUnit(
        unit_id="s1", isolation_id="s1", documents=[],
        queries=[{"id": "s1_q0", "text": "where?", "user_id": "s1",
                  "category": "single-hop", "gold_answers": ["paris"],
                  "evidence_doc_ids": ["s1_session_1"]}],
    )


def _response(content: str) -> list[AdapterResponse]:
    return [AdapterResponse(query_id="s1_q0",
                            documents=[Document(id="engine-uuid-xyz", content=content,
                                                metadata={"doc_id": "s1_session_1"})])]


def test_locomo_publishes_no_composite_however_retrieval_went():
    """Perfect retrieval and useless retrieval get the same non-answer.

    This module does not grade quality, so it must not appear to -- a number that varies with
    retrieval is one a reader will treat as a score.
    """
    assert LoCoMoBenchmark().score(_unit(), _response("we met in Paris"))["composite"] is None
    assert LoCoMoBenchmark().score(_unit(), _response("we ate lunch"))["composite"] is None


def test_locomo_names_the_judge_as_its_metric():
    assert "judge required" in LoCoMoBenchmark().score(_unit(), [])["metric"]


def test_the_question_counts_stay_visible():
    score = LoCoMoBenchmark().score(_unit(), [])

    assert score["n_queries"] == 1
    assert score["per_category_counts"] == {"single-hop": 1}


def test_a_query_that_retrieved_nothing_is_counted():
    """A shrinking denominator nobody reports is how the previous broken metric went unnoticed."""
    assert LoCoMoBenchmark().score(_unit(), [])["n_retrieved_nothing"] == 1
    assert LoCoMoBenchmark().score(
        _unit(), _response("we met in Paris"))["n_retrieved_nothing"] == 0


def test_load_keeps_numeric_gold_answers_and_emits_no_span(tmp_path, monkeypatch):
    """Gold answers feed the judge, so numeric ones must survive as strings.

    ``required_spans`` is absent: the reference answer is what a generated answer is graded
    against, never a string retrieval is required to surface.
    """
    import json

    fixture = [{
        "sample_id": "s1",
        "conversation": {
            "speaker_a": "A", "speaker_b": "B",
            "session_1": [{"dia_id": "D1:1", "speaker": "A", "text": "we met twice"}],
            "session_1_date_time": "1:00 pm on 8 May, 2023",
        },
        "qa": [{"category": 1, "question": "how many times?", "answer": 5, "evidence": ["D1:1"]}],
    }]
    path = tmp_path / "locomo.json"
    path.write_text(json.dumps(fixture), encoding="utf-8")
    monkeypatch.setenv("LOCOMO_DATA_PATH", str(path))

    query = LoCoMoBenchmark().load()[0].queries[0]

    assert query["gold_answers"] == ["5"]
    assert "required_spans" not in query
    assert query["evidence_doc_ids"] == ["s1_session_1"]


def test_score_reports_evidence_recall_but_never_a_composite():
    """The official recall_acc rides beside the counts as a retrieval diagnostic; the quality
    verdict stays the judge's (composite None regardless)."""
    score = LoCoMoBenchmark().score(_unit(), _response("we met in Paris"))

    assert score["composite"] is None
    assert score["evidence_recall"] == 0.0  # no ingested session text to match against
    assert score["n_queries_with_evidence"] == 1
    assert score["evidence_matcher_version"] == 1
