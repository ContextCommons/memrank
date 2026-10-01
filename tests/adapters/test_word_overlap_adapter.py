from memrank.adapters.word_overlap import WordOverlap
from memrank.core import Document


def test_word_overlap_retrieves_a_word_sharing_doc():
    a = WordOverlap()
    a.prepare("u1")
    a.ingest([
        Document(id="d1", content="favorite animal is the blue whale", metadata={"doc_id": "d1"}),
        Document(id="d2", content="the weather was rainy all week", metadata={"doc_id": "d2"}),
    ])
    docs = a.retrieve("what animal does she like", 1, "u1").documents
    a.cleanup()
    assert docs[0].content == "favorite animal is the blue whale"
    assert docs[0].metadata["doc_id"] == "d1"


def test_baseline_isolates_runs():
    a = WordOverlap()
    a.prepare("u1")
    a.ingest([Document(id="d1", content="alpha token", metadata={"doc_id": "d1"})])
    a.cleanup()
    a.prepare("u2")
    docs = a.retrieve("alpha", 5, "u2").documents
    a.cleanup()
    assert docs == []  # u1 content not visible under u2


def test_baseline_emits_required_metric_keys():
    a = WordOverlap()
    assert {"retrieve_p50_ms"}.issubset(a.latency_metrics())
    assert "tokens_per_query_mean" in a.token_metrics()
