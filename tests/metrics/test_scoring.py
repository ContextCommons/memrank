from memrank.core import Document
from memrank.metrics.scoring import EvidenceSpec, normalize, score_query


def _doc(doc_id, content, source_doc_id=None):
    meta = {"doc_id": source_doc_id} if source_doc_id else {}
    return Document(id=doc_id, content=content, metadata=meta)


def test_normalize_folds_case_space_punct():
    assert normalize("  Café,  is\tOPEN! ") == "cafe is open"


def test_hit_on_content_even_with_engine_uuid_id():
    spec = EvidenceSpec(required_spans=["blue whale"])
    docs = [_doc("a1b2-uuid", "Her favorite animal is the Blue Whale.")]
    r = score_query(spec, docs)
    assert r.hit is True
    assert r.matched_span == "blue whale"
    assert r.matched_doc_id == "a1b2-uuid"


def test_non_empty_irrelevant_retrieval_is_not_a_hit():
    spec = EvidenceSpec(required_spans=["blue whale"])
    docs = [_doc("x", "We talked about the weather and lunch.")]
    assert score_query(spec, docs).hit is False


def test_all_required_spans_must_be_in_one_doc():
    spec = EvidenceSpec(required_spans=["paris", "april"])
    split = [_doc("a", "We met in Paris."), _doc("b", "It was April.")]
    assert score_query(spec, split).hit is False
    together = [_doc("c", "We met in Paris in April.")]
    assert score_query(spec, together).hit is True


def test_forbidden_span_voids_the_hit():
    spec = EvidenceSpec(required_spans=["dog"], forbidden_spans=["cat"])
    docs = [_doc("a", "She has a dog and a cat.")]
    assert score_query(spec, docs).hit is False


def test_evidence_doc_ids_gate_the_match():
    spec = EvidenceSpec(required_spans=["sushi"], evidence_doc_ids=["sess_3"])
    wrong = [_doc("e", "loves sushi", source_doc_id="sess_1")]
    assert score_query(spec, wrong).hit is False
    right = [_doc("e", "loves sushi", source_doc_id="sess_3")]
    assert score_query(spec, right).hit is True


def test_evidence_gate_is_lenient_when_doc_has_no_doc_id():
    spec = EvidenceSpec(required_spans=["sushi"], evidence_doc_ids=["sess_3"])
    # doc carries NO doc_id (e.g. a fact-extracting engine) -> content match still counts
    docs = [_doc("engine-uuid", "loves sushi")]
    assert score_query(spec, docs).hit is True


def test_negative_query_is_correct_when_forbidden_absent():
    spec = EvidenceSpec(kind="negative", forbidden_spans=["allergic to peanuts"])
    clean = [_doc("a", "enjoys hiking on weekends")]
    assert score_query(spec, clean).hit is True
    leaked = [_doc("b", "is allergic to peanuts")]
    assert score_query(spec, leaked).hit is False
