"""Okapi BM25, pinned against arithmetic done by hand in this file.

Same three-document fixture as `test_tfidf_adapter.py`, so the two methods are compared on
identical tokenisation and the only difference between the numbers is the weighting. BM25's
length normalisation puts the shorter document first here too, but for its own reason and at
its own value -- `b` is what governs it, and a run that moves `b` moves these numbers.
"""

import pytest

from memrank.adapters.bm25 import B_DEFAULT, BM25, K1_DEFAULT
from memrank.core import Document

#: Three documents, 7 / 6 / 6 tokens long, so avgdl = 19 / 3 = 6.3333333.
CORPUS = [
    Document(id="d1", content="The blue whale is the largest animal", metadata={"doc_id": "d1"}),
    Document(id="d2", content="The weather was rainy all week", metadata={"doc_id": "d2"}),
    Document(id="d3", content="A whale watching trip last week", metadata={"doc_id": "d3"}),
]

# The arithmetic, with N = 3, k1 = 1.5, b = 0.75 and avgdl = 6.3333333:
#
#   df(whale) = 2, so idf = ln(1 + (3 - 2 + 0.5) / (2 + 0.5)) = ln(1.6) = 0.4700036
#   f(whale, d) = 1 in both d1 and d3, so the numerator is 1 * (k1 + 1) = 2.5 for each.
#
#   d1, |d| = 7:
#     k1 * (1 - b + b * 7 / 6.3333333) = 1.5 * (0.25 + 0.75 * 1.1052632)
#                                      = 1.5 * 1.0789474 = 1.6184211
#     score = 0.4700036 * 2.5 / (1 + 1.6184211) = 0.4700036 * 0.9547739 = 0.448747
#
#   d3, |d| = 6:
#     k1 * (1 - b + b * 6 / 6.3333333) = 1.5 * (0.25 + 0.75 * 0.9473684)
#                                      = 1.5 * 0.9605263 = 1.4407895
#     score = 0.4700036 * 2.5 / (1 + 1.4407895) = 0.4700036 * 1.0242537 = 0.481405
#
#   d2 carries no query term -> 0.0, and a zero score is never returned.
TOLERANCE = 1e-4
D3_SCORE = 0.481405
D1_SCORE = 0.448747


def _scored(query: str, k: int = 5, **options):
    system = BM25(**options)
    system.prepare("u1")
    system.ingest(CORPUS)
    recall = system.retrieve(query, k, "u1")
    system.cleanup()
    return recall


def test_bm25_ranks_the_shorter_document_first_and_scores_it_by_hand():
    """The ranking and both scores, against the arithmetic in the comment above."""
    recall = _scored("whale")

    assert [d.id for d in recall.documents] == ["d3", "d1"]
    assert recall.declared["results"] == ["d3", "d1"]
    assert recall.declared["scores"][0] == pytest.approx(D3_SCORE, abs=TOLERANCE)
    assert recall.declared["scores"][1] == pytest.approx(D1_SCORE, abs=TOLERANCE)


def test_the_defaults_are_the_conventional_ones():
    """1.5 and 0.75 are what "Okapi BM25" names when nobody states otherwise."""
    system = BM25()

    assert (system.k1, system.b) == (K1_DEFAULT, B_DEFAULT) == (1.5, 0.75)


def test_b_zero_turns_length_normalisation_off_and_ties_the_two_documents():
    """With b = 0 the length term vanishes, so the same single occurrence scores the same.

    score = idf * 1 * (k1 + 1) / (1 + k1) = idf = ln(1.6) = 0.4700036 for both, and the tie
    falls back to document order.
    """
    recall = _scored("whale", b=0.0)

    assert [d.id for d in recall.documents] == ["d1", "d3"]
    assert recall.declared["scores"] == [pytest.approx(0.4700036, abs=TOLERANCE)] * 2


def test_bm25_returns_nothing_when_nothing_scores_above_zero():
    """"Searched, found none" is a finding; padding the list to k would hide it."""
    assert _scored("helicopter").documents == []


def test_bm25_never_pads_to_k():
    assert len(_scored("whale", k=5).documents) == 2


def test_bm25_cuts_at_k():
    assert [d.id for d in _scored("whale", k=1).documents] == ["d3"]


def test_bm25_isolates_units():
    system = BM25()
    system.prepare("u1")
    system.ingest([Document(id="d1", content="alpha token")])
    system.cleanup()
    system.prepare("u2")
    ranked = system.retrieve("alpha", 5, "u2").documents
    system.cleanup()

    assert ranked == []  # u1 content is not visible under u2


def test_bm25_emits_required_metric_keys():
    system = BM25()

    assert {"retrieve_p50_ms"}.issubset(system.latency_metrics())
    assert "tokens_per_query_mean" in system.token_metrics()


def test_an_all_empty_corpus_scores_zero_rather_than_dividing_by_it():
    """avgdl is zero when every stored document is empty, and the length term needs it."""
    system = BM25()
    system.prepare("u1")
    system.ingest([Document(id="d1", content=""), Document(id="d2", content="   ")])
    ranked = system.retrieve("whale", 5, "u1").documents
    system.cleanup()

    assert ranked == []
