"""TF-IDF cosine ranking, pinned against arithmetic done by hand in this file.

A retrieval baseline whose numbers nobody has checked is a black box with a famous name on it.
The three-document fixture below is small enough that every weight can be written out, so a
change to the weighting fails here with the changed value rather than passing because the
ranking happened to survive it.

The fixture is also chosen so cosine and the bare dot product DISAGREE: the query has one term,
so both documents carrying it have the same dot product and only the norms separate them. An
implementation that dropped the normalisation would return `d1` first and still look plausible.
"""

import pytest

from memrank.adapters.tfidf import TFIDF
from memrank.core import Document

#: Three documents, 7 / 6 / 6 tokens long. `whale` is in two of them, `week` in two others.
CORPUS = [
    Document(id="d1", content="The blue whale is the largest animal", metadata={"doc_id": "d1"}),
    Document(id="d2", content="The weather was rainy all week", metadata={"doc_id": "d2"}),
    Document(id="d3", content="A whale watching trip last week", metadata={"doc_id": "d3"}),
]

# The arithmetic, with N = 3 documents and idf(t) = ln(N / df(t)):
#
#   df(whale) = 2 (d1, d3)   -> idf = ln(3/2)  = 0.4054651
#   df(the)   = 2 (d1, d2)   -> idf = ln(3/2)  = 0.4054651
#   df(week)  = 2 (d2, d3)   -> idf = ln(3/2)  = 0.4054651
#   every other term has df = 1 -> idf = ln(3) = 1.0986123
#
# The query is "whale", so its vector is the single weight 1 * 0.4054651, and its norm is the
# same number. The dot product with either document carrying `whale` is 0.4054651^2 = 0.1644020.
#
#   d1 = the(2) blue whale is largest animal
#     ||d1||^2 = (2*0.4054651)^2 + 4*(1.0986123)^2 + (0.4054651)^2
#              = 0.6576078 + 4.8277958 + 0.1644020 = 5.6498056 -> ||d1|| = 2.3769320
#     cosine   = 0.1644020 / (2.3769320 * 0.4054651) = 0.1644020 / 0.9637630 = 0.170583
#
#   d3 = a whale watching trip last week
#     ||d3||^2 = 4*(1.0986123)^2 + 2*(0.4054651)^2
#              = 4.8277958 + 0.3288039 = 5.1565997 -> ||d3|| = 2.2708148
#     cosine   = 0.1644020 / (2.2708148 * 0.4054651) = 0.1644020 / 0.9207361 = 0.178555
#
#   d2 shares no term with the query -> 0.0, and a zero score is never returned.
#
# d3 outranks d1 although both mention the whale exactly once: d3 is shorter, so the same
# weight is a larger share of its length. That is what the cosine buys over the dot product.
TOLERANCE = 1e-4
D3_COSINE = 0.178555
D1_COSINE = 0.170583


def _scored(query: str, k: int = 5):
    system = TFIDF()
    system.prepare("u1")
    system.ingest(CORPUS)
    recall = system.retrieve(query, k, "u1")
    system.cleanup()
    return recall


def test_tfidf_ranks_the_shorter_document_first_and_scores_it_by_hand():
    """The ranking and the top score, both against the arithmetic in the comment above."""
    recall = _scored("whale")

    assert [d.id for d in recall.documents] == ["d3", "d1"]
    assert recall.declared["results"] == ["d3", "d1"]
    assert recall.declared["scores"][0] == pytest.approx(D3_COSINE, abs=TOLERANCE)
    assert recall.declared["scores"][1] == pytest.approx(D1_COSINE, abs=TOLERANCE)


def test_tfidf_returns_nothing_when_nothing_scores_above_zero():
    """"Searched, found none" is a finding; padding the list to k would hide it."""
    assert _scored("helicopter").documents == []


def test_tfidf_never_pads_to_k():
    """Two documents score; asking for five returns two."""
    assert len(_scored("whale", k=5).documents) == 2


def test_tfidf_cuts_at_k():
    assert [d.id for d in _scored("whale", k=1).documents] == ["d3"]


def test_tfidf_breaks_ties_by_document_order():
    """Two identical documents cannot be separated, so the earlier one goes first.

    A third document is present because `alpha` has to be rare enough to weigh anything: with
    only the pair, df equals N and ln(N / df) is zero -- which is the next test.
    """
    system = TFIDF()
    system.prepare("u1")
    system.ingest([Document(id="first", content="alpha beta"),
                   Document(id="second", content="alpha beta"),
                   Document(id="third", content="gamma delta")])
    ranked = system.retrieve("alpha", 3, "u1").documents
    system.cleanup()

    assert [d.id for d in ranked] == ["first", "second"]


def test_a_term_in_every_document_weighs_nothing():
    """ln(N / df) is zero when df is N. It is the whole point of the IDF half, not an edge case.

    A question made only of such terms distinguishes nothing, and TF-IDF says so by scoring
    every document zero rather than by returning an arbitrary order.
    """
    system = TFIDF()
    system.prepare("u1")
    system.ingest([Document(id="d1", content="alpha beta"),
                   Document(id="d2", content="alpha gamma")])
    ranked = system.retrieve("alpha", 2, "u1").documents
    system.cleanup()

    assert ranked == []


def test_tfidf_isolates_units():
    system = TFIDF()
    system.prepare("u1")
    system.ingest([Document(id="d1", content="alpha token")])
    system.cleanup()
    system.prepare("u2")
    ranked = system.retrieve("alpha", 5, "u2").documents
    system.cleanup()

    assert ranked == []  # u1 content is not visible under u2


def test_tfidf_emits_required_metric_keys():
    system = TFIDF()

    assert {"retrieve_p50_ms"}.issubset(system.latency_metrics())
    assert "tokens_per_query_mean" in system.token_metrics()
