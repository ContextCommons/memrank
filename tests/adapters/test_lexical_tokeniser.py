"""One tokeniser for both lexical methods, so the weighting is the only thing between them.

Two arms that split words differently measure their tokenisers as much as their scoring, and
neither published method specifies one. This pins the rule -- lowercase, then split on anything
that is not a letter or a digit -- and asserts both classes actually go through it.
"""

from memrank.adapters.bm25 import BM25
from memrank.adapters.lexical import tokenise
from memrank.adapters.tfidf import TFIDF
from memrank.core import Document


def test_the_tokeniser_lowercases_and_splits_on_non_alphanumerics():
    assert tokenise("Acme's plan -- Enterprise (2026)!") == [
        "acme", "s", "plan", "enterprise", "2026"]


def test_repeats_are_kept_because_both_methods_weight_term_frequency():
    """A set would turn each method into the unweighted overlap it exists to improve on."""
    assert tokenise("whale whale week") == ["whale", "whale", "week"]


def test_both_methods_match_a_term_across_case_and_punctuation():
    """The same question reaches the same document whichever of the two is ranking it."""
    corpus = [Document(id="d1", content="Acme's plan: ENTERPRISE."),
              Document(id="d2", content="unrelated filler text here")]

    for system in (TFIDF(), BM25()):
        system.prepare("u1")
        system.ingest(corpus)
        ranked = system.retrieve("enterprise", 5, "u1").documents
        system.cleanup()
        assert [d.id for d in ranked] == ["d1"], type(system).__name__
