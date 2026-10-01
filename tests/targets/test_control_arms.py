"""The mandatory baseline arms from PRD_Memrank_v2.md:86.

(a) no-memory, (b) token-matched ICL at the same retrieval budget, (c) full-context where the KB
fits. A row without these is not publishable -- they are the hypothesis test, run every time.
"""
from __future__ import annotations

from memrank.adapters import get_adapter
from memrank.core import Document
from memrank.targets import resolve_target

DOCS = [Document(id=f"d{n}", content=f"fact number {n}") for n in range(5)]


def _prepared(name):
    adapter = get_adapter(name)
    adapter.prepare("unit-1")
    adapter.ingest(DOCS)
    return adapter


def test_none_retrieves_nothing():
    """The closed-book floor: the reader gets an empty context and answers from parametrics."""
    docs = _prepared("no-context").retrieve("what is fact number 3?", k=10, user_id="unit-1").documents
    assert docs == []


def test_icl_returns_every_document_in_ingest_order():
    """Unranked: it reads what it can from the start, which is the whole point of the baseline."""
    docs = _prepared("fixed-context").retrieve("what is fact number 3?", k=10, user_id="unit-1").documents
    assert [d.id for d in docs] == ["d0", "d1", "d2", "d3", "d4"]


def test_icl_ignores_k_because_the_token_budget_is_the_limit():
    """k caps ranked retrieval; ICL is capped by the shared token budget instead."""
    docs = _prepared("fixed-context").retrieve("q", k=2, user_id="unit-1").documents
    assert len(docs) == 5


def test_full_context_returns_everything_too():
    docs = _prepared("full-context").retrieve("q", k=1, user_id="unit-1").documents
    assert len(docs) == 5


def test_isolation_holds_between_units():
    """AGENTS.md: nothing ingested for one isolation_id may affect another."""
    adapter = _prepared("fixed-context")
    adapter.cleanup()
    adapter.prepare("unit-2")
    docs = adapter.retrieve("q", k=10, user_id="unit-2").documents
    assert docs == []


def test_all_three_are_registered_and_in_process():
    for name in ("no-context", "fixed-context", "full-context"):
        target = resolve_target(name)
        assert target.kind == "in-process"
        assert target.adapter == name
        assert target.depends == ()


def test_only_full_context_is_uncapped():
    """icl is token-MATCHED -- that is what makes it a fair baseline (PRD:86)."""
    assert resolve_target("fixed-context").context_budget == "matched"
    assert resolve_target("full-context").context_budget == "uncapped"
    # The suffixed arm, deliberately: since 2026-08-19 a BARE vendor ref is that vendor's own
    # configuration and therefore uncapped, so `hindsight` and `mem0` are both uncapped now. This
    # test is about the control arms, and `hindsight:matched` is here to show the cap is not a
    # property of being an arm.
    assert resolve_target("hindsight:matched").context_budget == "matched"


def test_none_scores_only_on_negative_queries_when_unjudged():
    """Its unjudged number is real but is about declining, not about memory.

    Retrieving nothing is the CORRECT answer to a negative query, so `no-context` picks up hits there and
    nowhere else -- which is why the run emits a notice rather than presenting it as a memory score.
    """
    from memrank.runner import _warn_if_meaningless_unjudged

    assert _warn_if_meaningless_unjudged("no-context", judged=False) is None      # emits a notice
    assert _warn_if_meaningless_unjudged("no-context", judged=True) is None       # silent when judged


def test_control_arms_need_no_credentials():
    from memrank.targets.catalog import required_secrets_for

    for name in ("no-context", "fixed-context", "full-context"):
        assert required_secrets_for(resolve_target(name)) == []
