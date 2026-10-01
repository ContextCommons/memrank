import pytest

from memrank.core import Document
from memrank.metrics import cost


def test_count_tokens_is_stable_and_pinned():
    assert cost.ENCODING_NAME == "o200k_base"
    assert cost.count_tokens("hello world") == 2


def test_context_tokens_truncates_to_budget():
    docs = [Document(id=str(i), content="word " * 100) for i in range(5)]
    full = cost.context_tokens(docs, token_budget=10_000)
    capped = cost.context_tokens(docs, token_budget=50)
    assert full > capped
    assert capped == 50


def test_an_uncapped_arm_reports_what_it_actually_carried():
    """The cap is the fairness control for matched targets, not a ceiling on what may be REPORTED.

    `hindsight` and `full-context` declare `context_budget: uncapped` and are handed their
    full context by `runner._context_text`. This function capped anyway, so the receipt recorded
    the budget for them no matter what they retrieved: a real faithful-hindsight LoCoMo run carried
    8,886 tokens per query and recorded 5,000. That erases the only number the faithful mode is
    for -- AMB's published LoCoMo row spends 36,235 tokens per question.
    """
    docs = [Document(id=str(i), content="word " * 100) for i in range(5)]
    uncapped = cost.context_tokens(docs, token_budget=50, budget_mode="uncapped")
    assert uncapped == cost.context_tokens(docs, token_budget=10_000)
    assert uncapped > 50


def test_a_matched_arm_still_stops_at_the_budget():
    """The regression guard on the fix: matched mode is the leaderboard's fairness control."""
    docs = [Document(id=str(i), content="word " * 100) for i in range(5)]
    assert cost.context_tokens(docs, token_budget=50, budget_mode="matched") == 50
    # Unstated means matched -- every real engine, and the default the runner falls back to.
    assert cost.context_tokens(docs, token_budget=50) == 50


def test_price_per_query_uses_input_rate():
    dollars = cost.price_per_query(1_000_000, model="gpt-4o-mini")
    assert dollars == pytest.approx(0.15)


def test_unknown_model_fails_loud():
    with pytest.raises(ValueError, match="Unknown model"):
        cost.price_per_query(100, model="totally-made-up")


def test_pricing_metadata_present():
    assert cost.PRICING_TABLE_VERSION
    assert cost.PRICING_EFFECTIVE_DATE


def test_truncate_to_tokens_caps_length():
    long = "word " * 1000
    out = cost.truncate_to_tokens(long, 10)
    assert cost.count_tokens(out) == 10


def test_truncate_to_tokens_passthrough_when_short():
    assert cost.truncate_to_tokens("hello world", 100) == "hello world"


def test_preload_tokenizer_is_idempotent_and_readies_counting():
    cost.preload_tokenizer()
    cost.preload_tokenizer()
    assert cost.count_tokens("hello world") == 2
