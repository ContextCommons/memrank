"""The one price table: every row cited, arithmetic only, and no model priced by default."""

from __future__ import annotations

import pytest

from memrank.metrics import cost
from memrank.metrics.prices import PRICE_TABLE_VERSION, PRICES, Unpriced, cost_usd


def test_every_row_cites_its_source_and_when_it_was_read():
    for key, price in PRICES.items():
        assert price.source and price.checked, key
        assert price.input_per_mtok >= 0 and price.output_per_mtok >= 0, key


def test_cost_is_tokens_times_the_listed_rates():
    assert cost_usd("anthropic", "claude-haiku-4-5", 1_000_000, 1_000_000) == pytest.approx(6.0)
    assert cost_usd("openai", "text-embedding-3-small", 1_000_000, None) == pytest.approx(0.02)


def test_a_self_hosted_model_costs_an_explicit_zero():
    assert cost_usd("atomic-router", "nomic-embed-text", 10_000, None) == 0.0


@pytest.mark.parametrize("provider, model", [("anthropic", "nope"), ("bedrock", "gpt-5.4")])
def test_a_model_not_in_the_table_has_no_price(provider, model):
    with pytest.raises(Unpriced, match=f"{provider}/{model}"):
        cost_usd(provider, model, 1, 1)


def test_a_call_past_the_length_its_rate_holds_for_has_no_price():
    assert cost_usd("openai", "gpt-5.4", 272_000, 0) == pytest.approx(0.68)
    with pytest.raises(Unpriced, match="over 272000"):
        cost_usd("openai", "gpt-5.4", 272_001, 0)


def test_the_older_estimates_read_the_same_table():
    assert cost.PRICING_TABLE_VERSION == PRICE_TABLE_VERSION
    assert cost.input_price_per_mtok("claude-haiku-4-5") == 1.00
    assert "bedrock/claude-haiku-4-5" not in cost.priced_models()
