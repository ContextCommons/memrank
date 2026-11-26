"""The one price table: only publicly disclosed prices, arithmetic only, no model priced by
default, and a supplied price winning over the table's."""

from __future__ import annotations

import pytest

from memrank.metrics import cost
from memrank.metrics.prices import (
    PRICE_TABLE_VERSION,
    PRICES,
    Price,
    Unpriced,
    cost_usd,
    price_of,
    supplied_price,
)


@pytest.mark.parametrize("key", sorted(PRICES))
def test_every_row_is_a_price_its_provider_publishes(key):
    price = PRICES[key]
    assert price.source.startswith("https://"), key  # the provider's own published page
    assert price.checked, key
    assert price.input_per_mtok > 0, key  # a disclosed price, never a placeholder zero


def test_cost_is_tokens_times_the_listed_rates():
    assert cost_usd("anthropic", "claude-haiku-4-5", 1_000_000, 1_000_000) == pytest.approx(6.0)
    assert cost_usd("openai", "text-embedding-3-small", 1_000_000, None) == pytest.approx(0.02)


def test_a_supplied_price_wins_over_the_table_and_prices_what_it_lacks():
    supplied = {("anthropic", "claude-haiku-4-5"): supplied_price(2.0, 10.0),
                ("my-router", "embedder"): supplied_price(0.0, 0.0)}
    assert cost_usd("anthropic", "claude-haiku-4-5", 1_000_000, 0, supplied) == 2.0
    assert cost_usd("my-router", "embedder", 10_000, None, supplied) == 0.0  # given, not assumed
    assert price_of("openai", "gpt-4o", supplied) is PRICES[("openai", "gpt-4o")]


def test_a_supplied_price_holds_past_a_tiered_length():
    supplied = {("openai", "gpt-5.4"): supplied_price(5.0, 22.5)}
    assert cost_usd("openai", "gpt-5.4", 1_000_000, 0, supplied) == pytest.approx(5.0)


@pytest.mark.parametrize("rate", [-0.01, float("inf"), float("nan"), True, "1.0", None])
def test_a_price_is_a_finite_number_of_at_least_zero(rate):
    with pytest.raises(ValueError, match="finite number"):
        Price(rate, 1.0, "s", "c")


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
