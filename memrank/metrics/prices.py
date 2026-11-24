# Copyright 2026 AtomicStrata
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.
"""memrank's one versioned price table: US dollars per million tokens, by provider and model.

Every price memrank puts on tokens comes from here -- an agent run's self-reported usage
(:mod:`memrank.service.usage`) and the older in-process estimates (:mod:`memrank.metrics.cost`,
which reads the ``anthropic`` and ``openai`` rows by bare model name). The version is recorded with
every cost computed from it, so a figure says which prices produced it.

Each row cites the provider's own published page and the date it was read. Rates are the
standard, non-batch, uncached input and output rates: cached input is priced as ordinary input.
A model missing from the table has no price -- never zero. A local or self-hosted model gets an
explicit $0 row so its tokens stay visible. Nothing heavy is imported, so the service can price
a run without the tokenizer :mod:`memrank.metrics.cost` loads.
"""

from __future__ import annotations

from dataclasses import dataclass

from memrank.errors import MemrankError

PRICE_TABLE_VERSION = "2026-11-19.1"
#: The date the newest rows were read from their sources.
PRICE_TABLE_EFFECTIVE_DATE = "2026-11-19"


@dataclass(frozen=True)
class Price:
    """One model's rates, in USD per 1M tokens, and where they were read."""

    input_per_mtok: float
    #: 0.0 for an embedding model, which bills input only.
    output_per_mtok: float
    source: str
    checked: str
    #: Above this many input tokens in one call the provider charges another rate this table
    #: does not hold; such a call has no price here. None: one rate at every length.
    max_input_tokens: int | None = None


_OPENAI = "https://developers.openai.com/api/docs/pricing"
_ANTHROPIC = "https://platform.claude.com/docs/en/about-claude/pricing"
#: AWS's Price List API offer for Bedrock foundation models in us-east-1 ("Claude Haiku 4.5
#: (Amazon Bedrock Edition)", On-demand Inference). The pricing page renders its rows in script.
_BEDROCK = ("https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/"
            "AmazonBedrockFoundationModels/current/us-east-1/index.json")
_CHECKED = "2026-11-19"
#: OpenAI's gpt-5.4 rates hold below 272K input tokens; above it the rate changes.
_GPT_5_4_SHORT_CONTEXT = 272_000
#: Bedrock's "Regional" on-demand rate, which a geographic (``us.``) inference profile bills at:
#: Anthropic's page says regional endpoints carry a 10% premium over global, and AWS's offer
#: lists only Regional and Global on-demand tiers. Read 2026-11-19.
_HAIKU_4_5_BEDROCK_REGIONAL = Price(1.10, 5.50, _BEDROCK, _CHECKED)

#: (provider, model) -> price. The provider is who the agent says billed it.
PRICES: dict[tuple[str, str], Price] = {
    ("openai", "gpt-5.4"): Price(2.50, 15.00, _OPENAI, _CHECKED, _GPT_5_4_SHORT_CONTEXT),
    ("openai", "gpt-5.4-pro"): Price(30.00, 180.00, _OPENAI, _CHECKED, _GPT_5_4_SHORT_CONTEXT),
    ("openai", "gpt-5.4-mini"): Price(0.75, 4.50, _OPENAI, _CHECKED),
    ("openai", "gpt-5.4-nano"): Price(0.20, 1.25, _OPENAI, _CHECKED),
    ("openai", "text-embedding-3-large"): Price(0.13, 0.0, _OPENAI, _CHECKED),
    ("openai", "text-embedding-3-small"): Price(0.02, 0.0, _OPENAI, _CHECKED),
    ("openai", "gpt-4o"): Price(2.50, 10.00, _OPENAI, _CHECKED),
    ("openai", "gpt-4o-mini"): Price(0.15, 0.60, _OPENAI, _CHECKED),
    ("openai", "gpt-4.1-mini"): Price(0.40, 1.60, _OPENAI, _CHECKED),
    ("anthropic", "claude-haiku-4-5"): Price(1.00, 5.00, _ANTHROPIC, _CHECKED),
    ("anthropic", "claude-sonnet-4-6"): Price(3.00, 15.00, _ANTHROPIC, _CHECKED),
    ("anthropic", "claude-opus-4-8"): Price(5.00, 25.00, _ANTHROPIC, _CHECKED),
    ("bedrock", "us.anthropic.claude-haiku-4-5-20251001-v1:0"): _HAIKU_4_5_BEDROCK_REGIONAL,
    ("bedrock", "global.anthropic.claude-haiku-4-5-20251001-v1:0"):
        Price(1.00, 5.00, _BEDROCK, _CHECKED),
    # Atomic's router passes calls through to the provider serving the model; it adds no charge
    # of its own. bedrock/claude-haiku-4-5 is served by the us. profile (leaderboard router.py).
    ("atomic-router", "bedrock/claude-haiku-4-5"): _HAIKU_4_5_BEDROCK_REGIONAL,
    # The router's self-hosted embedder (mem0:leaderboard): no provider bills it.
    ("atomic-router", "nomic-embed-text"): Price(0.0, 0.0, "self-hosted by Atomic's router",
                                                 _CHECKED),
}


class Unpriced(MemrankError):
    """A reported provider/model, or a call's length, this table has no price for."""


def price_of(provider: str, model: str) -> Price:
    """The table's price for ``model`` billed by ``provider``; :class:`Unpriced` when absent."""
    try:
        return PRICES[(provider, model)]
    except KeyError:
        raise Unpriced(f"{provider}/{model}") from None


def cost_usd(provider: str, model: str, input_tokens: int, output_tokens: int | None) -> float:
    """What ``input_tokens`` in and ``output_tokens`` out cost at the table's price.

    Raises :class:`Unpriced` for a model the table does not hold, or a call past the length its
    rate holds for.
    """
    price = price_of(provider, model)
    if price.max_input_tokens is not None and input_tokens > price.max_input_tokens:
        raise Unpriced(f"{provider}/{model} over {price.max_input_tokens} input tokens")
    spent = input_tokens * price.input_per_mtok + (output_tokens or 0) * price.output_per_mtok
    return spent / 1_000_000.0
