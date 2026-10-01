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
"""Prompt-cost estimation for the demo.

Counts the tokens an engine injects into context per query (pinned tiktoken
encoding) and prices them at a model's input rate. This is PROMPT cost only --
it excludes ingest-time extraction and answer generation, which the MVP does
not measure. The column is labeled accordingly in reports.
"""

from __future__ import annotations

from functools import lru_cache

import tiktoken

from memrank.core import Document

ENCODING_NAME = "o200k_base"
PRICING_TABLE_VERSION = "2026-06-10.1"  # added Anthropic claude-sonnet-4-6/opus-4-8/haiku-4-5
PRICING_EFFECTIVE_DATE = "2026-06-02"

# USD per 1M tokens: (input, output). Input rate is what the demo uses.
_PRICES: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    # Anthropic models (rates as of 2026-06-10)
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-opus-4-8": (15.00, 75.00),
    "claude-haiku-4-5": (1.00, 5.00),
}


@lru_cache(maxsize=1)
def _encoding() -> tiktoken.Encoding:
    try:
        return tiktoken.get_encoding(ENCODING_NAME)
    except Exception as exc:  # pragma: no cover - environment failure
        from memrank.provenance.install import dependency_instruction

        raise RuntimeError(
            f"tiktoken encoding {ENCODING_NAME!r} unavailable. Repair this install with: "
            f"{dependency_instruction('tiktoken')}"
        ) from exc


def preload_tokenizer() -> None:
    """Eagerly load the pinned tokenizer (first call may download ~4 MB)."""
    _encoding()


def _encode(text: str) -> list[int]:
    """Encode ``text`` with the pinned encoding.

    ``disallowed_special=()`` treats any literal special-token strings that
    appear verbatim in dataset content (e.g. a ``<|endoftext|>`` inside a
    LongMemEval turn) as ordinary text rather than raising ``ValueError``.
    All content encoding funnels through here so the policy is applied once.
    """
    return _encoding().encode(text, disallowed_special=())


def count_tokens(text: str) -> int:
    """Return the token count of ``text`` under the pinned encoding."""
    if not text:
        return 0
    return len(_encode(text))


def truncate_to_tokens(text: str, budget: int) -> str:
    """Return ``text`` truncated to at most ``budget`` tokens (pinned encoding)."""
    if not text:
        return ""
    tokens = _encode(text)
    if len(tokens) <= budget:
        return text
    return _encoding().decode(tokens[:budget])


def context_tokens(docs: list[Document], token_budget: int,
                   budget_mode: str = "matched") -> int:
    """Tokens injected per query, truncated to ``token_budget`` in rank order.

    ``budget_mode`` is the target's ``context_budget`` and must be the SAME answer
    ``runner._context_text`` gives: an "uncapped" arm is not truncated, so reporting the cap for
    it would describe a run that did not happen.

    That divergence was real. This function capped unconditionally while the reader was correctly
    handed the full context, so faithful `hindsight` recorded ``context_tokens_mean: 5000`` for queries
    that actually carried 8,886 tokens -- and no uncapped row could ever report a spend above the
    matched budget. The comparison faithful mode exists for is exactly that number: AMB's published
    LoCoMo row spends 36,235 tokens per question.
    """
    total = 0
    for doc in docs:
        total += count_tokens(doc.content or "")
        if budget_mode != "uncapped" and total >= token_budget:
            return token_budget
    return total


def context_truncated(docs: list[Document], token_budget: int,
                      budget_mode: str = "matched") -> bool:
    """Whether the cap dropped retrieved text the engine had already found.

    Deliberately answerable WITHOUT a judge: `context_tokens` above already walks the same
    documents, and "did the budget bite?" is a question about retrieval and the cap, not about
    grading. Requiring a judged run to ask it made it cost a full grading pass.

    An uncapped arm is never truncated, which is the same answer `runner._context_text` gives.
    """
    if budget_mode == "uncapped":
        return False
    return sum(count_tokens(doc.content or "") for doc in docs) > token_budget


def priced_models() -> list[str]:
    """Every model this build can price a run against, in stable order.

    Exposed because a picker that offers an unpriced model is a 422 waiting to happen: the
    planner refuses one with ``unknown_pricing_model``, and a browser has no other way to know
    what the table holds. The table is versioned (:data:`PRICING_TABLE_VERSION`), so this answer
    changes with the deployment rather than with whoever last edited a UI constant.
    """
    return sorted(_PRICES)


def input_price_per_mtok(model: str) -> float:
    """Input $/1M tokens for ``model``; fail loud on unknown model."""
    if model not in _PRICES:
        raise ValueError(f"Unknown model {model!r}. Known: {sorted(_PRICES)}")
    return _PRICES[model][0]


def output_price_per_mtok(model: str) -> float:
    """Output $/1M tokens for ``model``; fail loud on unknown model.

    The table has always carried output rates; this exposes them so a caller
    that models generated tokens (the judge/reader path) prices them from the
    one versioned table rather than keeping a second copy of the rates.
    """
    if model not in _PRICES:
        raise ValueError(f"Unknown model {model!r}. Known: {sorted(_PRICES)}")
    return _PRICES[model][1]


def price_per_query(ctx_tokens: int, model: str) -> float:
    """Estimated prompt $/query = context tokens x input price."""
    return ctx_tokens * input_price_per_mtok(model) / 1_000_000.0
