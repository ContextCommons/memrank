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
"""A run's self-reported token usage, added up and priced.

The agent reports, per feed and per ask step, what its paid components used
(:class:`~memrank.service.protocol.TokenUsage`). This sums it over the run -- per answer, per
conversation and per component -- and prices it with the one table
(:mod:`memrank.metrics.prices`): cost is arithmetic on the reported tokens and nothing else.

A step that reported nothing is left out of every figure rather than counted as zero. A
provider/model the table does not price refuses the whole run's cost, naming it; the tokens are
kept and shown all the same.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from statistics import fmean
from typing import Literal

from memrank.metrics.prices import PRICE_TABLE_VERSION, Unpriced, cost_usd
from memrank.service.protocol import (
    READER_COMPONENT,
    CaseOutcome,
    ComponentUsage,
    TokenStat,
    TokenUsage,
    UsageSummary,
)

Step = Literal["feed", "ask"]
_PER_MILLION = 1_000_000


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def _context_tokens(asks: list[list[TokenUsage]]) -> TokenStat | None:
    """The reader's input tokens per answer, over the answers that reported a reader entry."""
    per_answer = [float(sum(e.input_tokens for e in usage if e.component == READER_COMPONENT))
                  for usage in asks if any(e.component == READER_COMPONENT for e in usage)]
    if not per_answer:
        return None
    return TokenStat(mean=fmean(per_answer), p50=_percentile(per_answer, 0.5),
                     p95=_percentile(per_answer, 0.95), n=len(per_answer))


def _step_cost(usage: list[TokenUsage]) -> float:
    return sum(cost_usd(e.provider, e.model, e.input_tokens, e.output_tokens) for e in usage)


def _components(steps: Iterable[tuple[Step, list[TokenUsage]]],
                priced: bool) -> list[ComponentUsage]:
    """Each (step, component, provider, model)'s tokens summed, with its cost when ``priced``."""
    inputs: dict[tuple[Step, str, str, str], int] = defaultdict(int)
    outputs: dict[tuple[Step, str, str, str], int | None] = {}
    costs: dict[tuple[Step, str, str, str], float] = defaultdict(float)
    for step, usage in steps:
        for entry in usage:
            key = (step, entry.component, entry.provider, entry.model)
            inputs[key] += entry.input_tokens
            if entry.output_tokens is not None:
                outputs[key] = (outputs.get(key) or 0) + entry.output_tokens
            else:
                outputs.setdefault(key, None)
            if priced:
                costs[key] += cost_usd(entry.provider, entry.model, entry.input_tokens,
                                       entry.output_tokens)
    return [ComponentUsage(step=key[0], component=key[1], provider=key[2], model=key[3],
                           input_tokens=inputs[key], output_tokens=outputs[key],
                           cost_usd=costs[key] if priced else None)
            for key in sorted(inputs)]


def _unpriced(entries: Iterable[TokenUsage]) -> list[str]:
    """Every reported provider/model (or over-long call) the table has no price for, once."""
    missing: dict[str, None] = {}
    for entry in entries:
        try:
            cost_usd(entry.provider, entry.model, entry.input_tokens, entry.output_tokens)
        except Unpriced as exc:
            missing[str(exc)] = None
    return list(missing)


def summarize_usage(cases: list[CaseOutcome]) -> UsageSummary | None:
    """The run's reported usage and cost; None when no step of it reported any.

    Over each question's recorded answer and each case's last feed, as the result records them.
    """
    answered = [q for case in cases for q in case.questions if q.answer is not None]
    asks = [q.usage for q in answered if q.usage is not None]
    feeds = [case.feed_usage for case in cases if case.feed_usage is not None]
    if not asks and not feeds:
        return None
    steps: list[tuple[Step, list[TokenUsage]]] = [("feed", u) for u in feeds] + \
        [("ask", u) for u in asks]
    missing = _unpriced(e for _, usage in steps for e in usage)
    summary = UsageSummary(
        answers=len(answered), answers_reported=len(asks), conversations=len(cases),
        conversations_reported=len(feeds), context_tokens=_context_tokens(asks),
        components=_components(steps, priced=not missing),
        price_table_version=PRICE_TABLE_VERSION)
    if missing:
        summary.cost_refused = (f"no price in table {PRICE_TABLE_VERSION} for "
                                f"{', '.join(missing)}; its tokens are kept")
        return summary
    if asks:
        summary.cost_per_million_queries_usd = fmean(map(_step_cost, asks)) * _PER_MILLION
    if feeds:
        summary.ingest_cost_per_conversation_usd = fmean(map(_step_cost, feeds))
    return summary
