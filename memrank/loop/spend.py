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
"""A run's self-reported tokens and cost in plain words: what the ending, ``runs show`` and
``report.html`` say about :attr:`~memrank.service.protocol.RunResult.usage`.

Nothing here adds up or prices anything -- :mod:`memrank.service.usage` did that once -- and
every figure is named as the agent's report, because memrank neither counts nor checks it.
"""

from __future__ import annotations

from memrank.service.protocol import ComponentUsage, TokenUsage, UsageSummary

CONTEXT_TOKENS = "Context tokens per question"
COST_PER_MILLION = "Cost per million queries"
INGEST_COST = "Ingest cost per conversation"
#: Whose numbers these are, said wherever they are shown.
SELF_REPORTED = "self-reported by the agent"
#: What a cost is, said wherever one is shown.
COST_BASIS = ("reported tokens x the provider's list price per token, from price table {version}; "
              "cached input is priced as ordinary input")


def _dollars(value: float) -> str:
    """``$1,234.56``; under a cent, enough digits to be read: ``$0.00042``."""
    if value == 0 or value >= 0.01:
        return f"${value:,.2f}"
    return f"${value:.2g}" if value >= 1e-6 else f"${value:.1e}"


def _count(n: int, of: int, what: str) -> str:
    return "" if n == of else f" for {n} of {of} {what}"


def context_words(usage: UsageSummary) -> str | None:
    """``1,834 mean, 2,410 slow (1 in 20), ...``; None when no answer reported a reader."""
    stat = usage.context_tokens
    if stat is None:
        return None
    return (f"{stat.mean:,.0f} mean, {stat.p95:,.0f} slow (1 in 20), reader input tokens, "
            f"{SELF_REPORTED}{_count(stat.n, usage.answers, 'questions')}")


def cost_lines(usage: UsageSummary) -> list[tuple[str, str]]:
    """The two costs, each with how many steps it is over; the refusal when there is none."""
    if usage.cost_refused is not None:
        return [("Cost", f"not computed: {usage.cost_refused}")]
    rows = []
    if usage.cost_per_million_queries_usd is not None:
        rows.append((COST_PER_MILLION, f"{_dollars(usage.cost_per_million_queries_usd)}"
                     f"{_count(usage.answers_reported, usage.answers, 'questions')}"))
    if usage.ingest_cost_per_conversation_usd is not None:
        rows.append((INGEST_COST, f"{_dollars(usage.ingest_cost_per_conversation_usd)}"
                     f"{_count(usage.conversations_reported, usage.conversations, 'conversations')}"))
    return rows


def basis(usage: UsageSummary) -> str:
    return f"{SELF_REPORTED}; {COST_BASIS.format(version=usage.price_table_version)}"


def component_words(component: ComponentUsage) -> str:
    """``ask reader anthropic/claude-haiku-4-5: 18,340 in, 120 out, $0.019``."""
    out = "" if component.output_tokens is None else f", {component.output_tokens:,} out"
    cost = "" if component.cost_usd is None else f", {_dollars(component.cost_usd)}"
    return (f"{component.step} {component.component} {component.provider}/{component.model}: "
            f"{component.input_tokens:,} in{out}{cost}")


def usage_lines(usage: UsageSummary | None) -> list[tuple[str, str]]:
    """Every row a run's usage gives, label then words; none when the agent reported none."""
    if usage is None:
        return []
    rows: list[tuple[str, str]] = []
    context = context_words(usage)
    if context is not None:
        rows.append((CONTEXT_TOKENS, context))
    rows.extend(cost_lines(usage))
    rows.extend(("Tokens" if index == 0 else "", component_words(component))
                for index, component in enumerate(usage.components))
    rows.append(("Basis", basis(usage)))
    return rows


def entry_words(usage: list[TokenUsage] | None) -> str | None:
    """One step's reported usage, short: ``reader 1,834 in / 12 out``; None when not reported."""
    if usage is None:
        return None
    if not usage:
        return "no paid calls"
    return "; ".join(f"{e.component} {e.input_tokens:,} in"
                     + ("" if e.output_tokens is None else f" / {e.output_tokens:,} out")
                     for e in usage)
