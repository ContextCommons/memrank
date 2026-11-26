"""A run whose agent reports token usage: recorded through the real loop and shown, labelled."""

from __future__ import annotations

import pytest

from memrank.judging.providers.usage import record
from memrank.loop import report_html
from memrank.loop.report import result_facts
from memrank.loop.run import run_agent
from memrank.reference.full_context import with_usage
from memrank.service.protocol import TokenUsage
from memrank.service.supplied import SuppliedPrice
from tests.loop.test_run import LOCOMO, full_context, org, quiet, reader  # noqa: F401
from tests.service.conftest import engine  # noqa: F401 - pytest fixture

CONTEXT_TOKENS = 1500


def billed_reader(provider: str, model: str):
    """The reader, billed as a provider's response would say it was."""

    def complete(_model: str, system: str, user: str) -> str:
        record(provider, model, CONTEXT_TOKENS, 3)
        return reader(_model, system, user)

    return complete


def _run(engine, tmp_path, org, complete):  # noqa: F811
    return run_agent(engine, full_context(complete), LOCOMO, tmp_path, org.hosted(),
                     quiet([])).result


def test_the_reference_agent_reports_its_reader_and_a_free_feed(engine, tmp_path,  # noqa: F811
                                                                 org):  # noqa: F811
    result = _run(engine, tmp_path, org, billed_reader("anthropic", "claude-haiku-4-5"))
    usage = result.usage
    assert usage is not None and usage.context_tokens is not None
    assert usage.context_tokens.mean == CONTEXT_TOKENS
    assert usage.cost_per_million_queries_usd == pytest.approx(CONTEXT_TOKENS + 15)
    assert usage.ingest_cost_per_conversation_usd == 0.0  # history is stored, never billed
    assert all(case.feed_usage == [] for case in result.cases)
    facts = dict(result_facts(result, None))
    assert "self-reported by the agent" in facts["Context tokens per question"]
    assert facts["Cost per million queries"] == "$1,515.00"
    page = report_html.render(result)
    assert "Tokens and cost" in page and "reader 1,500 in / 3 out" in page


def test_an_unpriced_reader_keeps_its_tokens_and_names_why_there_is_no_cost(
        engine, tmp_path, org):  # noqa: F811
    result = _run(engine, tmp_path, org, billed_reader("stub", "claude-haiku-4-5"))
    assert result.usage is not None and result.usage.context_tokens is not None
    assert "stub/claude-haiku-4-5" in (result.usage.cost_refused or "")
    facts = dict(result_facts(result, None))
    assert facts["Cost"].startswith("not computed: no price in table")
    assert "`prices:` in the agent file" in facts["Cost"]
    assert "Cost per million queries" not in facts


def test_the_agent_files_prices_price_what_the_table_lacks(engine, tmp_path,  # noqa: F811
                                                           org):  # noqa: F811
    agent = full_context(billed_reader("stub", "claude-haiku-4-5"))
    agent.prices = [SuppliedPrice(provider="stub", model="claude-haiku-4-5",
                                  input_per_mtok=2.0, output_per_mtok=10.0)]
    result = run_agent(engine, agent, LOCOMO, tmp_path, org.hosted(), quiet([])).result
    assert result.usage is not None and result.usage.cost_refused is None
    assert result.usage.cost_per_million_queries_usd == pytest.approx(CONTEXT_TOKENS * 2 + 30)
    facts = dict(result_facts(result, None))
    assert "user-supplied prices for stub/claude-haiku-4-5 ($2 in / $10 out" in facts["Basis"]


def test_a_reader_that_billed_nothing_has_no_context_tokens(engine, tmp_path,  # noqa: F811
                                                            org):  # noqa: F811
    result = _run(engine, tmp_path, org, reader)  # no provider call, so nothing billed
    assert result.usage is not None and result.usage.answers_reported == 3
    assert result.usage.context_tokens is None
    assert result.usage.cost_per_million_queries_usd == 0.0


def test_with_usage_adds_nothing_when_there_is_nothing_to_report():
    assert with_usage({"a": 1}, None) == {"a": 1}
    entry = TokenUsage(component="reader", provider="p", model="m", input_tokens=1)
    assert with_usage({}, [entry])["memrank_usage"][0]["input_tokens"] == 1
