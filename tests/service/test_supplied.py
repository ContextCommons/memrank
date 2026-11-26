"""Prices a user supplies for a run: one accepted shape, held at the agent file and the service."""

from __future__ import annotations

import pytest
import yaml
from pydantic import ValidationError

from memrank.connect.base import ConnectorConfigError
from memrank.connect.spec import load_agent
from memrank.loop.spend import basis
from memrank.metrics.prices import PRICE_TABLE_VERSION
from memrank.service.protocol import AgentRef, RunCreate, UsageSummary
from memrank.service.supplied import AppliedPrice, SuppliedPrice, supplied_table

SPEC = "name: a\nconnector: command\nfeed: [echo]\nask: [echo]\n"
GOOD = {"provider": "my-router", "model": "bedrock/claude-haiku-4-5",
        "input_per_mtok": 1.1, "output_per_mtok": 5.5}


@pytest.mark.parametrize("change, accepted", [
    ({}, True),
    ({"input_per_mtok": 0, "output_per_mtok": 0}, True),  # an explicit zero is a price
    ({"input_per_mtok": 3}, True),  # a whole number is a number
    ({"provider": ""}, False),
    ({"model": ""}, False),
    ({"input_per_mtok": -0.5}, False),
    ({"output_per_mtok": float("inf")}, False),
    ({"input_per_mtok": float("nan")}, False),
    ({"input_per_mtok": "1.1"}, False),
    ({"output_per_mtok": True}, False),
    ({"cached_per_mtok": 0.1}, False),
])
def test_one_shape_is_accepted_by_the_service_and_the_agent_file(tmp_path, change, accepted):
    given = {**GOOD, **change}
    request = {"evaluation": "locomo", "agent": {"name": "a"}, "prices": [given]}
    rates = {k: v for k, v in given.items() if k not in ("provider", "model")}
    path = tmp_path / "agent.yaml"
    path.write_text(SPEC + yaml.safe_dump({"prices": {given["provider"]: {given["model"]: rates}}}))
    if accepted:
        assert RunCreate.model_validate(request).prices == [SuppliedPrice(**given)]
        assert load_agent(path).prices == [SuppliedPrice(**given)]
    else:
        with pytest.raises(ValidationError):
            RunCreate.model_validate(request)
        with pytest.raises(ConnectorConfigError):
            load_agent(path)


def test_the_agent_file_gives_its_prices_to_the_run(tmp_path):
    path = tmp_path / "agent.yaml"
    path.write_text(SPEC + "prices:\n  my-router:\n    bedrock/claude-haiku-4-5:\n"
                           "      {input_per_mtok: 1.1, output_per_mtok: 5.5}\n")
    assert load_agent(path).prices == [SuppliedPrice(**GOOD)]


@pytest.mark.parametrize("prices", ["1.1", "{my-router: 1.1}", "{my-router: {m: 1.1}}"])
def test_a_prices_section_of_another_shape_is_refused(tmp_path, prices):
    path = tmp_path / "agent.yaml"
    path.write_text(SPEC + f"prices: {prices}\n")
    with pytest.raises(ConnectorConfigError, match="maps each provider to its models"):
        load_agent(path)


def test_the_cost_refuses_a_price_built_around_the_parser():
    smuggled = SuppliedPrice.model_construct(**{**GOOD, "input_per_mtok": -1.0})
    with pytest.raises(ValueError, match="finite number"):
        supplied_table([smuggled])


def test_an_older_request_without_prices_still_reads():
    request = RunCreate(evaluation="locomo", agent=AgentRef(name="a"))
    assert RunCreate.model_validate_json(request.model_dump_json(exclude={"prices"})).prices == []


def test_the_basis_names_each_supplied_price_and_what_it_replaced():
    usage = UsageSummary(
        answers=1, answers_reported=1, conversations=1, conversations_reported=0,
        price_table_version=PRICE_TABLE_VERSION, supplied_prices=[
            AppliedPrice(**GOOD, replaces_table=False),
            AppliedPrice(provider="anthropic", model="claude-haiku-4-5", input_per_mtok=2,
                         output_per_mtok=10, replaces_table=True)])
    words = basis(usage)
    assert f"list price from price table {PRICE_TABLE_VERSION}, and user-supplied" in words
    assert "my-router/bedrock/claude-haiku-4-5 ($1.1 in / $5.5 out per million tokens)" in words
    assert "anthropic/claude-haiku-4-5 ($2 in / $10 out per million tokens, in place of the " \
           "table's)" in words
    table_only = usage.model_copy(update={"supplied_prices": []})
    assert "user-supplied" not in basis(table_only)
