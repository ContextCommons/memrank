"""Unit tests for the native adapter's handshake and its refusal to guess.

The native adapter is the only one that talks to code memrank did not write, so what it REFUSES
matters as much as what it accepts: a translator that under-reports its configuration, or opts out
of the shared token budget, would produce a run whose receipt is internally consistent and
externally false. Wire-level mapping is pinned in test_native_adapter_wire.py.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from memrank.adapters.native import CONTRACT_VERSION, ContractError, Native


def describe_body(**overrides: Any) -> dict[str, Any]:
    """A conforming describe payload, with fields swapped in per test."""
    body = {
        "contract_version": CONTRACT_VERSION,
        "adapter": {"name": "stub-translator", "version": "0.1.0"},
        "engine": {"name": "stub-engine", "version": "9.9.9"},
        "components": {"llm": {"provider": "anthropic", "model": "claude"}, "embedder": None},
        "capabilities": {"graph_snapshot": True, "context_budget": "matched"},
    }
    body.update(overrides)
    return body


def wire(adapter: Native, handler) -> Native:
    """Point an adapter at an in-memory transport instead of a real translator."""
    adapter._client = httpx.Client(
        base_url="http://translator.test", transport=httpx.MockTransport(handler))
    return adapter


def describing(body: dict[str, Any], *, status: int = 200):
    """A handler answering describe with ``body`` and every POST with an empty object."""
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/describe"):
            return httpx.Response(status, json=body)
        return httpx.Response(200, json={})
    return handler


def adapter_for(body: dict[str, Any]) -> Native:
    return wire(Native(base_url="http://translator.test"), describing(body))


# ----------------------------------------------------------------------- #
# Identity
# ----------------------------------------------------------------------- #

def test_declares_translator_transport():
    """Latency compares only within a transport class, so this label is load-bearing."""
    assert Native.name == "native"
    assert Native.transport == "translator"


def test_base_url_comes_from_the_placement(monkeypatch):
    monkeypatch.delenv("NATIVE_API_URL", raising=False)
    assert Native().base_url == "http://localhost:8099"
    monkeypatch.setenv("NATIVE_API_URL", "http://127.0.0.1:9001/")
    assert Native().base_url == "http://127.0.0.1:9001"


# ----------------------------------------------------------------------- #
# The handshake
# ----------------------------------------------------------------------- #

def test_describe_populates_identity_and_capabilities():
    adapter = adapter_for(describe_body())
    assert adapter.describe_engine() == {
        "llm": {"provider": "anthropic", "model": "claude"}, "embedder": {}}
    assert adapter.engine_version == "9.9.9"
    assert adapter.graph_capable is True


def test_describe_is_fetched_once():
    """Identity cannot change mid-run without invalidating every measurement before it did."""
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(200, json=describe_body())

    adapter = wire(Native(base_url="http://translator.test"), handler)
    adapter.describe_engine()
    adapter.describe_engine()
    assert calls == ["/memrank/v1/describe"]


def test_unknown_contract_version_is_refused():
    adapter = adapter_for(describe_body(contract_version="v2"))
    with pytest.raises(ContractError, match="contract_version"):
        adapter.describe_engine()


@pytest.mark.parametrize("section", ["adapter", "engine", "components", "capabilities"])
def test_missing_describe_section_is_refused(section: str):
    body = describe_body()
    del body[section]
    with pytest.raises(ContractError, match=section):
        adapter_for(body).describe_engine()


@pytest.mark.parametrize("role", ["llm", "embedder"])
def test_unstated_component_is_refused(role: str):
    """Silence is not the same claim as null: a blank knob is how an engine measures a
    configuration nobody chose."""
    body = describe_body()
    del body["components"][role]
    with pytest.raises(ContractError, match=f"components.{role}"):
        adapter_for(body).describe_engine()


def test_explicit_null_component_is_accepted():
    """A null says 'this engine has no such part' -- a positive statement, unlike absence."""
    body = describe_body(components={"llm": None, "embedder": None})
    assert adapter_for(body).describe_engine() == {"llm": {}, "embedder": {}}


@pytest.mark.parametrize("budget", ["uncapped", "none"])
def test_translator_may_not_opt_out_of_the_token_budget(budget: str):
    """The shared --token-budget is the fairness control; only memrank's own control arms differ."""
    body = describe_body(capabilities={"graph_snapshot": False, "context_budget": budget})
    with pytest.raises(ContractError, match="context_budget"):
        adapter_for(body).describe_engine()


def test_missing_capability_is_refused():
    body = describe_body(capabilities={"context_budget": "matched"})
    with pytest.raises(ContractError, match="graph_snapshot"):
        adapter_for(body).describe_engine()


def test_describe_failure_surfaces_the_translators_message():
    body = {"error": "engine not started"}
    adapter = wire(Native(base_url="http://translator.test"), describing(body, status=503))
    with pytest.raises(ContractError, match="engine not started"):
        adapter.describe_engine()


# ----------------------------------------------------------------------- #
# Reporting
# ----------------------------------------------------------------------- #

def test_components_are_reported_by_the_engine_not_read_from_env(monkeypatch):
    """The inversion that makes native work: memrank cannot configure a stranger's engine, so the
    translator states what it is running and memrank records that."""
    monkeypatch.setenv("NATIVE_LLM_PROVIDER", "wrong-from-env")
    adapter = adapter_for(describe_body())
    adapter.describe_engine()
    assert adapter.effective_config()["llm"] == {"provider": "anthropic", "model": "claude"}


def test_effective_config_never_raises_before_describe():
    """It runs while a run record is written, long after a network failure could be reported."""
    adapter = Native(base_url="http://127.0.0.1:1")
    assert adapter.effective_config()["llm"] == {"provider": None, "model": None}


def test_cleanup_before_prepare_is_a_no_op():
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(200, json={})

    wire(Native(base_url="http://translator.test"), handler).cleanup()
    assert calls == []
