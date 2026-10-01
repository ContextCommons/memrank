"""The installed MCP server exposes structured tools over the real protocol."""

from __future__ import annotations

import sys

import pytest

# Guards the extra the way tests/tracking/* guard theirs. Without this the module raised at
# COLLECTION on a `uv sync --extra dev` install, taking the whole suite down rather than skipping
# one file -- so a developer following the documented install could not run the tests at all.
pytest.importorskip("mcp.client")  # requires: uv sync --extra mcp

from mcp.client import Client, ClientSession  # noqa: E402  (imported after the extra gate)
from mcp.client.stdio import StdioServerParameters, stdio_client

from memrank.mcp_server import create_server


@pytest.mark.anyio
async def test_discovery_and_planning_over_in_memory_protocol(monkeypatch):
    monkeypatch.setattr("memrank.application.planning.config.secret", lambda name: "set")
    async with Client(create_server()) as client:
        listing = await client.list_tools()
        names = {tool.name for tool in listing.tools}
        result = await client.call_tool("plan_sweep", {"request": {
            "targets": ["word-overlap"], "evals": ["demo"],
            "axes": {"seeds": [1, 2]},
        }})

    assert {"plan_sweep", "submit_experiment", "get_run_result", "cancel_run"} <= names
    assert result.structured_content["outcome"] == "ok"
    assert len(result.structured_content["data"]["experiments"]) == 2


@pytest.mark.anyio
async def test_tool_annotations_distinguish_read_and_write_operations():
    async with Client(create_server()) as client:
        tools = {tool.name: tool for tool in (await client.list_tools()).tools}

    assert tools["plan_sweep"].annotations.read_only_hint is True
    assert tools["submit_experiment"].annotations.destructive_hint is True
    assert tools["submit_experiment"].annotations.idempotent_hint is False
    assert tools["cancel_run"].annotations.idempotent_hint is True


@pytest.mark.anyio
async def test_real_stdio_entrypoint_has_clean_protocol_stdout():
    params = StdioServerParameters(command=sys.executable,
                                   args=["-m", "memrank.mcp_server"])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()

    assert any(tool.name == "plan_sweep" for tool in tools.tools)
