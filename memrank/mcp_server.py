"""Deprecated: the local stdio MCP server over the `submit` operations. Evaluate an agent with
`memrank run` instead; see the README.

Local stdio MCP presentation over Memrank application operations."""

from __future__ import annotations

from typing import Any

from memrank.application import catalogs
from memrank.application.planning import plan_sweep as build_plan
from memrank.application.runs import (
    cancel_run as cancel,
)
from memrank.application.runs import (
    get_run as inspect_run,
)
from memrank.application.runs import (
    get_run_logs as logs,
)
from memrank.application.runs import (
    get_run_result as result,
)
from memrank.application.runs import (
    list_run_artifacts as artifacts,
)
from memrank.application.runs import (
    list_runs as runs,
)
from memrank.application.runs import (
    read_run_artifact as read_artifact,
)
from memrank.application.submission import submit_experiment as submit_one
from memrank.application.types import Notice, SweepPlan, SweepRequest, ToolEnvelope
from memrank.errors import MemrankError


def _dump(value: ToolEnvelope) -> dict[str, Any]:
    return value.model_dump(mode="json")


def _catalog_call(operation) -> dict[str, Any]:
    try:
        return _dump(ToolEnvelope(data=operation()))
    except (MemrankError, ValueError) as exc:
        refusal = Notice(code="invalid_catalog_reference", message=str(exc))
        return _dump(ToolEnvelope(outcome="refused", refusals=[refusal]))


def create_server():
    """Build the MCP server lazily so ordinary CLI imports do not require the SDK."""
    from memrank.errors import optional_import

    # Both modules come from the same extra, so the first missing one names it. Imported through
    # the helper rather than directly so a missing SDK reads as an install step rather than as an
    # internal bug (memrank/errors.py:optional_import).
    MCPServer = optional_import("mcp.server", "mcp").MCPServer
    ToolAnnotations = optional_import("mcp_types", "mcp").ToolAnnotations

    server = MCPServer("memrank")
    read_only = ToolAnnotations(readOnlyHint=True, destructiveHint=False,
                                idempotentHint=True, openWorldHint=False)
    consequential = ToolAnnotations(readOnlyHint=False, destructiveHint=True,
                                    idempotentHint=False, openWorldHint=True)
    control = ToolAnnotations(readOnlyHint=False, destructiveHint=True,
                              idempotentHint=True, openWorldHint=True)

    @server.tool(annotations=read_only)
    def list_targets() -> dict[str, Any]:
        """List target references available to this Memrank installation."""
        return _dump(ToolEnvelope(data={"targets": catalogs.list_targets()}))

    @server.tool(annotations=read_only)
    def get_target(ref: str, overrides: list[str] | None = None) -> dict[str, Any]:
        """Inspect one resolved target and its explicit components."""
        return _catalog_call(lambda: catalogs.get_target(ref, overrides))

    @server.tool(annotations=read_only)
    def list_evals() -> dict[str, Any]:
        """List evaluation goals available to this Memrank installation."""
        return _dump(ToolEnvelope(data={"evals": catalogs.list_evals()}))

    @server.tool(annotations=read_only)
    def get_eval(ref: str) -> dict[str, Any]:
        """Inspect one eval's declared slices, tiers, metrics, and constraints."""
        return _catalog_call(lambda: catalogs.get_eval(ref))

    @server.tool(annotations=read_only)
    def plan_sweep(request: SweepRequest) -> dict[str, Any]:
        """Preview a deterministic, non-executing experiment matrix."""
        plan = build_plan(request)
        if plan.refusals:
            envelope = ToolEnvelope(outcome="refused", data=plan,
                                    warnings=plan.warnings, refusals=plan.refusals)
        else:
            envelope = ToolEnvelope(data=plan, warnings=plan.warnings)
        return _dump(envelope)

    @server.tool(annotations=consequential)
    def submit_experiment(plan: SweepPlan, experiment_id: str,
                          acknowledgements: list[str] | None = None) -> dict[str, Any]:
        """Asynchronously launch exactly one cell from a revalidated plan."""
        return _dump(submit_one(plan, experiment_id, acknowledgements))

    @server.tool(annotations=read_only)
    def list_runs(limit: int = 20, org: str | None = None, mine: bool = True,
                  cursor: str | None = None) -> dict[str, Any]:
        """List bounded local and optionally org-scoped runs."""
        return _dump(runs(limit=limit, org=org, mine=mine, cursor=cursor))

    @server.tool(annotations=read_only)
    def get_run(run_id: str, org: str | None = None) -> dict[str, Any]:
        """Inspect one run's durable and live state."""
        return _dump(inspect_run(run_id, org=org))

    @server.tool(annotations=read_only)
    def get_run_logs(run_id: str, org: str | None = None, cursor: str | None = None,
                     limit: int = 200, container: str | None = None) -> dict[str, Any]:
        """Read a bounded page of local or cloud logs."""
        return _dump(logs(run_id, org=org, cursor=cursor, limit=limit, container=container))

    @server.tool(annotations=read_only)
    def get_run_result(run_id: str, org: str | None = None) -> dict[str, Any]:
        """Read bounded result metrics and reproducibility receipts."""
        return _dump(result(run_id, org=org))

    @server.tool(annotations=read_only)
    def list_run_artifacts(run_id: str, org: str | None = None) -> dict[str, Any]:
        """List artifact handles without embedding their content."""
        return _dump(artifacts(run_id, org=org))

    @server.tool(annotations=read_only)
    def read_run_artifact(run_id: str, name: str, org: str | None = None,
                          offset: int = 0, limit: int = 262_144) -> dict[str, Any]:
        """Read one bounded artifact range as text or base64."""
        return _dump(read_artifact(run_id, name, org=org, offset=offset, limit=limit))

    @server.tool(annotations=control)
    def cancel_run(run_id: str, org: str | None = None) -> dict[str, Any]:
        """Idempotently request cancellation while preserving the run record."""
        return _dump(cancel(run_id, org=org))

    return server


def main() -> None:
    """Run the local MCP server over stdio."""
    create_server().run(transport="stdio")


if __name__ == "__main__":
    main()
