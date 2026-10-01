"""Bounded run observation and control operations for machine clients."""

from __future__ import annotations

import base64
import json
import os
import signal
from pathlib import Path
from typing import Any

from memrank.application.types import Notice, ToolEnvelope
from memrank.placement import run_api_client
from memrank.runs import registry
from memrank.runs import status as run_status

MAX_LIST_RUNS = 100
MAX_LOG_LINES = 500
MAX_ARTIFACT_BYTES = 262_144


def _local_record(run_id: str) -> tuple[Path, dict[str, Any]] | None:
    run_dir = registry.runs_root() / run_id
    data = run_status.read(run_dir)
    return (run_dir, data) if data is not None else None


def _remote(org: str, operation):
    with run_api_client.authenticated_client() as http:
        return operation(http, org)


def list_runs(*, limit: int = 20, org: str | None = None, mine: bool = True,
              cursor: str | None = None) -> ToolEnvelope:
    """List bounded local runs and, when requested, the org's runs."""
    limit = max(1, min(limit, MAX_LIST_RUNS))
    local = run_status.active_runs(registry.runs_root())
    if cursor:
        local = [item for item in local if (item.get("started_at") or "") < cursor]
    local = local[:limit]
    remote: list[dict[str, Any]] = []
    if org:
        page = _remote(org, lambda http, slug: run_api_client.list_runs(
            http, slug, mine=mine, limit=limit, before=cursor))
        remote = page.get("runs", page.get("items", []))
    by_id = {item.get("run_id", item.get("id")): item for item in local + remote}
    page_rows = sorted(by_id.values(), key=lambda item: (
        item.get("started_at") or item.get("created_at") or ""), reverse=True)[:limit]
    next_cursor = None
    if len(page_rows) == limit:
        next_cursor = page_rows[-1].get("started_at") or page_rows[-1].get("created_at")
    return ToolEnvelope(data={"runs": page_rows, "limit": limit,
                              "next_cursor": next_cursor})


def get_run(run_id: str, *, org: str | None = None) -> ToolEnvelope:
    """Inspect one local or explicitly scoped cloud run."""
    local = _local_record(run_id)
    if local:
        data = dict(local[1])
        data["status"] = run_status.classify(data)
        return ToolEnvelope(data=data)
    if org:
        return ToolEnvelope(data=_remote(
            org, lambda http, slug: run_api_client.get_run(http, slug, run_id)))
    return ToolEnvelope(outcome="refused", refusals=[Notice(
        code="run_not_found", message=f"no local run {run_id!r}; provide org for cloud runs")])


def get_run_logs(run_id: str, *, org: str | None = None, cursor: str | None = None,
                 limit: int = 200, container: str | None = None) -> ToolEnvelope:
    """Read one bounded page of logs without following."""
    limit = max(1, min(limit, MAX_LOG_LINES))
    local = _local_record(run_id)
    if local:
        log_run = local[1].get("log_run", run_id)
        path = registry.runs_root() / log_run / "run.log"
        lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
        start = int(cursor or 0)
        end = min(start + limit, len(lines))
        return ToolEnvelope(data={"lines": lines[start:end],
                                  "next_cursor": str(end) if end < len(lines) else None,
                                  "complete": run_status.classify(local[1]) in ("done", "failed")})
    if org:
        page = _remote(org, lambda http, slug: run_api_client.run_logs(
            http, slug, run_id, container=container, next_token=cursor))
        return ToolEnvelope(data=page)
    return ToolEnvelope(outcome="refused", refusals=[Notice(
        code="run_not_found", message=f"no local run {run_id!r}; provide org for cloud logs")])


def _summary(data: dict[str, Any]) -> dict[str, Any]:
    omitted = {"per_query", "documents", "units", "queries", "retrieved"}
    return {key: value for key, value in data.items() if key not in omitted}


def get_run_result(run_id: str, *, org: str | None = None) -> ToolEnvelope:
    """Return bounded scientific results, retaining receipts and summary metrics."""
    local = _local_record(run_id)
    if local:
        cells = [json.loads(path.read_text(encoding="utf-8"))
                 for path in registry.cell_files(local[0])]
        return ToolEnvelope(data={"run_id": run_id, "cells": [_summary(cell) for cell in cells]})
    if org:
        files = _remote(org, lambda http, slug: run_api_client.list_artifacts(http, slug, run_id))
        summaries = [item for item in files if item["name"].startswith("summary__")]
        readable = next((item for item in summaries
                         if item.get("size", MAX_ARTIFACT_BYTES + 1) <= MAX_ARTIFACT_BYTES), None)
        if readable is None:
            return ToolEnvelope(data={"run_id": run_id, "summary_artifacts": summaries})
        raw = _remote(org, lambda http, slug: run_api_client.fetch_artifact(
            http, slug, run_id, readable["name"]))
        return ToolEnvelope(data={"run_id": run_id, "summary": _summary(json.loads(raw))})
    return ToolEnvelope(outcome="refused", refusals=[Notice(
        code="run_not_found", message=f"no local run {run_id!r}; provide org for cloud results")])


def list_run_artifacts(run_id: str, *, org: str | None = None) -> ToolEnvelope:
    """List artifact handles without reading their content."""
    local = _local_record(run_id)
    if local:
        files = [{"name": path.name, "size": path.stat().st_size}
                 for path in sorted(local[0].iterdir()) if path.is_file()]
        return ToolEnvelope(data={"run_id": run_id, "files": files})
    if org:
        files = _remote(org, lambda http, slug: run_api_client.list_artifacts(http, slug, run_id))
        return ToolEnvelope(data={"run_id": run_id, "files": files})
    return ToolEnvelope(outcome="refused", refusals=[Notice(
        code="run_not_found", message=f"no local run {run_id!r}; provide org for cloud artifacts")])


def _safe_artifact(run_dir: Path, name: str) -> Path:
    path = (run_dir / name).resolve()
    if path.parent != run_dir.resolve():
        raise ValueError("artifact name must identify a file directly inside the run")
    return path


def read_run_artifact(run_id: str, name: str, *, org: str | None = None,
                      offset: int = 0, limit: int = MAX_ARTIFACT_BYTES) -> ToolEnvelope:
    """Read a bounded artifact range as UTF-8 text or base64."""
    limit = max(1, min(limit, MAX_ARTIFACT_BYTES))
    local = _local_record(run_id)
    if local:
        try:
            path = _safe_artifact(local[0], name)
        except ValueError as exc:
            return ToolEnvelope(outcome="refused", refusals=[Notice(
                code="invalid_artifact", message=str(exc))])
        if not path.is_file():
            return ToolEnvelope(outcome="refused", refusals=[Notice(
                code="artifact_not_found", message=f"run has no artifact {name!r}")])
        with path.open("rb") as handle:
            handle.seek(offset)
            content = handle.read(limit)
        total = path.stat().st_size
    elif org:
        content = _remote(org, lambda http, slug: run_api_client.fetch_artifact(
            http, slug, run_id, name))
        total = len(content)
        content = content[offset:offset + limit]
    else:
        return ToolEnvelope(outcome="refused", refusals=[Notice(
            code="run_not_found", message=f"no local run {run_id!r}; provide org")])
    try:
        rendered, encoding = content.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        rendered, encoding = base64.b64encode(content).decode("ascii"), "base64"
    return ToolEnvelope(data={"content": rendered, "encoding": encoding, "offset": offset,
                              "next_offset": offset + len(content) if offset + len(content) < total else None,
                              "size": total})


def cancel_run(run_id: str, *, org: str | None = None) -> ToolEnvelope:
    """Idempotently request cancellation without deleting the run record."""
    local = _local_record(run_id)
    if local:
        state = run_status.classify(local[1])
        if state in ("done", "failed"):
            return ToolEnvelope(data={"run_id": run_id, "state": state, "cancelled": False})
        if local[1].get("placement") == "cloud":
            org = org or local[1].get("org")
        else:
            pid = local[1].get("pid")
            if not pid:
                return ToolEnvelope(outcome="refused", refusals=[Notice(
                    code="run_not_killable", message="run records no process to signal")])
            os.killpg(pid, signal.SIGTERM)
            return ToolEnvelope(data={"run_id": run_id, "state": state, "cancelled": True})
    if org:
        record = _remote(org, lambda http, slug: run_api_client.kill_run(http, slug, run_id))
        return ToolEnvelope(data={"run_id": run_id, "state": record["state"], "cancelled": True})
    return ToolEnvelope(outcome="refused", refusals=[Notice(
        code="run_not_found", message=f"no local run {run_id!r}; provide org for cloud control")])
