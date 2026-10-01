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
"""Deprecated: the hidden `memrank ps`/`logs` commands that follow `submit` runs. Evaluate an
agent with `memrank run` instead; see the README. `render_status` is used by `memrank runs
show` and does not warn.

`memrank ps` / `status` / `logs` -- observe running evals (from any terminal).

Reads the per-run heartbeat (:mod:`memrank.runs.status`) written by ``memrank submit``. Every
submission runs in the background (the interface model's one lifecycle), so this surface is how
a run is observed at all; blocking on one is :mod:`memrank.cli.watch`'s ``watch``.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import typer

from memrank.runs import registry
from memrank.runs import status as run_status
from memrank.term import detail, fmt, progress, style

_ORDER = {"running": 0, "queued": 1, "failed": 2, "done": 3, "stale": 4}

#: What a bare `ps` shows: runs someone is (or will be) attending to. A queued sibling of a
#: sweep is as live as the running one -- the same process will start it next.
_ACTIVE = ("running", "queued")


def elapsed(rec: dict[str, Any]) -> str:
    """Human elapsed time: start -> (finish for terminal runs, else now)."""
    try:
        start = datetime.fromisoformat(rec["started_at"])
        end_state = rec.get("status") in ("done", "failed", "stale")
        end = datetime.fromisoformat(rec["updated_at"]) if end_state else datetime.now(timezone.utc)
        secs = int((end - start).total_seconds())
    except (KeyError, TypeError, ValueError):
        return "—"
    return f"{secs // 60}m{secs % 60:02d}s" if secs >= 60 else f"{secs}s"


def _where(data: dict) -> str:
    """Where a run is executing: a local pid, or the ECS task id for a cloud run.

    A cloud run's `pid=None` is not missing information -- the process lives inside a Fargate task,
    and the task id is the handle that actually lets you find it.
    """
    if data.get("placement") == "cloud":
        return f"task={str(data.get('task_arn', '?')).rsplit('/', 1)[-1]}"
    return f"pid={data.get('pid')}"


def _progress_line(record: dict[str, Any] | None) -> str:
    """One line of a run's stage progress, or a dash.

    Rendered rather than dumped: the record is a nested dict of two stages with rates and ETAs,
    and printing it raw put a JSON blob in the middle of a human-readable block. Uses the same
    renderer `watch` does when it cannot redraw, so the two agree on what a run's state looks like.
    """
    if not record:
        return "—"
    if not record.get("stage"):
        # Pre-stage records (an old run, or one that never started work) carry nothing to render.
        return str(record.get("pct", "—"))
    return progress.line("", record).strip()


def render_status(run_id: str, data: dict[str, Any]) -> None:
    """Print one run's heartbeat block.

    Shared with ``memrank runs show``, which prints this and then the recorded results. Two
    renderers of one thing drift; this one is the only one.
    """
    data["status"] = run_status.classify(data)
    raw_state = data.get("state")
    started = data.get("started_at")
    fields = [
        ("state", f"{style.state(data['status'])} {style.unit(f'({raw_state})')}"),
        ("cell", f"{style.accent(str(data.get('target') or '?'))} × "
                 f"{style.accent(str(data.get('benchmark')))}"),
        ("progress", _progress_line(data.get("progress"))),
        ("message", data.get("message") or style.unit(fmt.UNKNOWN)),
        ("elapsed", f"{elapsed(data)}  {style.unit(f'(started {started})')}"),
    ]
    if data.get("placement") == "cloud":
        task_id = str(data.get("task_arn", "?")).rsplit("/", 1)[-1]
        cluster = data.get("cluster", "?")
        fields.append(("placement",
                       f"cloud {style.unit(f'(task {task_id}, cluster {cluster})')}"))
        # `memrank logs`, not an `aws logs tail` command. The API holds the AWS identity so a user
        # does not need one; printing our log-group and stream naming told them to reach for an
        # account they have no credentials for.
        fields.append(("logs", style.unit(f"memrank logs {run_id}")))
    if data.get("error"):
        fields.append(("error", style.bad(data["error"])))
    detail.emit(detail.panel(run_id, fields))


#: How often a follow asks for more. Seconds, not milliseconds: each poll is a billed CloudWatch
#: call, and a benchmark cell takes minutes.
_FOLLOW_INTERVAL_S = 2.0


def _cloud_logs(run_id: str, *, container: str | None, follow: bool) -> None:
    """Print a cloud run's output, read through the API rather than from AWS directly.

    Polls rather than streams: the API client is synchronous with a per-request timeout, and a
    page-with-cursor route fits it without a new streaming primitive. `complete` ends the loop,
    so an idle stretch mid-run is not mistaken for the end.
    """
    import time

    from memrank import settings
    from memrank.placement import run_api_client

    org = settings.get("defaults.org")
    if not org:
        raise typer.BadParameter(
            "a cloud run's log is read through the API, which needs an org: set one with "
            "`memrank config set defaults.org <slug>`")
    token: str | None = None
    with run_api_client.authenticated_client() as http:
        while True:
            page = run_api_client.run_logs(http, org, run_id, container=container,
                                           next_token=token)
            for event in page.get("events") or []:
                style.out(event["message"].rstrip("\n"))
            fresh = page.get("next_token")
            if not follow or page.get("complete"):
                if not (page.get("events") or fresh):
                    style.say(f"(no output yet for {run_id})")
                return
            # CloudWatch answers with the SAME token at the end of a stream, so an unchanged token
            # means "nothing new" rather than "done".
            token = fresh
            time.sleep(_FOLLOW_INTERVAL_S)


def logs(
    run_id: str = typer.Argument(..., help="run-id"),
    follow: bool = typer.Option(False, "-f", "--follow", help="stream new output (tail -f)"),
    container: str = typer.Option(None, "--container",
                                  help="which container's output, for a cloud run "
                                       "(default the harness; try `engine` to see the engine's)"),
) -> None:
    """Print (or ``-f`` stream) a run's log, wherever it ran."""
    logfile = registry.runs_root() / run_id / "run.log"
    if not logfile.exists():
        data = run_status.read(registry.runs_root() / run_id) or {}
        if data.get("placement") == "cloud":
            _cloud_logs(run_id, container=container, follow=follow)
            return
        # A sweep runs in one process with one stdout: siblings share the first run's log and
        # say so via `log_run`. Following the pointer beats claiming the run produced nothing.
        shared = data.get("log_run")
        shared_file = registry.runs_root() / shared / "run.log" if shared else None
        if shared_file is None or not shared_file.exists():
            style.say(f"(no log for {run_id} -- background submissions write run.log; "
                      f"this run executed in the foreground or predates them)")
            return
        style.say(f"({run_id} is part of a sweep; its output is shared with {shared})")
        logfile = shared_file
    if follow:
        os.execvp("tail", ["tail", "-f", str(logfile)])  # hand off to tail; replaces this process
    style.out(logfile.read_text(encoding="utf-8"), nl=False)
