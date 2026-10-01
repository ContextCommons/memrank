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
"""`memrank logs` and `memrank status`, for a run wherever it ran.

This module had no tests at all, which is part of why a cloud run's output stayed unreachable
through the product for so long: nothing failed when `logs` answered "see the aws logs tail
command" instead of printing the log.

What these pin is the property the accounts design exists for -- a user reads their own run's
output without an AWS account -- and its converse, that no user-facing output mentions our
infrastructure.
"""
from __future__ import annotations

import json

import pytest
import typer

from memrank.cli import monitor as monitor_cli
from memrank.runs import registry
from memrank.runs import status as run_status

CLOUD_STATUS = {
    "run_id": "20260803-204239__demo__smoke__6582aa",
    "state": "running", "target": "hindsight", "benchmark": "demo",
    "placement": "cloud", "pid": None,
    "task_arn": "arn:aws:ecs:us-east-1:1:task/memrank-bench-staging/a0b3e24c",
    "region": "us-east-1", "cluster": "memrank-bench-staging",
    "log_group": "/ecs/memrank-bench-staging",
    "started_at": "2026-08-03T20:42:39+00:00", "updated_at": "2026-08-03T20:42:39+00:00",
}
RUN_ID = CLOUD_STATUS["run_id"]


@pytest.fixture
def cloud_run(monkeypatch, tmp_path):
    """A cloud run's local heartbeat, with no run.log -- the shape `submit --on cloud` leaves."""
    monkeypatch.setattr(registry, "runs_root", lambda: tmp_path)
    directory = tmp_path / RUN_ID
    directory.mkdir(parents=True)
    (directory / "status.json").write_text(json.dumps(CLOUD_STATUS), encoding="utf-8")
    return directory


@pytest.fixture
def api(monkeypatch):
    """Stand in for the API, recording the requests `logs` makes."""
    from memrank import settings
    from memrank.placement import run_api_client

    calls: list[dict] = []
    pages: list[dict] = [{"events": [], "next_token": None, "complete": True}]

    class _Client:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    def run_logs(http, org, run_id, *, container=None, next_token=None):
        calls.append({"org": org, "run_id": run_id, "container": container,
                      "next_token": next_token})
        return pages[min(len(calls), len(pages)) - 1]

    monkeypatch.setattr(settings, "get", lambda key: "acme" if key == "defaults.org" else None)
    monkeypatch.setattr(run_api_client, "authenticated_client", lambda: _Client())
    monkeypatch.setattr(run_api_client, "run_logs", run_logs)
    monkeypatch.setattr(monitor_cli, "_FOLLOW_INTERVAL_S", 0)
    return {"calls": calls, "pages": pages}


# --- a cloud run's output comes back through memrank -------------------------------------------- #

def test_a_cloud_runs_output_is_printed_rather_than_deferred_to_aws(cloud_run, api, capsys):
    """It used to answer "its output is in CloudWatch, not run.log" and point at an `aws logs
    tail` command against an account the user has no credentials for."""
    api["pages"][0] = {"events": [{"timestamp": 1, "message": "ingesting unit 1/1"},
                                  {"timestamp": 2, "message": "recall=1.000"}],
                       "next_token": "t1", "complete": True}

    monitor_cli.logs(RUN_ID, follow=False, container=None)

    printed = capsys.readouterr().out
    assert "ingesting unit 1/1" in printed and "recall=1.000" in printed
    assert "aws logs" not in printed


def test_another_containers_output_can_be_asked_for(cloud_run, api):
    """An engine's own failure is legible only in its log; the harness reported an empty
    retrieval while the engine reported the 401 that caused it."""
    monitor_cli.logs(RUN_ID, follow=False, container="engine")

    assert api["calls"][-1]["container"] == "engine"


def test_following_stops_when_the_run_completes(cloud_run, api):
    """`complete` ends the loop. An empty page cannot: a running task produces nothing for
    stretches at a time, and treating that as the end truncates the log."""
    api["pages"][:] = [
        {"events": [{"timestamp": 1, "message": "first"}], "next_token": "t1", "complete": False},
        {"events": [], "next_token": "t1", "complete": False},
        {"events": [{"timestamp": 2, "message": "last"}], "next_token": "t2", "complete": True},
    ]

    monitor_cli.logs(RUN_ID, follow=True, container=None)

    assert len(api["calls"]) == 3
    assert api["calls"][1]["next_token"] == "t1", "the cursor must advance, or pages repeat"


def test_a_run_with_no_output_yet_says_so_rather_than_printing_nothing(cloud_run, api, capsys):
    """A task that has not started has no stream. Silence would read as "the run produced
    nothing", which is a different and wrong claim."""
    monitor_cli.logs(RUN_ID, follow=False, container=None)

    assert "no output yet" in capsys.readouterr().err


def test_reading_a_cloud_log_needs_an_org_and_says_which_setting(cloud_run, api, monkeypatch):
    """The API is org-scoped, so the request cannot be made without one. Naming the command that
    fixes it beats a KeyError from inside an HTTP client."""
    from memrank import settings

    monkeypatch.setattr(settings, "get", lambda key: None)

    with pytest.raises(typer.BadParameter, match="defaults.org"):
        monitor_cli.logs(RUN_ID, follow=False, container=None)


# --- and no user-facing output mentions our infrastructure -------------------------------------- #

def test_status_points_at_memrank_logs_not_at_the_aws_cli(cloud_run, capsys):
    data = run_status.read(cloud_run)
    monitor_cli.render_status(RUN_ID, data)

    printed = capsys.readouterr().out
    assert f"memrank logs {RUN_ID}" in printed
    assert "aws logs tail" not in printed
    assert "--log-stream-name-prefix" not in printed


def test_logs_is_reachable_by_its_canonical_name_and_by_its_alias(monkeypatch, tmp_path, capsys):
    """The flat form is an ALIAS, so the form it aliases must exist: `memrank logs` worked while
    `memrank runs logs` did not, which made the alias the only form of the command."""
    from typer.testing import CliRunner

    from memrank.runner import app

    monkeypatch.setenv("MEMRANK_RUNS_DIR", str(tmp_path))
    directory = tmp_path / "local-run"
    directory.mkdir()
    (directory / "run.log").write_text("local output\n", encoding="utf-8")
    invoke = CliRunner().invoke

    assert invoke(app, ["runs", "logs", "local-run"]).stdout == invoke(app, ["logs", "local-run"]).stdout
    assert "local output" in invoke(app, ["runs", "logs", "local-run"]).output


def test_a_local_run_still_reads_its_own_file(monkeypatch, tmp_path, capsys):
    """The cloud path is additive: a local run's log is a file, and nothing about that changed."""
    monkeypatch.setattr(registry, "runs_root", lambda: tmp_path)
    directory = tmp_path / "local-run"
    directory.mkdir()
    (directory / "run.log").write_text("local output\n", encoding="utf-8")

    monitor_cli.logs("local-run", follow=False, container=None)

    assert "local output" in capsys.readouterr().out
