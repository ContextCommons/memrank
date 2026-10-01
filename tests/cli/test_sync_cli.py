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
"""``memrank runs sync`` -- what reaches the org, and what deliberately does not.

The API is faked at the transport seam (``run_api_client``), so these pin the client's half
of the contract: which runs are considered pending, what the uploaded payload contains -- the
stripping assertion is the load-bearing one -- and that a marked run is not offered twice.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from typer.testing import CliRunner

from memrank import settings
from memrank.cli import sync as sync_cli
from memrank.placement import run_api_client
from memrank.runner import app
from memrank.runs import status as run_status

runner = CliRunner()


@pytest.fixture
def runs_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMRANK_RUNS_DIR", str(tmp_path / "runs"))
    (tmp_path / "runs").mkdir()
    return tmp_path / "runs"


@pytest.fixture
def signed_in(monkeypatch, tmp_path, runs_dir):
    """A machine with a session and a default org; every PUT is recorded, none is sent."""
    monkeypatch.setenv("MEMRANK_CONFIG_DIR", str(tmp_path / "cfg"))
    for spec in settings.SETTINGS:
        monkeypatch.delenv(spec.env, raising=False)
    settings.put("defaults.org", "acme")
    state: dict = {"calls": [], "raise_for": {}}

    class _Client:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def _sync(http, org, run_id, payload):
        state["calls"].append({"org": org, "run_id": run_id, "payload": payload})
        problem = state["raise_for"].get(run_id)
        if problem is not None:
            raise problem
        return {"id": run_id, "place": "local"}

    monkeypatch.setattr(run_api_client, "authenticated_client", lambda: _Client())
    monkeypatch.setattr(run_api_client, "sync_run", _sync)
    return state


def _finished_run(runs_dir, run_id="20260803-090000__demo__smoke__loc001", *, state="done",
                  placement=None, synced_org=None, cell=True):
    """A run dir as a finished local run leaves it: heartbeat, cell, summary."""
    run_dir = runs_dir / run_id
    run_dir.mkdir()
    payload = {"run_id": run_id, "pid": None, "target": "word-overlap", "benchmark": "demo",
               "slice": "smoke", "state": state, "progress": {}, "message": "",
               "started_at": "2026-08-03T09:00:00+00:00",
               "updated_at": "2026-08-03T09:05:00+00:00", "error": None}
    if placement:
        payload["placement"] = placement
    if synced_org:
        payload["synced_org"] = synced_org
    (run_dir / "status.json").write_text(json.dumps(payload))
    if cell:
        (run_dir / "baseline__demo.json").write_text(json.dumps({
            "adapter": "word-overlap", "benchmark": "demo", "composite": 0.5,
            "receipt": {"config_hash": "abc123", "started_at": "2026-08-03T09:00:00+00:00"},
            "per_query": [{"query_id": "q1", "text": "x" * 5000}],
            "ingested_documents": [{"id": "d1", "content": "y" * 5000}]}))
        (run_dir / "summary__demo.json").write_text(json.dumps({"benchmark": "demo",
                                                                "tier": "100k"}))
    return run_dir


def test_a_bare_sync_pushes_every_pending_run_and_marks_it(signed_in, runs_dir):
    first = _finished_run(runs_dir)
    second = _finished_run(runs_dir, "20260803-091000__demo__smoke__loc002")

    result = runner.invoke(app, ["runs", "sync"])

    assert result.exit_code == 0
    assert {c["run_id"] for c in signed_in["calls"]} == {first.name, second.name}
    assert run_status.read(first)["synced_org"] == "acme"
    assert first.name in result.stdout  # bare ids on stdout: the machine-readable handle


def test_a_synced_run_is_not_offered_twice(signed_in, runs_dir):
    _finished_run(runs_dir)
    runner.invoke(app, ["runs", "sync"])
    signed_in["calls"].clear()

    result = runner.invoke(app, ["runs", "sync"])

    assert result.exit_code == 0
    assert signed_in["calls"] == []
    assert "nothing pending" in result.output


def test_the_payload_is_the_record_not_the_dataset(signed_in, runs_dir):
    """The receipt travels; the per-query transcript and ingested corpus do not."""
    _finished_run(runs_dir)

    runner.invoke(app, ["runs", "sync"])

    cell = signed_in["calls"][0]["payload"]["record"]["cells"][0]
    assert cell["composite"] == 0.5
    assert cell["receipt"]["config_hash"] == "abc123"
    assert "per_query" not in cell and "ingested_documents" not in cell


def test_the_payload_carries_the_runs_own_times_and_tier(signed_in, runs_dir):
    _finished_run(runs_dir)

    runner.invoke(app, ["runs", "sync"])

    payload = signed_in["calls"][0]["payload"]
    assert payload["started_at"] == "2026-08-03T09:00:00+00:00"
    assert payload["finished_at"] == "2026-08-03T09:05:00+00:00"
    assert (payload["tier"], payload["state"]) == ("100k", "done")


def test_a_failed_run_syncs_as_failed(signed_in, runs_dir):
    _finished_run(runs_dir, state="failed", cell=False)

    runner.invoke(app, ["runs", "sync"])

    assert signed_in["calls"][0]["payload"]["state"] == "failed"
    assert signed_in["calls"][0]["payload"]["record"] == {"cells": [], "summary": None}


def test_a_dirty_source_run_is_local_only(signed_in, runs_dir):
    run_dir = _finished_run(runs_dir)
    cell_path = run_dir / "baseline__demo.json"
    cell = json.loads(cell_path.read_text())
    cell["receipt"]["engine_provenance"] = {
        "properties": {"memrank:source_dirty": "true"}}
    cell_path.write_text(json.dumps(cell))

    result = runner.invoke(app, ["runs", "sync", run_dir.name])

    assert result.exit_code != 0
    assert signed_in["calls"] == []
    assert "dirty source tree" in result.output


def test_a_cloud_born_run_is_never_pending(signed_in, runs_dir):
    _finished_run(runs_dir, placement="cloud")

    result = runner.invoke(app, ["runs", "sync"])

    assert signed_in["calls"] == []
    assert "nothing pending" in result.output


def test_a_running_run_is_not_pending(signed_in, runs_dir):
    run_dir = _finished_run(runs_dir, state="running")
    fresh = datetime.now(timezone.utc).isoformat()
    data = json.loads((run_dir / "status.json").read_text())
    data["updated_at"] = fresh
    (run_dir / "status.json").write_text(json.dumps(data))

    runner.invoke(app, ["runs", "sync"])

    assert signed_in["calls"] == []


def test_signed_out_refuses_loudly_and_marks_nothing(monkeypatch, runs_dir):
    run_dir = _finished_run(runs_dir)

    def refuse():
        raise run_api_client.RunApiError("not signed in -- run `memrank auth login`",
                                         code="no_session")

    monkeypatch.setattr(run_api_client, "authenticated_client", refuse)
    result = runner.invoke(app, ["runs", "sync"])

    assert result.exit_code == 1
    assert "auth login" in result.output
    assert "synced_org" not in run_status.read(run_dir)


def test_without_a_default_org_it_names_the_setting(monkeypatch, tmp_path, runs_dir):
    monkeypatch.setenv("MEMRANK_CONFIG_DIR", str(tmp_path / "cfg"))
    for spec in settings.SETTINGS:
        monkeypatch.delenv(spec.env, raising=False)
    monkeypatch.setattr(run_api_client, "authenticated_client",
                        lambda: __import__("contextlib").nullcontext(None))
    _finished_run(runs_dir)

    result = runner.invoke(app, ["runs", "sync"])

    assert result.exit_code != 0
    assert "defaults.org" in result.output


def test_naming_a_cloud_run_is_refused_by_name(signed_in, runs_dir):
    run_dir = _finished_run(runs_dir, placement="cloud")

    result = runner.invoke(app, ["runs", "sync", run_dir.name])

    assert result.exit_code != 0
    assert "born in the org" in result.output


def test_naming_an_unknown_run_is_refused(signed_in, runs_dir):
    result = runner.invoke(app, ["runs", "sync", "no-such-run"])

    assert result.exit_code != 0
    assert "no run 'no-such-run'" in result.output


def test_naming_an_already_synced_run_pushes_it_again(signed_in, runs_dir):
    """Naming a run IS the ask; the route is idempotent, so this is the repair path."""
    run_dir = _finished_run(runs_dir, synced_org="acme")

    result = runner.invoke(app, ["runs", "sync", run_dir.name])

    assert result.exit_code == 0
    assert [c["run_id"] for c in signed_in["calls"]] == [run_dir.name]


def test_one_refusal_does_not_abandon_the_rest(signed_in, runs_dir):
    doomed = _finished_run(runs_dir)
    other = _finished_run(runs_dir, "20260803-091000__demo__smoke__loc002")
    signed_in["raise_for"][doomed.name] = run_api_client.RunApiError(
        "already names a run this sync may not overwrite", code="run_id_conflict")

    result = runner.invoke(app, ["runs", "sync"])

    assert result.exit_code == 1
    assert run_status.read(other)["synced_org"] == "acme"
    assert "synced_org" not in run_status.read(doomed)
    assert doomed.name in result.output and "1 of 2" in result.output


def test_a_run_without_a_usable_heartbeat_is_named_not_crashed(signed_in, runs_dir):
    run_dir = runs_dir / "20260803-092000__demo__smoke__loc003"
    run_dir.mkdir()
    (run_dir / "status.json").write_text(json.dumps(
        {"run_id": run_dir.name, "state": "done", "updated_at": "2026-08-03T09:05:00+00:00"}))

    result = runner.invoke(app, ["runs", "sync"])

    assert result.exit_code == 1
    assert "no complete heartbeat" in result.output


def test_is_pending_is_the_one_definition(runs_dir):
    """The listing's SYNCED column and the verb must never disagree about "pending"."""
    done = run_status.read(_finished_run(runs_dir))
    cloud = run_status.read(_finished_run(runs_dir, "b__demo__x", placement="cloud"))
    marked = run_status.read(_finished_run(runs_dir, "c__demo__x", synced_org="acme"))

    assert sync_cli.is_pending(done)
    assert not sync_cli.is_pending(cloud)
    assert not sync_cli.is_pending(marked)
    assert not sync_cli.is_pending(None)
