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
"""`runs show` fetches the results it would otherwise report missing.

What this closes, verbatim from a real run:

    state:    done (stopped-success)
    artifact: s3://memrank-bench-artifacts-staging-.../cloud-runs/20260804-040634__locomo__e850cb
    results:  (none recorded yet)

State came from the org, results came from disk, and nothing had ever brought the two together for
a run nobody watched. The second line is also false -- the run recorded results; this machine did
not have them -- and a false "none recorded" sends a reader looking for a failure that never
happened.
"""
from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from memrank import settings
from memrank.cli import runs_show as runs_show_cli
from memrank.placement import run_api_client
from memrank.runner import app

runner = CliRunner()

RUN_ID = "20260804-040634__locomo__e850cb"
CELL = {"adapter": "myengine", "benchmark": "locomo", "composite": 0.6234,
        "quality_metric": "substring_recall", "receipt": {"config_hash": "1eb86e6c"}}


def _record(state: str) -> dict:
    return {"id": RUN_ID, "state": state, "target_ref": "myengine", "benchmark": "locomo",
            "place": "cloud", "cluster": "memrank-bench-staging", "region": "us-east-1",
            "created_at": "2026-08-04T04:06:36+00:00", "exit_code": 0,
            "artifact": {"bucket": "b", "prefix": f"cloud-runs/{RUN_ID}"}}


@pytest.fixture
def cloud_run(tmp_path, monkeypatch):
    """A cloud run as submit leaves it: a heartbeat, no results, and an org that has both."""
    monkeypatch.setenv("MEMRANK_RUNS_DIR", str(tmp_path))
    directory = tmp_path / RUN_ID
    directory.mkdir()
    (directory / "status.json").write_text(json.dumps({
        "run_id": RUN_ID, "pid": None, "target": "myengine", "benchmark": "locomo", "slice": None,
        "state": "running", "progress": {}, "message": "submitted via API",
        "started_at": "2026-08-04T04:06:36+00:00", "updated_at": "2026-08-04T04:06:36+00:00",
        "error": None, "placement": "cloud"}), encoding="utf-8")
    monkeypatch.setattr(settings, "get", lambda key, *a, **k: "acme"
                        if key == "defaults.org" else None)
    monkeypatch.setattr(run_api_client, "authenticated_client", _NullClient)
    monkeypatch.setattr(run_api_client, "list_artifacts",
                        lambda http, org, rid: [{"name": "myengine__locomo.json", "size": 4}])
    monkeypatch.setattr(run_api_client, "download_artifact",
                        lambda http, org, rid, name, dest, *a, **kw: dest.write_text(
                            json.dumps(CELL)))
    monkeypatch.setattr("memrank.orchestration.sweep._mirror_to_mlflow", lambda run_dir: None)
    return directory


class _NullClient:
    """Stands in for the authenticated httpx client, which these tests never speak through."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_a_finished_cloud_run_shows_the_numbers_it_recorded(cloud_run, monkeypatch):
    """The bug report, as an assertion."""
    monkeypatch.setattr(runs_show_cli, "platform_record",
                        lambda run_id: _record("stopped-success"))

    output = runner.invoke(app, ["runs", "show", RUN_ID]).output

    assert "composite=0.6234" in output
    assert "none recorded yet" not in output


def test_the_fetched_results_are_on_disk_afterwards(cloud_run, monkeypatch):
    """Fetched, not merely displayed -- so `runs ls`, `report` and the leaderboard see them too."""
    monkeypatch.setattr(runs_show_cli, "platform_record",
                        lambda run_id: _record("stopped-success"))

    runner.invoke(app, ["runs", "show", RUN_ID])

    assert json.loads((cloud_run / "myengine__locomo.json").read_text())["composite"] == 0.6234


def test_a_refused_fetch_says_so_instead_of_none_recorded(cloud_run, monkeypatch):
    """The regression this whole change exists to prevent: a download the org denied must not be
    reported as a run that produced nothing."""
    def refuse(http, org, rid):
        raise run_api_client.RunApiError("403 Forbidden", code="denied")

    monkeypatch.setattr(runs_show_cli, "platform_record",
                        lambda run_id: _record("stopped-success"))
    monkeypatch.setattr(run_api_client, "list_artifacts", refuse)

    output = runner.invoke(app, ["runs", "show", RUN_ID]).output

    assert "in the org, not on this machine" in output
    assert "403" in output
    assert "none recorded yet" not in output


def test_an_unfinished_cloud_run_is_not_fetched(cloud_run, monkeypatch):
    """There is nothing to fetch yet, and asking would refuse -- `show` on a running run is the
    commonest call there is."""
    called = []
    monkeypatch.setattr(runs_show_cli, "platform_record", lambda run_id: _record("running"))
    monkeypatch.setattr(run_api_client, "list_artifacts",
                        lambda *a: called.append(a) or [])

    runner.invoke(app, ["runs", "show", RUN_ID])

    assert not called


def test_a_local_run_with_no_results_still_says_none_recorded(tmp_path, monkeypatch):
    """The one case the old message was right about, kept."""
    monkeypatch.setenv("MEMRANK_RUNS_DIR", str(tmp_path))
    directory = tmp_path / "20260804-000000__demo__smoke__aaaaaa"
    directory.mkdir()
    (directory / "status.json").write_text(json.dumps({
        "run_id": directory.name, "pid": None, "target": "hindsight", "benchmark": "demo",
        "slice": "smoke", "state": "failed", "progress": {}, "message": "",
        "started_at": "2026-08-04T00:00:00+00:00", "updated_at": "2026-08-04T00:01:00+00:00",
        "error": "boom"}), encoding="utf-8")
    monkeypatch.setattr(runs_show_cli, "platform_record", lambda run_id: None)

    output = runner.invoke(app, ["runs", "show", directory.name]).output

    assert "(none recorded yet)" in output
