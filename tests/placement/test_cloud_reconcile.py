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
"""Reconciling one cloud run's local record with the org's, and who is allowed to do it.

The defect these close: a cloud run's numbers stay in S3 and its heartbeat stays at `running`
until something reconciles it, and that something lived inside `watch`. A run submitted and not
watched to completion therefore kept BOTH -- 60 of 101 cloud runs with no results on the machine
where this was found, 47 still claiming to run. `runs show` printed a live `done` directly above
`(none recorded yet)`, which is the disagreement made visible.

So the interesting assertions here are not "does it fetch" but "does every path fetch".
"""
from __future__ import annotations

import json

import pytest

from memrank.placement import run_api_client
from memrank.runs import reconcile
from memrank.runs import status as run_status


@pytest.fixture
def run_dir(tmp_path, monkeypatch):
    """A cloud run as submit leaves it: a heartbeat saying `running`, and nothing else."""
    monkeypatch.setenv("MEMRANK_RUNS_DIR", str(tmp_path))
    directory = tmp_path / "20260804-040634__locomo__e850cb"
    directory.mkdir()
    (directory / "status.json").write_text(json.dumps({
        "run_id": directory.name, "pid": None, "target": "myengine", "benchmark": "locomo",
        "slice": None, "state": "running", "progress": {}, "message": "submitted via API",
        "started_at": "2026-08-04T04:06:36+00:00", "updated_at": "2026-08-04T04:06:36+00:00",
        "error": None, "placement": "cloud"}), encoding="utf-8")
    return directory


@pytest.fixture
def api(monkeypatch):
    """A stand-in for the org: one artifact, and a record the test sets."""
    monkeypatch.setattr(run_api_client, "list_artifacts",
                        lambda http, org, rid: [{"name": "myengine__locomo.json",
                                                 "size": 83_491_477}])
    monkeypatch.setattr(run_api_client, "download_artifact",
                        lambda http, org, rid, name, dest, *a, **kw: dest.write_text(json.dumps(
                            {"adapter": "myengine", "benchmark": "locomo", "composite": 0.62})))
    monkeypatch.setattr("memrank.orchestration.sweep._mirror_to_mlflow", lambda run_dir: None)


def test_a_finished_run_gets_its_numbers_and_says_done(run_dir, api):
    state = reconcile.reconcile(None, "acme", run_dir, {"state": "stopped-success"})

    assert state == "done"
    assert (run_dir / "myengine__locomo.json").is_file()
    assert run_status.read(run_dir)["state"] == "done"


def test_a_failed_run_records_why_and_fetches_nothing(run_dir, api):
    """There are no artifacts to fetch, and asking for them would turn one failure into two."""
    state = reconcile.reconcile(None, "acme", run_dir, {
        "state": "stopped-failed", "exit_code": 1, "stopped_reason": "OutOfMemoryError"})

    assert state == "failed"
    assert "OutOfMemoryError" in run_status.read(run_dir)["error"]
    assert not list(run_dir.glob("myengine__*.json"))


def test_a_running_run_has_its_heartbeat_restamped(run_dir, api):
    """Without this a healthy cloud task reads `stale` in `ps` -- classify() calls a heartbeat
    older than 90s stale, and a cloud record has no pid to check instead."""
    before = run_status.read(run_dir)["updated_at"]

    state = reconcile.reconcile(None, "acme", run_dir, {"state": "running"})

    assert state == "running"
    assert run_status.read(run_dir)["updated_at"] != before


def test_a_refused_download_is_raised_not_swallowed(run_dir, monkeypatch):
    """A run reported successful whose results did not arrive must not read as recorded. Catching
    this here is how `show` would go back to printing "(none recorded yet)" for a 403."""
    def refuse(http, org, rid):
        raise run_api_client.RunApiError("403 Forbidden", code="denied")

    monkeypatch.setattr(run_api_client, "list_artifacts", refuse)

    with pytest.raises(run_api_client.RunApiError):
        reconcile.reconcile(None, "acme", run_dir, {"state": "stopped-success"})


def test_reconciling_twice_changes_nothing(run_dir, api):
    """Callers run this whenever they are unsure, so it has to be safe to be unsure."""
    reconcile.reconcile(None, "acme", run_dir, {"state": "stopped-success"})
    first = (run_dir / "myengine__locomo.json").read_bytes()

    reconcile.reconcile(None, "acme", run_dir, {"state": "stopped-success"})

    assert (run_dir / "myengine__locomo.json").read_bytes() == first


def test_the_size_of_what_is_about_to_download_is_reported_first(run_dir, api):
    """A real locomo cell is 83 MB. Fetched silently that is a minute of a command that has
    printed nothing, which is indistinguishable from a hang -- and was reported as one.

    Two lines now, because a run is no longer a handful of files: it uploads one detail JSON per
    question, so naming every one of them printed 1540 lines and buried the 83 MB one among them.
    The summary carries the count and the total; the named line survives for anything big enough
    that a user is deciding whether to wait for it.
    """
    said = []

    reconcile.reconcile(None, "acme", run_dir, {"state": "stopped-success"},
                              report=said.append)

    assert said == ["1 artifact(s), 83.5 MB", "myengine__locomo.json (83.5 MB)"]


class _Stream:
    """The streaming half of an httpx client, with a body that can die partway through."""

    def __init__(self, chunks, status=200, fail_after=None):
        self.chunks, self.status_code, self.fail_after = chunks, status, fail_after
        self.headers, self.text = {}, ""

    def stream(self, method, url):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return b""

    def iter_bytes(self):
        for index, chunk in enumerate(self.chunks):
            if self.fail_after is not None and index == self.fail_after:
                raise ConnectionError("connection reset")
            yield chunk


def test_a_streamed_artifact_lands_whole(tmp_path):
    dest = tmp_path / "myengine__locomo.json"

    run_api_client.download_artifact(_Stream([b'{"a":', b'1}']), "acme", "r", "n", dest)

    assert dest.read_bytes() == b'{"a":1}'


def test_an_interrupted_download_leaves_no_truncated_artifact(tmp_path):
    """Written to a `.part` and renamed only once whole. A half-written cell on disk is one
    `runs ls` skips, and one a reader could mistake for a run that scored nothing."""
    dest = tmp_path / "myengine__locomo.json"

    with pytest.raises(ConnectionError):
        run_api_client.download_artifact(_Stream([b'{"a":', b'1}'], fail_after=1),
                                         "acme", "r", "n", dest)

    assert not dest.exists(), "a partial body must never appear under the artifact's real name"


def test_a_refused_stream_is_a_refusal_not_a_partial_file(tmp_path):
    """A 403 body is an error document, and writing it to the cell's name would make the next
    reader parse an error as a result."""
    dest = tmp_path / "myengine__locomo.json"

    with pytest.raises(run_api_client.RunApiError):
        run_api_client.download_artifact(_Stream([b"denied"], status=403), "acme", "r", "n", dest)

    assert not dest.exists()


def test_every_path_that_reconciles_goes_through_this_module():
    """The point of the module. `watch`, `runs show` and `runs sync` each reach it by name; a
    fourth caller that fetches artifacts itself would pass every other test in this file and
    reintroduce exactly the drift that made 47 runs read as permanently running."""
    from memrank.cli import runs_show as runs_show_cli
    from memrank.cli import sync as sync_cli
    from memrank.cli import watch as watch_cli

    for module in (watch_cli, runs_show_cli, sync_cli):
        assert module.reconcile is reconcile, module.__name__
    assert not hasattr(watch_cli, "_fetch_artifacts"), (
        "watch must not keep a private copy of the fetch it used to own")
