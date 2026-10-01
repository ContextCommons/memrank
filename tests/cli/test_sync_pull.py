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
"""`runs sync` pulls as well as pushes.

Sync used to mean "push finished local runs to the org", and refused cloud runs outright -- *"it
was born in the org"*. True, and it left the other half of the disagreement standing: a cloud run
born in the org has its numbers there and nothing here, and its local heartbeat says `running`
until something restamps it. 47 runs on the machine where this was found had been reading as
permanently running, some for four days.

A verb named `sync` that reconciles one direction is the drift, not the fix.
"""
from __future__ import annotations

import json
import threading

import pytest

from memrank.cli import sync as sync_cli
from memrank.placement import run_api_client
from memrank.runs import reconcile


def _write(root, run_id, *, placement, state, cells=()):
    directory = root / run_id
    directory.mkdir(parents=True)
    (directory / "status.json").write_text(json.dumps({
        "run_id": run_id, "pid": None, "target": "myengine", "benchmark": "locomo", "slice": None,
        "state": state, "progress": {}, "message": "", "error": None,
        "started_at": "2026-08-04T04:06:36+00:00", "updated_at": "2026-08-04T04:06:36+00:00",
        "placement": placement}), encoding="utf-8")
    for name in cells:
        (directory / name).write_text(json.dumps(
            {"adapter": "myengine", "benchmark": "locomo", "composite": 0.5}), encoding="utf-8")
    return directory


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMRANK_RUNS_DIR", str(tmp_path))
    return tmp_path


def test_a_cloud_run_stuck_at_running_is_a_pull_candidate(root):
    """The 47. Submitted, never watched, still claiming to run."""
    _write(root, "20260803-222021__locomo__e566b1", placement="cloud", state="running")

    assert [d.name for d, _ in sync_cli.unreconciled_run_dirs()] == [
        "20260803-222021__locomo__e566b1"]


def test_a_cloud_run_that_finished_without_its_numbers_is_also_a_candidate(root):
    """The other way the lag shows: `watch` timed out after stamping done, or the fetch failed.
    A run marked done with nothing to show for it is not reconciled."""
    _write(root, "20260803-222023__locomo__4869c9", placement="cloud", state="done")

    assert len(sync_cli.unreconciled_run_dirs()) == 1


def test_a_reconciled_cloud_run_is_left_alone(root):
    """Finished AND holding its results -- asking the org again is pure cost."""
    _write(root, "20260803-222442__locomo__985797", placement="cloud", state="done",
           cells=("myengine__locomo.json",))

    assert sync_cli.unreconciled_run_dirs() == []


def test_a_local_run_is_never_pulled(root):
    """Its results were born here; there is nothing in the org to bring down."""
    _write(root, "20260804-000000__demo__smoke__aaaaaa", placement="local", state="running")

    assert sync_cli.unreconciled_run_dirs() == []


def test_pull_reconciles_each_candidate_through_the_shared_function(root, monkeypatch):
    """Not through a private copy -- that is how `watch` came to be the only path that worked."""
    _write(root, "20260803-222021__locomo__e566b1", placement="cloud", state="running")
    _write(root, "20260803-222023__locomo__4869c9", placement="cloud", state="running")
    seen = []
    monkeypatch.setattr(run_api_client, "list_runs", lambda http, org, **k: {
        "runs": [{"id": "20260803-222021__locomo__e566b1", "state": "stopped-success"},
                 {"id": "20260803-222023__locomo__4869c9", "state": "stopped-failed"}]})
    monkeypatch.setattr(reconcile, "reconcile",
                        lambda http, org, run_dir, record, **kw: seen.append(run_dir.name)
                        or "done")

    sync_cli._pull(None, "acme")

    assert sorted(seen) == ["20260803-222021__locomo__e566b1", "20260803-222023__locomo__4869c9"]


def test_a_run_the_listing_missed_is_asked_about_by_name(root, monkeypatch):
    """The listing is newest-first across the org and pages; an old run can fall off the end. A
    run silently absent from the map would be reported as untouched when it was never asked
    about."""
    _write(root, "20260730-200910__demo__smoke__93ee66", placement="cloud", state="running")
    monkeypatch.setattr(run_api_client, "list_runs", lambda http, org, **k: {"runs": []})
    monkeypatch.setattr(run_api_client, "get_run",
                        lambda http, org, rid: {"id": rid, "state": "stopped-success"})
    monkeypatch.setattr(reconcile, "reconcile",
                        lambda http, org, run_dir, record, **kw: record["state"])

    sync_cli._pull(None, "acme")  # would KeyError if the fallback were missing


def test_one_runs_refusal_does_not_end_the_batch(root, monkeypatch, capsys):
    """A reconciler does all the work it can, and names what it could not do -- the rule the push
    side already follows."""
    _write(root, "20260803-222021__locomo__e566b1", placement="cloud", state="running")
    _write(root, "20260803-222023__locomo__4869c9", placement="cloud", state="running")
    monkeypatch.setattr(run_api_client, "list_runs", lambda http, org, **k: {
        "runs": [{"id": "20260803-222021__locomo__e566b1", "state": "stopped-success"},
                 {"id": "20260803-222023__locomo__4869c9", "state": "stopped-success"}]})

    def sometimes(http, org, run_dir, record, **kw):
        if run_dir.name.endswith("e566b1"):
            raise run_api_client.RunApiError("403 Forbidden", code="denied")
        return "done"

    monkeypatch.setattr(reconcile, "reconcile", sometimes)

    sync_cli._pull(None, "acme")

    err = capsys.readouterr().err
    assert "403 Forbidden" in err and "1 done" in err and "1 refused" in err


# ------------------------------------------------------------------ #
# Several runs at once, and the output that has to survive it
# ------------------------------------------------------------------ #

_RUN_IDS = [f"20260803-2220{i:02d}__locomo__{i:06x}" for i in range(6)]


def _org(monkeypatch, states=None):
    monkeypatch.setattr(run_api_client, "list_runs", lambda http, org, **k: {
        "runs": [{"id": rid, "state": (states or {}).get(rid, "stopped-success")}
                 for rid in _RUN_IDS]})


@pytest.mark.parametrize("run_workers", (1, 2, 3, 6))
def test_the_lines_come_out_in_listing_order_at_every_width(root, monkeypatch, capsys,
                                                            run_workers):
    """Runs reconcile concurrently, so each one's narration is buffered and replayed by the
    reduce rather than printed from a worker. Two runs interleaving their lines into one stream
    would be unreadable, and output that changed with the width would be untestable."""
    for run_id in _RUN_IDS:
        _write(root, run_id, placement="cloud", state="running")
    _org(monkeypatch)
    monkeypatch.setattr(reconcile, "reconcile", lambda http, org, run_dir, record, **kw: "done")

    sync_cli._pull(None, "acme", run_workers=run_workers)

    said = [line for line in capsys.readouterr().err.splitlines() if "__locomo__" in line]
    assert [line.split()[0] for line in said] == _RUN_IDS


def test_the_runs_really_do_overlap(root, monkeypatch):
    """Without this, every assertion above would also pass a pull that stayed sequential."""
    for run_id in _RUN_IDS:
        _write(root, run_id, placement="cloud", state="running")
    _org(monkeypatch)
    ready = threading.Barrier(2, timeout=10)
    threads: set[int] = set()
    guard = threading.Lock()

    def reconcile_one(http, org, run_dir, record, **kw):
        with guard:
            threads.add(threading.get_ident())
        try:
            ready.wait()
        except threading.BrokenBarrierError:
            pass
        return "done"

    monkeypatch.setattr(reconcile, "reconcile", reconcile_one)

    sync_cli._pull(None, "acme", run_workers=3)

    assert len(threads) > 1


def test_a_batch_of_runs_draws_no_byte_meters(root, monkeypatch):
    """One meter belongs to one transfer. Several runs transferring at once would each redraw
    over the others on the same stream, so the batch keeps its per-run summary lines instead."""
    for run_id in _RUN_IDS:
        _write(root, run_id, placement="cloud", state="running")
    _org(monkeypatch)
    asked: list[bool] = []
    monkeypatch.setattr(reconcile, "reconcile",
                        lambda http, org, run_dir, record, **kw: asked.append(
                            kw["show_progress"]) or "done")

    sync_cli._pull(None, "acme", run_workers=3)

    assert asked == [False] * len(_RUN_IDS)


def test_a_lone_run_keeps_its_bar(root, monkeypatch):
    """It has nobody to collide with, and a single fetch is exactly where the meter earns its
    place -- a 271 MB cell that says nothing for minutes reads as a hang."""
    _write(root, _RUN_IDS[0], placement="cloud", state="running")
    _org(monkeypatch)
    asked: list[bool] = []
    monkeypatch.setattr(reconcile, "reconcile",
                        lambda http, org, run_dir, record, **kw: asked.append(
                            kw["show_progress"]) or "done")

    sync_cli._pull(None, "acme", run_workers=3)

    assert asked == [True]
