"""Run-status heartbeat + classification tests (deterministic -- injected pids/timestamps, no sleeps)."""
from __future__ import annotations

import os
import subprocess
from datetime import datetime, timedelta, timezone

from memrank.runs import status as run_status


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def test_status_roundtrip(tmp_path):
    run_dir = tmp_path / "run-1"
    run_dir.mkdir()
    rs = run_status.RunStatus.create(run_dir, target="mem0", benchmark="demo", slice_="smoke")
    rs.update(state="retrieving", message="halfway", pct=50)
    data = run_status.read(run_dir)
    assert data["state"] == "retrieving"
    assert data["progress"]["pct"] == 50
    assert data["target"] == "mem0"
    assert data["pid"] == os.getpid()


def test_classify_terminal_states():
    assert run_status.classify({"state": "done"}) == "done"
    assert run_status.classify({"state": "failed"}) == "failed"


def test_classify_queued_by_pid_liveness_not_heartbeat_age():
    """A queued sibling's heartbeat is not stamped while an earlier target runs, so freshness
    would call it stale mid-sweep. The pid is the fact that decides: alive means still queued."""
    cold = (datetime.now(timezone.utc) - timedelta(seconds=200)).isoformat()
    live = {"state": "queued", "pid": os.getpid(), "updated_at": cold}
    assert run_status.classify(live) == "queued"
    dead = subprocess.Popen(["true"])
    dead.wait()                                                                 # reaped -> pid dead
    assert run_status.classify({**live, "pid": dead.pid}) == "stale"
    assert run_status.classify({**live, "pid": None}) == "stale"


def test_classify_running_and_stale():
    live = {"state": "retrieving", "pid": os.getpid(), "updated_at": _now()}
    assert run_status.classify(live) == "running"
    dead = subprocess.Popen(["true"])
    dead.wait()                                                                 # reaped -> pid dead
    assert run_status.classify({"state": "retrieving", "pid": dead.pid,
                                "updated_at": _now()}) == "stale"               # pid not alive


def test_classify_believes_a_live_pid_over_a_cold_heartbeat():
    """A local run's pid IS the answer; a quiet phase is not a death.

    This asserted `stale` until 2026-08-04, when a judged run spent ~15 minutes in sequential
    LLM calls without stamping its heartbeat: liveness was established at the pid check and then
    thrown away, so `watch` announced a working run as a corpse -- which `kill` and `watch` both
    read as terminal. The heartbeat is now stamped through judging too, and this is the second,
    independent guard: no future long phase can resurrect the false report.
    """
    cold = (datetime.now(timezone.utc) - timedelta(seconds=200)).isoformat()
    assert run_status.classify({"state": "judging", "pid": os.getpid(),
                                "updated_at": cold}) == "running"


def test_classify_still_reads_a_cold_cloud_heartbeat_as_unknown():
    """No pid, so age is the only evidence -- and for a cloud run it proves nothing terminal.

    `mark_remote` clears the pid precisely so the liveness shortcut above cannot apply here; an
    unwatched task goes cold in 90 seconds while running perfectly well."""
    cold = (datetime.now(timezone.utc) - timedelta(seconds=200)).isoformat()
    remote = {"state": "retrieving", "pid": None, "placement": "cloud", "updated_at": cold}
    assert run_status.classify(remote) == run_status.UNKNOWN_STATE
    assert run_status.classify({**remote, "placement": "local"}) == "stale"


def test_active_runs_scans_and_annotates(tmp_path):
    for name, state in (("run-a", "done"), ("run-b", "retrieving")):
        rd = tmp_path / name
        rd.mkdir()
        run_status.RunStatus.create(rd, target="mem0", benchmark="demo",
                                    slice_=None).update(state=state)
    runs = run_status.active_runs(tmp_path)
    assert {r["run_id"] for r in runs} == {"run-a", "run-b"}
    assert all("status" in r for r in runs)


# --- cloud runs (M6.5) ------------------------------------------------------------------------- #
# A cloud run's pid, if any, belongs to a process inside a Fargate task. Checking it with a local
# os.kill is meaningless at best; at worst it matches an unrelated local process and reports a
# finished run as running. These records carry no pid, so freshness is the only usable signal.

def _cloud(**over):
    data = {"run_id": "r", "pid": None, "placement": "cloud", "task_arn": "arn:.../abc123",
            "state": "running", "updated_at": _now(), "started_at": _now()}
    return data | over


def test_a_fresh_cloud_run_is_running_not_stale():
    """The regression this guards: pid_alive(None) is False, so every cloud run read as stale."""
    assert run_status.classify(_cloud()) == "running"


def test_a_cloud_run_with_a_cold_heartbeat_is_unknown_not_stale():
    """`stale` is a claim this machine cannot make about a run it never executed.

    Only `watch` restamps a cloud heartbeat, so an unwatched cloud run goes cold within 90
    seconds while the task runs on perfectly well. `stale` means something specific and
    terminal elsewhere -- `kill` writes "the process died without recording an outcome" on
    seeing it -- so saying it here reported a fleet of healthy runs as corpses the moment the
    platform could not be reached (2026-08-03). Not knowing is the honest answer.
    """
    cold = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    assert run_status.classify(_cloud(updated_at=cold)) == run_status.UNKNOWN_STATE


def test_a_cloud_run_with_no_usable_timestamp_is_also_unknown():
    assert run_status.classify(_cloud(updated_at=None)) == run_status.UNKNOWN_STATE
    assert run_status.classify(_cloud(updated_at="not-a-date")) == run_status.UNKNOWN_STATE


def test_a_naive_timestamp_is_unusable_not_an_exception():
    """Every heartbeat this code writes is timezone-aware, but a hand-edited or pre-timezone
    record must not take a whole listing down: naive minus aware raises TypeError, which
    `classify` used to let escape."""
    naive = {"run_id": "r", "pid": None, "state": "running",
             "updated_at": "2026-01-01T00:00:00"}
    assert run_status.classify(naive) == "stale"
    assert run_status.classify(naive | {"placement": "cloud"}) == run_status.UNKNOWN_STATE


def test_a_local_run_with_a_cold_heartbeat_is_still_stale():
    """The narrowing is about cloud placement only: a local run that stopped writing IS stale,
    and `kill`/`watch` depend on that meaning."""
    cold = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    local = {"run_id": "r", "pid": None, "state": "running", "updated_at": cold}
    assert run_status.classify(local) == "stale"


def test_a_terminal_cloud_run_stays_terminal():
    assert run_status.classify(_cloud(state="done")) == "done"
    assert run_status.classify(_cloud(state="failed")) == "failed"


def test_a_local_run_with_a_dead_pid_is_still_stale():
    """The pid check must keep working for local runs -- this is a narrowing, not a removal."""
    dead = {"run_id": "r", "pid": 2 ** 30, "state": "running", "updated_at": _now()}
    assert run_status.classify(dead) == "stale"
