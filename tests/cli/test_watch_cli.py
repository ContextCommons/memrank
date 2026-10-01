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
"""`memrank watch` / `kill` -- the lifecycle verbs the detached-by-default model requires.

`watch <id...>` is the composable heir of the deleted `--wait`: attach late, re-attach, attach
from CI, and watch a whole sweep TOGETHER. Exit code is the worst outcome (0 done, 1 failed,
2 still running when the watch elapsed). Tests are deterministic: terminal runs never poll,
and the timeout case is a zero-second deadline, never a sleep.
"""
from __future__ import annotations

import contextlib
import json
import os
import signal
from datetime import datetime, timezone

import pytest
from typer.testing import CliRunner

from memrank.runner import app
from memrank.runs import status as run_status

runner = CliRunner()


@pytest.fixture
def runs_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMRANK_RUNS_DIR", str(tmp_path))
    return tmp_path


def _run(runs_dir, run_id, *, state, target="word-overlap", pid=None, error=None, **extra):
    run_dir = runs_dir / run_id
    run_dir.mkdir()
    now = datetime.now(timezone.utc).isoformat()
    payload = {"run_id": run_id, "pid": pid, "target": target, "benchmark": "demo",
               "slice": None, "state": state, "progress": {}, "message": "",
               "started_at": now, "updated_at": now, "error": error, **extra}
    (run_dir / "status.json").write_text(json.dumps(payload))
    return run_dir


def test_watching_finished_runs_exits_clean(runs_dir):
    _run(runs_dir, "a", state="done")
    _run(runs_dir, "b", state="done")

    result = runner.invoke(app, ["watch", "a", "b"])
    assert result.exit_code == 0, result.output


def test_any_failure_makes_the_watch_fail(runs_dir):
    _run(runs_dir, "ok", state="done")
    _run(runs_dir, "bad", state="failed", error="engine exploded")

    result = runner.invoke(app, ["watch", "ok", "bad"])
    assert result.exit_code == 1
    assert "engine exploded" in result.output


def test_a_stale_run_is_a_failure_not_a_wait(runs_dir):
    """A dead process will never finish its run; watching it forever would hang a CI job."""
    _run(runs_dir, "orphan", state="running", pid=2 ** 30)

    result = runner.invoke(app, ["watch", "orphan"])
    assert result.exit_code == 1
    assert "stale" in result.output


def test_still_running_at_timeout_is_exit_two(runs_dir, monkeypatch):
    """Giving up is not failure -- the run keeps going; 2 keeps the outcomes distinguishable."""
    from memrank.cli import watch as watch_cli

    monkeypatch.setattr(watch_cli, "WATCH_TIMEOUT_S", 0.0)
    _run(runs_dir, "busy", state="running", pid=os.getpid())

    result = runner.invoke(app, ["watch", "busy"])
    assert result.exit_code == 2
    assert "still running" in result.output


def test_watching_an_unknown_id_is_refused(runs_dir):
    result = runner.invoke(app, ["watch", "no-such-run"])
    assert result.exit_code != 0
    assert "no-such-run" in result.output


def test_watch_is_also_a_runs_subcommand(runs_dir):
    _run(runs_dir, "a", state="done")
    assert runner.invoke(app, ["runs", "watch", "a"]).exit_code == 0


def test_watching_a_cloud_run_fetches_its_artifacts(runs_dir, monkeypatch):
    """watch inherits --wait's promise: a finished cloud run looks like a local one.

    Fetched through the API rather than from S3, so reading your own results needs no AWS account --
    submission never did, and the two not matching was the last leak in the accounts model.
    """
    from memrank.cli import watch as watch_cli

    _run(runs_dir, "c1", state="running", placement="cloud", artifact_bucket="bkt",
         task_arn="arn:aws:ecs:us-east-1:1:task/x/abc")
    monkeypatch.setattr(watch_cli, "_cloud_client", lambda: (_FakeHttp(), "acme"))
    monkeypatch.setattr(watch_cli.run_api_client, "get_run",
                        lambda http, org, rid: {"state": "stopped-success"})
    monkeypatch.setattr(watch_cli.run_api_client, "list_artifacts",
                        lambda http, org, rid: [{"name": "hindsight__demo.json", "size": 3}])
    monkeypatch.setattr(watch_cli.run_api_client, "download_artifact",
                        lambda http, org, rid, name, dest, *a, **kw: dest.write_bytes(b"{}\n"))

    result = runner.invoke(app, ["watch", "c1"])

    assert result.exit_code == 0, result.output
    assert (runs_dir / "c1" / "hindsight__demo.json").read_bytes() == b"{}\n"
    assert run_status.read(runs_dir / "c1")["state"] == "done"


def test_a_run_only_the_org_knows_is_adopted_rather_than_refused(runs_dir, monkeypatch):
    """The failure this fixes: four finished cloud runs whose local directory was gone.

    `watch` refused them -- "no run recorded here" -- so their results were permanently unreachable
    even though the org knew the state and the artifacts were sitting in S3.
    """
    from memrank import settings
    from memrank.cli import watch as watch_cli

    monkeypatch.setattr(settings, "get", lambda key: "acme" if key == "defaults.org" else None)
    monkeypatch.setattr(watch_cli.run_api_client, "authenticated_client", _FakeHttp)
    monkeypatch.setattr(watch_cli, "_cloud_client", lambda: (_FakeHttp(), "acme"))
    monkeypatch.setattr(watch_cli.run_api_client, "get_run", lambda http, org, rid: {
        "state": "stopped-success", "target_ref": "supermemory", "benchmark": "demo",
        "slice": "smoke", "task_arn": "arn:task/x", "region": "us-east-1",
        "cluster": "bench", "log_group": "/ecs/bench", "artifact": {"bucket": "bkt"}})
    monkeypatch.setattr(watch_cli.run_api_client, "list_artifacts",
                        lambda http, org, rid: [{"name": "supermemory__demo.json", "size": 2}])
    monkeypatch.setattr(watch_cli.run_api_client, "download_artifact",
                        lambda http, org, rid, name, dest, *a, **kw: dest.write_bytes(b"{}"))

    result = runner.invoke(app, ["watch", "never-seen-here"])

    assert result.exit_code == 0, result.output
    assert "adopted from your org" in result.output
    assert (runs_dir / "never-seen-here" / "supermemory__demo.json").exists()


def test_a_run_neither_this_machine_nor_the_org_knows_is_still_refused(runs_dir, monkeypatch):
    """Adoption must not turn a typo into a directory full of nothing."""
    from memrank import settings
    from memrank.cli import watch as watch_cli

    monkeypatch.setattr(settings, "get", lambda key: "acme" if key == "defaults.org" else None)
    monkeypatch.setattr(watch_cli.run_api_client, "authenticated_client", _FakeHttp)
    monkeypatch.setattr(watch_cli.run_api_client, "get_run",
                        lambda http, org, rid: (_ for _ in ()).throw(RuntimeError("404")))

    result = runner.invoke(app, ["watch", "typo"])

    assert result.exit_code != 0
    assert "or in your org" in result.output
    assert not (runs_dir / "typo").exists()


class _FakeHttp:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_kill_signals_the_process_group_and_writes_nothing(runs_dir, monkeypatch):
    """The executing child is the only writer of a live run's terminal state: kill delivers
    SIGTERM to the process group (reaching docker compose too) and leaves the record alone --
    the unwinding child marks itself and its own sweep."""
    from memrank.cli import watch as watch_cli

    signals: list = []
    monkeypatch.setattr(watch_cli.os, "killpg", lambda pgid, sig: signals.append((pgid, sig)))
    monkeypatch.setattr(watch_cli.run_status, "pid_alive", lambda pid: True)
    _run(runs_dir, "active", state="running", pid=777)
    _run(runs_dir, "waiting", state="queued", target="none", pid=777)

    result = runner.invoke(app, ["kill", "active"])
    assert result.exit_code == 0, result.output
    assert signals == [(777, signal.SIGTERM)]
    assert run_status.read(runs_dir / "active")["state"] == "running"
    assert run_status.read(runs_dir / "waiting")["state"] == "queued"
    assert "stop signalled" in result.output


def test_a_refused_cloud_kill_prints_the_refusal_not_a_traceback(runs_dir, monkeypatch):
    """The API's refusal is already phrased for a person; it must land as one error line
    (the sibling paths' pattern), never as a RunApiError traceback. Observed 2026-08-03."""
    from memrank.cli import watch as watch_cli
    from memrank.placement.run_api_client import RunApiError

    def refuse(http, org, rid):
        raise RunApiError("could not stop the task (502)", code="stop_failed")

    monkeypatch.setattr(watch_cli, "_cloud_client", lambda: (_FakeHttp(), "acme"))
    monkeypatch.setattr(watch_cli.run_api_client, "kill_run", refuse)
    _run(runs_dir, "c9", state="running", placement="cloud")

    result = runner.invoke(app, ["kill", "c9"])
    assert result.exit_code == 1
    assert not isinstance(result.exception, RunApiError), "refusal must be rendered, not raised"
    assert "could not stop the task" in result.output
    assert run_status.read(runs_dir / "c9")["state"] == "running", "a refused kill changed nothing"


def test_killing_a_stale_run_records_the_crash_not_a_user_kill(runs_dir, monkeypatch):
    """No process exists to write a stale run's end, so kill is the writer -- but the cause
    of death is the crash, never "killed by user", which would relabel the corpse."""
    from memrank.cli import watch as watch_cli

    signals: list = []
    monkeypatch.setattr(watch_cli.os, "killpg", lambda pgid, sig: signals.append((pgid, sig)))
    _run(runs_dir, "orphan", state="running", pid=2 ** 30)

    result = runner.invoke(app, ["kill", "orphan"])
    assert result.exit_code == 0, result.output
    assert signals == []
    data = run_status.read(runs_dir / "orphan")
    assert data["state"] == "failed"
    assert data["error"] == "the process died without recording an outcome"


def test_kill_without_a_recorded_pid_is_refused(runs_dir):
    """A live-looking record with no pid gives kill nothing to signal; guessing is worse."""
    _run(runs_dir, "ghost", state="running", pid=None)

    result = runner.invoke(app, ["kill", "ghost"])
    assert result.exit_code != 0
    assert "pid" in result.output


def test_kill_is_also_a_runs_subcommand(runs_dir):
    _run(runs_dir, "old", state="done")
    assert runner.invoke(app, ["runs", "kill", "old"]).exit_code == 0


def test_kill_stops_every_id_it_is_given(runs_dir, monkeypatch):
    """A comma sweep hands back a list of ids, so `kill` takes the shape `watch` takes:
    stopping that list must not cost N invocations."""
    from memrank.cli import watch as watch_cli

    signals: list = []
    monkeypatch.setattr(watch_cli.os, "killpg", lambda pgid, sig: signals.append(pgid))
    monkeypatch.setattr(watch_cli.run_status, "pid_alive", lambda pid: True)
    _run(runs_dir, "a", state="running", pid=101)
    _run(runs_dir, "b", state="running", pid=102)

    result = runner.invoke(app, ["kill", "a", "b"])
    assert result.exit_code == 0, result.output
    assert signals == [101, 102]


def test_one_unusable_id_does_not_abandon_the_rest_of_the_batch(runs_dir, monkeypatch):
    """The bad target is reported and costs the exit code; its siblings are still stopped --
    a batch that halts on its first bad id leaves live runs running."""
    from memrank.cli import watch as watch_cli

    signals: list = []
    monkeypatch.setattr(watch_cli.os, "killpg", lambda pgid, sig: signals.append(pgid))
    monkeypatch.setattr(watch_cli.run_status, "pid_alive", lambda pid: True)
    _run(runs_dir, "real", state="running", pid=303)

    result = runner.invoke(app, ["kill", "nope", "real"])
    assert result.exit_code == watch_cli.KILL_UNUSABLE_ID
    assert "no run 'nope'" in result.output
    assert signals == [303], "the reachable run was still stopped"


def test_killing_a_cloud_run_asks_the_api_and_records_the_stop(runs_dir, monkeypatch):
    """Place-transparent kill: the API stops the task (ECS's stop is authoritative, so
    there is no writer to race), and the local mirror records the same terminal facts."""
    from memrank.cli import watch as watch_cli

    killed: list = []
    monkeypatch.setattr(watch_cli, "_cloud_client", lambda: (_FakeHttp(), "acme"))
    monkeypatch.setattr(watch_cli.run_api_client, "kill_run",
                        lambda http, org, rid: killed.append((org, rid)) or {"id": rid})
    _run(runs_dir, "c1", state="running", placement="cloud",
         task_arn="arn:aws:ecs:us-east-1:1:task/x/abc")

    result = runner.invoke(app, ["kill", "c1"])
    assert result.exit_code == 0, result.output
    assert killed == [("acme", "c1")]
    data = run_status.read(runs_dir / "c1")
    assert data["state"] == "failed"
    assert data["error"] == "killed by user"
    assert runner.invoke(app, ["kill", "c1"]).output.count("already ended") == 1


def test_killing_a_finished_run_changes_nothing(runs_dir):
    _run(runs_dir, "old", state="done")

    result = runner.invoke(app, ["kill", "old"])
    assert result.exit_code == 0
    assert run_status.read(runs_dir / "old")["state"] == "done"


# --- detaching from a watch ---------------------------------------------------------------------- #

def _running_run(tmp_path, run_id="20260804-060000__demo__aaaaaa"):
    """A local run that never finishes, so the watch has to be ended some other way."""
    directory = tmp_path / run_id
    directory.mkdir(parents=True)
    (directory / "status.json").write_text(json.dumps({
        "run_id": run_id, "pid": os.getpid(), "target": "hindsight", "benchmark": "demo",
        "slice": None, "state": "retrieving", "progress": {}, "message": "", "error": None,
        "started_at": "2026-08-04T06:00:00+00:00",
        "updated_at": datetime.now(timezone.utc).isoformat()}), encoding="utf-8")
    return run_id


def test_pressing_q_ends_the_watch_and_not_the_run(tmp_path, monkeypatch):
    """Detaching is not failure. Exit 2 -- the code `watch` already documents as "still going" --
    because the watch ended and the run did not."""
    from memrank.cli import watch as watch_cli

    run_id = _running_run(tmp_path)
    monkeypatch.setenv("MEMRANK_RUNS_DIR", str(tmp_path))
    monkeypatch.setattr(watch_cli.key_input, "raw_mode",
                        lambda *a, **k: contextlib.nullcontext(True))
    monkeypatch.setattr(watch_cli.key_input, "wait_for_quit", lambda *a, **k: True)

    result = runner.invoke(app, ["watch", run_id])

    assert result.exit_code == 2, result.output
    assert "you quit the watch" in result.output
    assert "it keeps going" in result.output


def test_ctrl_c_is_not_a_traceback(tmp_path, monkeypatch):
    """It used to be. An unhandled KeyboardInterrupt reads like the evaluation crashed, when
    nothing did -- only the viewer left."""
    from memrank.cli import watch as watch_cli

    run_id = _running_run(tmp_path)
    monkeypatch.setenv("MEMRANK_RUNS_DIR", str(tmp_path))
    monkeypatch.setattr(watch_cli.key_input, "raw_mode",
                        lambda *a, **k: contextlib.nullcontext(True))

    def interrupt(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(watch_cli.key_input, "wait_for_quit", interrupt)

    result = runner.invoke(app, ["watch", run_id])

    assert result.exit_code == 2, result.output
    assert "Traceback" not in result.output
    assert "you interrupted the watch" in result.output


def test_the_quit_hint_appears_only_where_a_key_is_read(tmp_path, monkeypatch, capsys):
    """Advertising `q` on a pipe or a background job promises something that will not happen."""
    from memrank.cli import watch as watch_cli

    run_id = _running_run(tmp_path)
    (tmp_path / run_id / "status.json").write_text(json.dumps({
        "run_id": run_id, "pid": os.getpid(), "target": "hindsight", "benchmark": "demo",
        "slice": None, "state": "retrieving", "message": "", "error": None,
        "progress": {"stage": "ingest", "unit": 1, "units": 1,
                     "ingest": {"done": 1, "total": 3, "rate_per_second": 0.2,
                                "eta_seconds": 10},
                     "retrieve": {"done": 0, "total": 5}, "pct": None, "eta_seconds": 10},
        "started_at": "2026-08-04T06:00:00+00:00",
        "updated_at": datetime.now(timezone.utc).isoformat()}), encoding="utf-8")
    monkeypatch.setattr(watch_cli.progress, "supports_redraw", lambda stream: True)

    display = watch_cli.ProgressDisplay()
    display.listening = False
    display.show(tmp_path, [run_id], cloud=None)
    silent = capsys.readouterr().err

    display.listening = True
    display.show(tmp_path, [run_id], cloud=None)

    assert "press q" not in silent
    assert "press q" in capsys.readouterr().err
