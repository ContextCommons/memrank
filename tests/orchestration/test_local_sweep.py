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
"""The submit lifecycle: detach by default, one run per target, execute mode via --run-id.

The interface model ("Everything detaches"): ``submit`` returns immediately with one run id per
target, everywhere. A background child (or the cloud container) re-enters ``submit`` with the
hidden repeatable ``--run-id`` -- execute mode, the only blocking path. Refusal gates run in the
submitting parent so they land on a terminal and a refused sweep records nothing.
"""
from __future__ import annotations

import json
import signal
import sys

import pytest
from typer.testing import CliRunner

from memrank.runner import app
from memrank.runs import registry
from memrank.runs import status as run_status

runner = CliRunner()


@pytest.fixture(autouse=True)
def _restore_sigterm():
    """Execute mode installs a real SIGTERM handler; tests must not leak it into each other."""
    original = signal.getsignal(signal.SIGTERM)
    yield
    signal.signal(signal.SIGTERM, original)

#: `demo` declares no slices and always ignored `--slice`; the flag is now retired in favour
#: of the eval ref, so the sweep names the evaluation and nothing else.
_SWEEP_ARGS = ["submit", "word-overlap,no-context", "demo", "--on", "none"]


class _FakeProc:
    pid = 424242


def _submit(tmp_path, monkeypatch, *args):
    """Submit mode: default behavior, with the background spawn captured instead of run."""
    from memrank.orchestration import sweep

    spawned: dict = {}

    def fake_popen(argv, **kwargs):
        spawned["argv"] = argv
        return _FakeProc()

    monkeypatch.setenv("MEMRANK_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(sweep.subprocess, "Popen", fake_popen)
    result = runner.invoke(app, [*_SWEEP_ARGS, "--output-dir", str(tmp_path / "out"), *args])
    return result, spawned


def _execute(tmp_path, monkeypatch, *args):
    """Execute mode (--run-id): what the spawned child / cloud container runs; blocks."""
    monkeypatch.setenv("MEMRANK_RUNS_DIR", str(tmp_path / "runs"))
    return runner.invoke(app, [*_SWEEP_ARGS, "--output-dir", str(tmp_path / "out"),
                               "--run-id", "r1", "--run-id", "r2", *args])


def _statuses() -> dict[str, dict]:
    root = registry.runs_root()
    return {d.name: run_status.read(d) for d in root.iterdir() if d.is_dir()}


def test_submit_returns_immediately_with_bare_ids_on_stdout(tmp_path, monkeypatch):
    result, spawned = _submit(tmp_path, monkeypatch)

    assert result.exit_code == 0, result.output
    statuses = _statuses()
    assert len(statuses) == 2
    assert all(s["state"] == "queued" for s in statuses.values())
    assert {s["target"] for s in statuses.values()} == {"word-overlap", "no-context"}
    assert all(s["experiment_id"] for s in statuses.values())
    assert all(s["experiment_spec"]["eval_ref"] == "demo" for s in statuses.values())
    assert spawned["argv"], "a background child must have been spawned"
    stdout_ids = [ln for ln in result.stdout.splitlines() if ln in statuses]
    assert sorted(stdout_ids) == sorted(statuses), "bare ids, one per line, on stdout"


def test_the_child_argv_is_rebuilt_from_params_not_sys_argv(tmp_path, monkeypatch):
    """sys.argv is not reliably this command's own (under CliRunner it is pytest's); the child
    must be handed the parsed parameters, or a detached run silently runs something else."""
    result, spawned = _submit(tmp_path, monkeypatch)

    argv = spawned["argv"]
    assert argv[:3] == [sys.executable, "-m", "memrank.runner"]
    child = argv[3:]
    assert child[:3] == ["submit", "word-overlap,no-context", "demo"]
    # The one legitimate user path is --output-dir's value (here a pytest tmp dir); nothing
    # ELSE may smell of the invoking process's argv.
    out_i = child.index("--output-dir")
    scrapable = child[:out_i] + child[out_i + 2:]
    assert not any("pytest" in a or a.endswith(".py") for a in scrapable)
    assert "--on none" in " ".join(child)
    ids = [child[i + 1] for i, a in enumerate(child[:-1]) if a == "--run-id"]
    assert sorted(ids) == sorted(_statuses()), "one --run-id per minted run, in order"


def test_the_heartbeats_adopt_the_child_pid(tmp_path, monkeypatch):
    """The parent exits immediately; heartbeats carrying its pid would classify the queued
    sweep as stale. They must carry the pid of the process that will actually run it."""
    _submit(tmp_path, monkeypatch)

    assert all(s["pid"] == _FakeProc.pid for s in _statuses().values())


def test_a_refused_sweep_records_nothing_and_spawns_nothing(tmp_path, monkeypatch):
    """Gates run in the parent: the refusal lands on the terminal, not in a background log."""
    result, spawned = _submit(tmp_path, monkeypatch, "--unit", "no-such-unit-xyz")

    assert result.exit_code != 0
    assert not spawned, "nothing may be spawned after a refusal"
    root = registry.runs_root()
    assert not root.exists() or not list(root.iterdir()), "no run may be recorded"


def test_execute_mode_runs_to_completion_under_the_given_ids(tmp_path, monkeypatch):
    result = _execute(tmp_path, monkeypatch)

    assert result.exit_code == 0, result.output
    statuses = _statuses()
    assert set(statuses) == {"r1", "r2"}
    assert all(s["state"] == "done" for s in statuses.values())
    assert {s["target"] for s in statuses.values()} == {"word-overlap", "no-context"}


def test_a_run_dir_holds_only_its_own_cell(tmp_path, monkeypatch):
    _execute(tmp_path, monkeypatch)

    for run_id, status in _statuses().items():
        run_dir = registry.runs_root() / run_id
        cells = sorted(p.name for p in run_dir.glob("*__*.json")
                       if not p.name.startswith("summary__"))
        assert cells == [f"{status['target']}__demo.json"]
        summary = json.loads((run_dir / "summary__demo.json").read_text(encoding="utf-8"))
        assert [c["adapter"] for c in summary["cells"]] == [status["target"]]


def test_the_output_dir_summary_still_covers_the_whole_sweep(tmp_path, monkeypatch):
    """The overwritten "latest" view in --output-dir keeps its shape -- leaderboard reads it."""
    _execute(tmp_path, monkeypatch)

    summary = json.loads((tmp_path / "out" / "summary__demo.json").read_text(encoding="utf-8"))
    assert sorted(c["adapter"] for c in summary["cells"]) == ["no-context", "word-overlap"]


def test_a_mid_sweep_failure_leaves_no_run_queued(tmp_path, monkeypatch):
    """Fail fast, but honestly: the failed run says what broke, the sibling that never
    started says so -- nothing stays `queued` with nobody left to start it."""
    from memrank.orchestration import sweep

    def boom(*args, **kwargs):
        raise RuntimeError("engine exploded")

    monkeypatch.setattr(sweep, "_run_and_persist", boom)
    result = _execute(tmp_path, monkeypatch)

    assert result.exit_code != 0
    by_target = {s["target"]: s for s in _statuses().values()}
    assert by_target["word-overlap"]["state"] == "failed"
    assert "engine exploded" in by_target["word-overlap"]["error"]
    assert by_target["no-context"]["state"] == "failed"
    assert "not started" in by_target["no-context"]["error"]


def test_a_killed_sweep_marks_the_active_run_and_its_siblings(tmp_path, monkeypatch):
    """SIGTERM's exception unwinds the sweep: the active run records the user kill, the
    sibling that never started says so -- kill itself never touches the records."""
    from memrank.orchestration import sweep

    def killed(*args, **kwargs):
        raise sweep.SweepKilled()

    monkeypatch.setattr(sweep, "_run_and_persist", killed)
    result = _execute(tmp_path, monkeypatch)

    assert result.exit_code != 0
    by_target = {s["target"]: s for s in _statuses().values()}
    assert by_target["word-overlap"]["state"] == "failed"
    assert by_target["word-overlap"]["error"] == "killed by user"
    assert by_target["no-context"]["state"] == "failed"
    assert "not started" in by_target["no-context"]["error"]


def test_the_kill_handler_raises_and_resets_the_disposition(tmp_path, monkeypatch):
    """First SIGTERM unwinds gracefully; resetting to SIG_DFL makes a SECOND one hard-kill --
    that reset is the whole escalation policy."""
    from memrank.orchestration import sweep

    sweep._install_kill_handler()
    handler = signal.getsignal(signal.SIGTERM)
    with pytest.raises(sweep.SweepKilled):
        handler(signal.SIGTERM, None)
    assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL


def test_the_sweep_log_lives_with_the_first_run(tmp_path, monkeypatch):
    """One child, one stdout: run.log in the first run's dir, siblings point at it."""
    _submit(tmp_path, monkeypatch)

    statuses = _statuses()
    with_log = [rid for rid, s in statuses.items()
                if (registry.runs_root() / rid / "run.log").exists()]
    assert len(with_log) == 1
    shared = [s for s in statuses.values() if s.get("log_run")]
    assert len(shared) == 1
    assert shared[0]["log_run"] == with_log[0]


def test_wait_and_detach_no_longer_exist(tmp_path, monkeypatch):
    """Rejected shapes (interface model): waiting is a verb (`watch`); detached is the only
    lifecycle, so a flag naming it would name an inconsistency."""
    for flag in ("--wait", "--detach"):
        result, _ = _submit(tmp_path, monkeypatch, flag)
        assert result.exit_code != 0
        assert "no such option" in result.output.lower()


# --- what a finished run does about the org universe ------------------------------------------- #

@pytest.fixture
def api(monkeypatch, tmp_path):
    """A stubbed transport: records every PUT, and can be told to refuse.

    Config is redirected per test rather than reusing the session-wide isolation, because
    these tests WRITE settings -- leaving `defaults.org` or `sync.auto` behind would make a
    later test's behavior depend on whether this file ran first.
    """
    from memrank.placement import run_api_client

    monkeypatch.setenv("MEMRANK_CONFIG_DIR", str(tmp_path / "cfg"))

    state: dict = {"calls": [], "session": True, "refusal": None}

    class _Client:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def _client():
        if not state["session"]:
            raise run_api_client.RunApiError("not signed in -- run `memrank auth login`",
                                             code="no_session")
        return _Client()

    def _sync(http, org, run_id, payload):
        state["calls"].append(run_id)
        if state["refusal"] is not None:
            raise state["refusal"]
        return {"id": run_id}

    monkeypatch.setattr(run_api_client, "authenticated_client", _client)
    monkeypatch.setattr(run_api_client, "sync_run", _sync)
    from memrank import settings
    settings.put("defaults.org", "acme")
    return state


def test_a_finished_run_reaches_the_org_on_its_own(tmp_path, monkeypatch, api):
    result = _execute(tmp_path, monkeypatch)

    assert result.exit_code == 0, result.output
    assert set(api["calls"]) == {"r1", "r2"}
    assert all(s["synced_org"] == "acme" for s in _statuses().values())


def test_signed_out_the_run_finishes_and_says_nothing_about_syncing(tmp_path, monkeypatch, api):
    """Local-first, login-free: records accumulate and nothing nags."""
    api["session"] = False

    result = _execute(tmp_path, monkeypatch)

    assert result.exit_code == 0, result.output
    assert api["calls"] == []
    assert "sync" not in result.output.lower()
    assert all("synced_org" not in s for s in _statuses().values())


def test_a_refused_sync_leaves_the_run_done_and_pending(tmp_path, monkeypatch, api):
    """A finished evaluation is not failed by a flaky network -- it stays retryable."""
    from memrank.placement import run_api_client

    api["refusal"] = run_api_client.RunApiError("the memrank API is unreachable")

    result = _execute(tmp_path, monkeypatch)

    assert result.exit_code == 0, result.output
    assert all(s["state"] == "done" for s in _statuses().values())
    assert all("synced_org" not in s for s in _statuses().values())
    assert "runs sync" in result.output


def test_a_killed_sweep_reaches_the_org_too(tmp_path, monkeypatch, api):
    """The org must learn a run ended even when it ended badly: the killed run and the
    sibling that never started both sync -- nothing sits pending forever."""
    from memrank.orchestration import sweep

    def killed(*args, **kwargs):
        raise sweep.SweepKilled()

    monkeypatch.setattr(sweep, "_run_and_persist", killed)
    _execute(tmp_path, monkeypatch)

    assert set(api["calls"]) == {"r1", "r2"}


def test_an_operator_interrupt_marks_failed_but_does_not_sync(tmp_path, monkeypatch, api):
    """Ctrl-C is not an outcome to report: issuing an HTTP request inside a KeyboardInterrupt
    unwind is a worse bug than a pending record -- the run stays retryable via `runs sync`."""
    from memrank.orchestration import sweep

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(sweep, "_run_and_persist", interrupted)
    _execute(tmp_path, monkeypatch)

    assert api["calls"] == []
    assert all(s["state"] == "failed" for s in _statuses().values())


def test_turning_auto_sync_off_stops_it(tmp_path, monkeypatch, api):
    from memrank import settings

    settings.put("sync.auto", "false")

    _execute(tmp_path, monkeypatch)

    assert api["calls"] == []


def test_an_eval_failure_reaches_the_org_but_an_interrupt_still_does_not(tmp_path, monkeypatch,
                                                                        api):
    """The condition that decides site 1, pinned from both sides in one place.

    `auto_sync` moved out of `memrank.cli` into `memrank.runs.push` (ATO-2142), and the thing
    that had to survive the move is not the call but the `isinstance(exc, (Exception,
    SweepKilled))` guard around it: an eval failure IS an outcome and reports, while an
    operator interrupt does not, because an HTTP request inside a KeyboardInterrupt unwind is
    worse than a pending record. `test_a_killed_sweep_reaches_the_org_too` and
    `test_an_operator_interrupt_marks_failed_but_does_not_sync` each cover one side; a guard
    that reported for everything would pass the first and a guard that reported for nothing
    would pass the second, so this asserts the plain-Exception side that neither pins.
    """
    from memrank.orchestration import sweep

    def boom(*args, **kwargs):
        raise RuntimeError("engine exploded")

    monkeypatch.setattr(sweep, "_run_and_persist", boom)
    _execute(tmp_path, monkeypatch)

    assert set(api["calls"]) == {"r1", "r2"}, (
        "an eval failure and the sibling it stopped are both outcomes the org must learn")
