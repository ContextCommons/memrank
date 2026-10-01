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
"""A placement reports what it needs, and the run refuses before it is minted.

The failure this closes: with no Docker on PATH, `submit --on local` printed
``error: internal error: FileNotFoundError: ... 'docker'`` followed by "this is a bug in memrank" --
memrank blaming itself for the user's machine -- and it printed it *after* minting a run, so what
the user was left with was a run record marked failed rather than a refusal.

There is deliberately no `doctor` command wrapping these rows. They come from the placement
that needs them and are what `provision` enforces on, so the refusal a user reads and the
failure that would have happened cannot drift apart -- which is the SkyPilot `sky check`
failure mode, where a green check and a red run answer to different code.
"""
from __future__ import annotations

import subprocess

import pytest
from typer.testing import CliRunner

from memrank import runner as runner_mod
from memrank.placement.base import Requirement, require
from memrank.placement.inprocess import InProcessPlacement
from memrank.placement.local import LocalPlacement, PlacementError

runner = CliRunner()


def _docker(monkeypatch, *, returncode=0, stdout="27.0.3", stderr="", missing=False):
    """Stand in for the `docker version` probe: answering, refusing, or absent entirely."""
    def run(argv, **kwargs):
        if missing:
            raise FileNotFoundError(2, "No such file or directory: 'docker'")
        return subprocess.CompletedProcess(argv, returncode, stdout=stdout, stderr=stderr)

    monkeypatch.setattr(subprocess, "run", run)


def test_a_working_daemon_reports_ok(monkeypatch):
    _docker(monkeypatch)

    assert [(r.name, r.ok) for r in LocalPlacement(run_id="x").check()] == [("docker", True)]


def test_an_absent_binary_is_reported_not_raised(monkeypatch):
    """`check` never raises -- a caller must be able to collect every unmet precondition."""
    _docker(monkeypatch, missing=True)

    rows = LocalPlacement(run_id="x").check()
    assert [r.ok for r in rows] == [False]
    assert "install Docker" in rows[0].fix


def test_an_installed_but_stopped_daemon_is_also_unmet(monkeypatch):
    """Docker Desktop installed and not started looks identical to a working one from `which`."""
    _docker(monkeypatch, returncode=1, stdout="", stderr="Cannot connect to the Docker daemon")

    rows = LocalPlacement(run_id="x").check()
    assert rows[0].ok is False
    assert "Cannot connect" in rows[0].detail
    assert "start Docker" in rows[0].fix


def test_in_process_needs_nothing(monkeypatch):
    """The harness is already running, which is the whole placement."""
    assert InProcessPlacement().check() == []


def test_require_names_every_unmet_row_not_just_the_first():
    """Failing in resolution order tells a user to install one thing, then the next."""
    with pytest.raises(PlacementError) as caught:
        require([Requirement(name="docker", ok=False, detail="absent", fix="install it"),
                 Requirement(name="disk", ok=False, detail="full", fix="free some"),
                 Requirement(name="fine", ok=True, detail="present")])

    message = str(caught.value)
    assert "docker: absent -- install it" in message
    assert "disk: full -- free some" in message
    assert "fine" not in message


def test_require_passes_when_everything_is_met():
    require([Requirement(name="docker", ok=True, detail="27.0.3")])


# --- and the run refuses before it exists -------------------------------------------------------- #

def test_submit_on_local_refuses_before_minting_a_run(monkeypatch, tmp_path, capsys):
    """The whole point: a refusal at the terminal, not a run record marked failed.

    Driven through ``runner.main`` rather than ``CliRunner.invoke``, because the error boundary
    lives on ``_BoundedTyper.__call__`` and CliRunner calls the click command underneath it --
    see tests/cli/test_cli_errors.py. What is asserted here is what the user actually reads.
    """
    monkeypatch.setenv("MEMRANK_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("MEMRANK_CONFIG_DIR", str(tmp_path / "cfg"))
    _docker(monkeypatch, missing=True)
    monkeypatch.setattr("sys.argv", ["memrank", "submit", "word-overlap", "demo",
                                     "--on", "local", "--output-dir", str(tmp_path / "out")])

    with pytest.raises(SystemExit) as exited:
        runner_mod.main()

    printed = capsys.readouterr()
    assert exited.value.code == 1
    assert "docker" in printed.err
    assert "internal error" not in printed.err, "the user's machine is not a memrank bug"
    assert "Traceback" not in printed.err + printed.out
    assert not (tmp_path / "runs").exists(), "no run was minted"


def test_a_placement_that_needs_nothing_is_not_gated(monkeypatch, tmp_path):
    """`--on none` must not ask for Docker: it evaluates against something already running."""
    monkeypatch.setenv("MEMRANK_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("MEMRANK_CONFIG_DIR", str(tmp_path / "cfg"))
    _docker(monkeypatch, missing=True)

    result = runner.invoke(runner_mod.app,
                           ["submit", "word-overlap", "demo", "--on", "none",
                            "--run-id", "in-process", "--output-dir", str(tmp_path / "out")])

    assert result.exit_code == 0, result.output
