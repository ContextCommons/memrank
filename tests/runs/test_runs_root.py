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
"""Where this machine keeps its runs, and how a legacy directory is noticed.

The registry root used to be the RELATIVE path ``runs/``, so it was whichever directory you were
standing in. `memrank runs show <id>` answered with a score in one checkout and "none recorded yet"
in another, for the same id -- which is not a listing bug but a broken handle: a run id has to mean
one thing per machine.

That was survivable while memrank was a repo you `cd`'d into. It stopped being survivable when
memrank became a tool on PATH.
"""
from __future__ import annotations

import pytest

from memrank.runs import registry


@pytest.fixture
def unset(monkeypatch):
    """No ambient override, so the default is what is under test."""
    monkeypatch.delenv("MEMRANK_RUNS_DIR", raising=False)
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)


def test_the_root_is_absolute_so_a_run_id_means_one_thing_per_machine(unset):
    """The defect, stated: a relative root makes the answer depend on the caller's directory."""
    assert registry.runs_root().is_absolute()


def test_the_default_is_the_xdg_data_location(unset, monkeypatch):
    """Data, not configuration. The wallet and settings live in ~/.config/memrank
    (secret_store.config_dir); run artifacts grow without bound and are not something to carry in
    a dotfiles repo, so they take the matching data location instead of joining them."""
    monkeypatch.setattr("pathlib.Path.home", classmethod(lambda cls: cls("/home/probe")))

    assert registry.runs_root().as_posix() == "/home/probe/.local/share/memrank/runs"


def test_xdg_data_home_is_honoured(monkeypatch):
    monkeypatch.delenv("MEMRANK_RUNS_DIR", raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", "/data")

    assert registry.runs_root().as_posix() == "/data/memrank/runs"


def test_an_explicit_runs_dir_still_wins(monkeypatch, tmp_path):
    """What every test in this suite relies on, and what CI sets -- so the new default can never
    send a test into a real home directory."""
    monkeypatch.setenv("MEMRANK_RUNS_DIR", str(tmp_path))
    monkeypatch.setenv("XDG_DATA_HOME", "/data")

    assert registry.runs_root() == tmp_path


# --- noticing what the old default left behind -------------------------------------------------- #

def _run_dir(root, name):
    (root / name).mkdir(parents=True)
    return root / name


def test_a_legacy_directory_holding_unknown_runs_is_reported(monkeypatch, tmp_path):
    """Reported, never touched: two checkouts can hold the same id, and merging them silently is
    where a receipt ends up attached to the wrong run."""
    legacy, root = tmp_path / "checkout", tmp_path / "registry"
    _run_dir(legacy / "runs", "20260804-000000__demo__smoke__aaaaaa")
    monkeypatch.setenv("MEMRANK_RUNS_DIR", str(root))

    assert registry.legacy_runs(cwd=legacy) == legacy / "runs"


def test_a_legacy_directory_whose_runs_are_already_migrated_is_silent(monkeypatch, tmp_path):
    """Once the ids are in the registry, saying so again is noise."""
    legacy, root = tmp_path / "checkout", tmp_path / "registry"
    _run_dir(legacy / "runs", "20260804-000000__demo__smoke__aaaaaa")
    _run_dir(root, "20260804-000000__demo__smoke__aaaaaa")
    monkeypatch.setenv("MEMRANK_RUNS_DIR", str(root))

    assert registry.legacy_runs(cwd=legacy) is None


def test_a_directory_with_no_runs_is_not_reported(monkeypatch, tmp_path):
    legacy, root = tmp_path / "checkout", tmp_path / "registry"
    (legacy / "runs").mkdir(parents=True)
    monkeypatch.setenv("MEMRANK_RUNS_DIR", str(root))

    assert registry.legacy_runs(cwd=legacy) is None


def test_the_registry_root_is_never_reported_as_its_own_legacy(monkeypatch, tmp_path):
    """Someone whose MEMRANK_RUNS_DIR IS ./runs must not be told to migrate onto themselves."""
    root = tmp_path / "runs"
    _run_dir(root, "20260804-000000__demo__smoke__aaaaaa")
    monkeypatch.setenv("MEMRANK_RUNS_DIR", str(root))

    assert registry.legacy_runs(cwd=tmp_path) is None
