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
"""Which build a receipt says produced the run (ATO-1855).

``memrank_version`` used to be the package version plus ``git rev-parse HEAD`` run in the
CALLER's working directory. Inside this repository that answer is right by accident; from a
wheel install inside a user's own project it records their commit as memrank's provenance --
a plausible wrong identifier in the artifact whose whole purpose is reproducibility.

The install a receipt is built from is faked here (the exact ``direct_url.json`` pip and uv
write, per PEP 610), while the repositories are real ``git init``s, because the defect was
precisely that a real, unrelated repository answered a question about memrank.
"""
from __future__ import annotations

import importlib.metadata
import json
import subprocess
from pathlib import Path

import pytest

from memrank import __version__
from memrank.provenance.receipt import build_receipt

_RESOLVED_COMMIT = "1537ebb0c0ffee0000000000000000000000dead"


class _FakeDistribution:
    """Enough of ``importlib.metadata.Distribution`` for the one file the reader reads."""

    def __init__(self, record: dict | None):
        self._record = record

    def read_text(self, filename: str) -> str | None:
        return json.dumps(self._record) if self._record is not None else None


@pytest.fixture
def installed_as(monkeypatch):
    """Present memrank as though installed from `record` (``None`` = no PEP 610 file)."""
    def _install(record: dict | None) -> None:
        monkeypatch.setattr(importlib.metadata.Distribution, "from_name",
                            staticmethod(lambda name: _FakeDistribution(record)))
    return _install


def _git(path: Path, *args: str) -> str:
    """`git` in `path`, failing the test rather than the code under test if it errors."""
    done = subprocess.run(["git", "-C", str(path), *args], check=True,
                          capture_output=True, text=True)
    return done.stdout.strip()


def _repository(path: Path, message: str) -> str:
    """A real one-commit checkout at `path`, and its HEAD."""
    path.mkdir(parents=True, exist_ok=True)
    _git(path, "init", "-q")
    _git(path, "config", "user.email", "test@example.com")
    _git(path, "config", "user.name", "Test")
    _git(path, "commit", "-qm", message, "--allow-empty")
    return _git(path, "rev-parse", "HEAD")


def _receipt_version(**overrides) -> str:
    return build_receipt(adapter_name="word-overlap", adapter_version="0.1.0",
                         engine_version="0.1.0", benchmark_name="demo",
                         dataset_version="demo@v1", seed=42, **overrides).memrank_version


def test_a_wheel_install_records_no_sha_from_the_repository_it_runs_in(installed_as, tmp_path,
                                                                       monkeypatch):
    """The normal published case: a user's project is not a fact about memrank's build."""
    theirs = _repository(tmp_path / "their-project", "their own work")
    monkeypatch.chdir(tmp_path / "their-project")
    installed_as(None)

    version = _receipt_version()

    assert theirs not in version and theirs[:8] not in version
    assert version == __version__


def test_a_source_checkout_still_identifies_its_commit(installed_as, tmp_path, monkeypatch):
    """An editable install's code is whatever is in the tree it tracks -- so that tree's HEAD.

    Run from an unrelated repository on purpose: the answer must come from the recorded tree
    and not from wherever the process happens to be standing.
    """
    checkout = tmp_path / "memrank"
    head = _repository(checkout, "memrank source")
    elsewhere = _repository(tmp_path / "somewhere-else", "unrelated")
    monkeypatch.chdir(tmp_path / "somewhere-else")
    installed_as({"url": checkout.as_uri(), "dir_info": {"editable": True}})

    version = _receipt_version()

    assert version == f"{__version__}+{head[:8]}"
    assert elsewhere[:8] not in version


def test_a_wheel_install_still_says_which_released_build_produced_it(installed_as, tmp_path,
                                                                     monkeypatch):
    """Absent commit, present release: the version IS the identity of a wheel build."""
    _repository(tmp_path / "their-project", "their own work")
    monkeypatch.chdir(tmp_path / "their-project")
    installed_as({"url": "https://example.invalid/memrank-0.2.0-py3-none-any.whl",
                  "dir_info": {}})

    assert _receipt_version() == __version__


def test_a_git_install_is_identified_by_the_commit_its_own_metadata_resolved(installed_as,
                                                                            tmp_path, monkeypatch):
    """`uv tool install git+...@dev` pins a commit at install time; no `git` call can beat it."""
    _repository(tmp_path / "their-project", "their own work")
    monkeypatch.chdir(tmp_path / "their-project")
    installed_as({"url": "https://github.com/ContextCommons/memrank",
                  "vcs_info": {"vcs": "git", "requested_revision": "dev",
                               "commit_id": _RESOLVED_COMMIT}})

    assert _receipt_version() == f"{__version__}+{_RESOLVED_COMMIT[:8]}"
