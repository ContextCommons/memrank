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
"""The projection, checked.

`tests/repo/test_public_boundary.py` proves the manifest classifies every path. This proves the tool
that ACTS on it agrees -- that what `tools/project.py` writes is exactly what the classifier says is
public, that it refuses rather than emitting a wrong-looking tree, and that the `pyproject.toml`
rewrite drops what does not ship.

It also proves the projection is a function of a COMMIT and of nothing else, which is what lets one
projector serve both the local command and the export: an uncommitted edit, an untracked file at a
public path and a deleted file all leave the output unchanged.

Nothing here runs an install or a subprocess: the expensive end-to-end proof is a fresh virtualenv,
and it belongs in a human's hands or CI, not in the unit suite.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tools import manifest as mf
from tools import project as tp

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def manifest() -> dict:
    return mf.load(ROOT)


def _init_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)


def _commit_all(path: Path) -> None:
    """Stage and commit everything, with an identity so the call works on a bare CI account."""
    subprocess.run(["git", "add", "-A"], cwd=path, check=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false",
                    "commit", "-qm", "commit"], cwd=path, check=True)


@pytest.fixture
def toy(tmp_path, monkeypatch) -> Path:
    """A committed repository small enough to reason about, with the floor lowered to match.

    The real floor exists to catch a rule that matches nothing; a fixture with three files would
    trip it for the wrong reason, and building three hundred files to avoid that would prove less.
    """
    monkeypatch.setattr(tp, "MINIMUM_PUBLIC_PATHS", 1)
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "publish.toml").write_text(
        'public = ["src/", "pyproject.toml"]\ninternal = ["secret/"]\n'
        "[exceptions]\ninternal = []\npublic = []\n")
    (root / "pyproject.toml").write_text("[project]\nname = \"toy\"\n")
    (root / "src" / "kept.py").write_text("committed\n")
    _init_repo(root)
    _commit_all(root)
    return root


def test_the_projection_is_exactly_the_public_set(manifest, tmp_path):
    """No file is added on the way out, and none is lost."""
    written = tp.project(ROOT, tmp_path / "out")
    expected = mf.public_paths(manifest, tp.committed_paths(ROOT))

    assert list(written) == list(expected)
    on_disk = {str(p.relative_to(tmp_path / "out"))
               for p in (tmp_path / "out").rglob("*") if p.is_file()}
    assert on_disk == set(expected)


def test_no_internal_path_survives(manifest, tmp_path):
    """The assertion the whole boundary exists to support, made against a real tree."""
    tp.project(ROOT, tmp_path / "out")
    leaked = sorted(str(p.relative_to(tmp_path / "out"))
                    for p in (tmp_path / "out").rglob("*") if p.is_file()
                    and mf.classify(str(p.relative_to(tmp_path / "out")), manifest)[0] != mf.PUBLIC)

    assert leaked == [], f"{len(leaked)} internal path(s) reached the projection: {leaked[:10]}"


def test_a_projection_that_would_be_nearly_empty_is_refused(tmp_path):
    """A prefix rule matching nothing produces a plausible tree that is simply wrong.

    The same invariant `scripts/internal/ci_public_safe_gate.py` states as "zero files scanned is a
    FAIL" -- a tool that silently emits an empty output passes every check that runs after it.
    """
    _init_repo(tmp_path)
    (tmp_path / "publish.toml").write_text(
        'public = []\ninternal = ["x/"]\n[exceptions]\ninternal = []\npublic = []\n')
    (tmp_path / "x").mkdir()
    (tmp_path / "x" / "only.txt").write_text("internal")
    _commit_all(tmp_path)

    with pytest.raises(tp.ProjectionError, match="refusing to emit"):
        tp.project(tmp_path, tmp_path / "out")


def test_a_path_escaping_the_output_is_refused(tmp_path):
    with pytest.raises(tp.ProjectionError, match="escapes"):
        tp._safe_destination(tmp_path / "out", "../../etc/passwd")


def test_the_rewrite_drops_what_does_not_ship(manifest):
    """Console scripts, optional-dependency groups and ruff ignores whose subject is internal."""
    out = tp._public_pyproject((ROOT / "pyproject.toml").read_text(encoding="utf-8"), manifest)

    assert "memrank-ops" not in out, "a console script for an unpublished module"
    assert "psycopg" not in out, "the `api` extra exists for `memrank/api/**`"
    assert "fastapi" in out, "the public `service` extra ships the evaluation service's stack"
    assert "jupyterlab" not in out, "the `notebooks` extra exists for `notebooks/**`"
    assert '"memrank[api]"' not in out, "the dev extra's self-reference to it"
    assert "tests/internal/" not in out, "ruff ignores for paths that do not ship"
    assert "memrank-internal" not in out, "`Homepage` naming the private repository"
    # ...and keeps what does.
    assert 'memrank = "memrank.runner:main"' in out
    assert "memrank/runner.py" in out


def test_the_projection_is_deterministic(tmp_path):
    """A second export with no source change must be a no-op -- the coordination document's
    idempotence criterion, which a publication pipeline compares against."""
    assert tp.project(ROOT, tmp_path / "a") == tp.project(ROOT, tmp_path / "b")


def test_an_uncommitted_edit_does_not_reach_the_output(toy, tmp_path):
    """The class of bug a working-tree copy could not exclude: whatever is on disk gets shipped."""
    (toy / "src" / "kept.py").write_text("EDITED ON DISK\n")

    tp.project(toy, tmp_path / "out")

    assert (tmp_path / "out" / "src" / "kept.py").read_text() == "committed\n"


def test_an_untracked_file_at_a_public_path_does_not_reach_the_output(toy, tmp_path):
    """A scratch file under a public prefix classifies public. It is still not in the commit."""
    (toy / "src" / "scratch.py").write_text("never committed\n")

    written = tp.project(toy, tmp_path / "out")

    assert "src/scratch.py" not in written
    assert not (tmp_path / "out" / "src" / "scratch.py").exists()


def test_a_deleted_file_still_reaches_the_output(toy, tmp_path):
    """The same property in the other direction: the commit is the source, not the disk."""
    (toy / "src" / "kept.py").unlink()

    written = tp.project(toy, tmp_path / "out")

    assert "src/kept.py" in written
    assert (tmp_path / "out" / "src" / "kept.py").read_text() == "committed\n"


def test_the_same_revision_projects_identically_from_any_working_state(toy, tmp_path):
    """Two machines project one commit to one tree -- the whole point of reading committed objects.

    A second checkout of the same repository stands in for the second machine; the first is left
    dirty, which under a working-tree copy is exactly what would make the two differ.
    """
    elsewhere = tmp_path / "elsewhere"
    subprocess.run(["git", "clone", "-q", str(toy), str(elsewhere)], check=True)
    (toy / "src" / "kept.py").write_text("dirty\n")
    (toy / "src" / "untracked.py").write_text("dirty\n")

    here = tp.project(toy, tmp_path / "here-out")
    there = tp.project(elsewhere, tmp_path / "there-out")

    assert here == there
    assert {p.name: p.read_bytes() for p in (tmp_path / "here-out").rglob("*") if p.is_file()} \
        == {p.name: p.read_bytes() for p in (tmp_path / "there-out").rglob("*") if p.is_file()}


def test_an_unknown_revision_is_refused(toy, tmp_path):
    with pytest.raises(tp.ProjectionError, match="cannot read the tree"):
        tp.project(toy, tmp_path / "out", rev="no-such-ref")


def test_the_executable_bit_survives_the_projection(toy, tmp_path):
    """`scripts/` ships shell entry points; a projection that drops the mode ships broken ones."""
    script = toy / "src" / "run.sh"
    script.write_text("#!/bin/sh\n")
    script.chmod(0o755)
    _commit_all(toy)

    tp.project(toy, tmp_path / "out")

    assert (tmp_path / "out" / "src" / "run.sh").stat().st_mode & 0o111
