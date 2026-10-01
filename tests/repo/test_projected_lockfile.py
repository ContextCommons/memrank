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
"""The projected `uv.lock`, checked against the `pyproject.toml` it ships beside.

WHAT THIS CAN AND CANNOT PROVE. The real criterion is `uv sync --locked` in a fresh public tree,
which needs a network and a virtualenv; it is the `lockfile` job in `.github/workflows/public-ci.yml`
and it is not reproduced here. What IS provable offline is every structural property that criterion
rests on -- no dropped extra survives, nothing dangles, nothing is orphaned, and no pin was invented
-- and those are the properties a future change to `publish.toml` or to the dependency set would
break.

The last of them is the one that makes the prune safe rather than merely convenient. `uv.lock` is a
reproducibility artifact: a public tree pinned to a version we did not pin would be wrong in the
one way nobody would notice. `test_the_prune_only_ever_deletes` says that cannot happen, because
the transform has no branch that writes a version at all.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

# `tomllib` is stdlib only from 3.11, and `pyproject.toml` declares `requires-python = ">=3.10"`.
try:
    import tomllib
except ModuleNotFoundError:                            # pragma: no cover - 3.10 only
    import tomli as tomllib

from tools import lockfile
from tools import manifest as mf
from tools import project as tp

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def dropped() -> list[str]:
    """The extras `publish.toml` says do not ship -- the one input both rewrites read."""
    return mf.load(ROOT)["projection"]["drop_optional_dependencies"]


@pytest.fixture(scope="module")
def internal() -> str:
    return (ROOT / "uv.lock").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def public(internal, dropped) -> dict:
    return tomllib.loads(lockfile.public_lock(internal, dropped))


def _root(lock: dict) -> dict:
    return next(p for p in lock["package"] if p.get("source", {}).get("editable") == ".")


def test_the_projected_lock_offers_no_extra_the_projected_pyproject_dropped(public, dropped):
    """The exact disagreement that made `uv sync --locked` refuse the public tree (ATO-1856)."""
    root = _root(public)

    assert set(root["metadata"]["provides-extras"]).isdisjoint(dropped)
    assert set(root.get("optional-dependencies", {})).isdisjoint(dropped)


def test_no_surviving_extra_still_pulls_a_dropped_one_in(public, dropped):
    """uv flattens `dev = ["memrank[api]"]`, so removing the reference is only half the job."""
    root = _root(public)
    for group in root.get("optional-dependencies", {}).values():
        for requirement in group:
            assert not (requirement["name"] == root["name"]
                        and set(requirement.get("extras", ())) & set(dropped))

    for requirement in root["metadata"]["requires-dist"]:
        assert not set(requirement.get("extras", ())) & set(dropped)
        assert not any(f"extra == '{extra}'" in requirement.get("marker", "")
                       for extra in dropped)


def test_every_requirement_names_a_package_the_lock_still_carries(public):
    """A prune that removes one package too many leaves a reference pointing at nothing."""
    present = {package["name"] for package in public["package"]}
    dangling = sorted(
        requirement["name"]
        for package in public["package"]
        for requirement in (list(package.get("dependencies", []))
                            + [r for g in package.get("optional-dependencies", {}).values()
                               for r in g])
        if requirement["name"] not in present)

    assert dangling == []


def test_nothing_orphaned_survives_the_prune(public):
    """The other direction, and what `uv lock` itself would do: a package nothing reaches is gone.

    An orphan is not an install error -- uv simply rewrites the lock and `--locked` refuses. So the
    property has to be asserted here rather than waited for in CI.
    """
    by_name: dict[str, list[dict]] = {}
    for package in public["package"]:
        by_name.setdefault(package["name"], []).append(package)
    root = _root(public)

    def edges(package: dict, extras: tuple[str, ...]) -> list[dict]:
        groups = package.get("optional-dependencies", {})
        return list(package.get("dependencies", [])) + [r for e in extras for r in groups.get(e, [])]

    def target(requirement: dict) -> tuple[str, tuple[str, ...]]:
        return requirement["name"], tuple(requirement.get("extras",
                                                          requirement.get("extra", ())))

    reached, pending = {root["name"]}, [target(r) for r in
                                        edges(root, tuple(root.get("optional-dependencies", {})))]
    seen: set[tuple[str, tuple[str, ...]]] = set()
    while pending:
        name, extras = pending.pop()
        if (name, extras) in seen:
            continue
        seen.add((name, extras))
        reached.add(name)
        for package in by_name.get(name, ()):
            pending += [target(r) for r in edges(package, extras)]

    assert sorted(set(by_name) - reached) == []


def test_the_prune_only_ever_deletes(internal, dropped):
    """No line in the public lock is a line the internal lock did not already have.

    One exception, and it is stated rather than waived: `provides-extras` is the single line the
    prune rewrites, because a list of extras minus two is not a sublist of anything.
    """
    before = set(internal.splitlines())
    invented = [line for line in lockfile.public_lock(internal, dropped).splitlines()
                if line not in before]

    assert [line for line in invented if not line.startswith("provides-extras = ")] == []


def test_every_surviving_package_keeps_the_pin_it_had(internal, public, dropped):
    """Same names, same versions -- what makes the public tree reproduce OUR resolution."""
    was = {(p["name"], p.get("version")) for p in tomllib.loads(internal)["package"]}
    now = {(p["name"], p.get("version")) for p in public["package"]}

    assert now <= was
    assert now, "a prune that emptied the lock is not a prune"


def test_the_projection_writes_the_pruned_lock(tmp_path, dropped):
    """The end of the wire: what `python -m tools.project` actually puts on disk.

    Idempotence rather than difference, because this file SHIPS: run inside the published tree the
    prune has nothing left to remove, and a test asserting the output differs from the input would
    be red there for the reason it is green here.
    """
    tp.project(ROOT, tmp_path / "out")
    written = (tmp_path / "out" / "uv.lock").read_text(encoding="utf-8")

    assert written == lockfile.public_lock((ROOT / "uv.lock").read_text(encoding="utf-8"), dropped)
    assert lockfile.public_lock(written, dropped) == written


def test_a_second_projection_is_byte_identical(tmp_path):
    """The lock is rewritten by the projector, so idempotence has to be checked on the BYTES.

    `test_the_projection_is_deterministic` compares the path lists; a rewrite that varied with
    dictionary order or with the clock would pass that and still push a different tree each export.
    """
    tp.project(ROOT, tmp_path / "a")
    tp.project(ROOT, tmp_path / "b")

    def tree(where: Path) -> dict[str, bytes]:
        return {str(p.relative_to(where)): p.read_bytes()
                for p in where.rglob("*") if p.is_file()}

    assert tree(tmp_path / "a") == tree(tmp_path / "b")


def test_a_malformed_lock_is_refused_rather_than_half_pruned():
    with pytest.raises(lockfile.LockfileError, match="no \\[\\[package\\]\\]"):
        lockfile.public_lock("version = 1\n", ["api"])


def test_a_lock_with_no_editable_root_is_refused():
    text = '[[package]]\nname = "x"\nversion = "1"\nsource = { registry = "u" }\n'
    with pytest.raises(lockfile.LockfileError, match="editable root"):
        lockfile.public_lock(text, ["api"])


def test_the_public_ci_lockfile_check_is_wired_to_the_locked_path():
    """The point of consequence names itself: every sync in the public workflow is `--locked`."""
    workflow = (ROOT / ".github/workflows/public-ci.yml").read_text(encoding="utf-8")
    syncs = re.findall(r"^\s*- run: uv sync.*$", workflow, re.MULTILINE)

    assert syncs, "the public workflow installs nothing"
    assert [line for line in syncs if "--locked" not in line] == []
