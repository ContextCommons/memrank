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
"""Reading `publish.toml`: the one implementation of the classification rule.

Two things need it and they must never disagree -- `tests/repo/test_public_boundary.py`, which asserts
the boundary holds, and `tools/project.py`, which acts on it. A second implementation of "longest
prefix wins, exceptions win outright" is a second thing to keep in step, and the failure mode of
the pair drifting is a tree that passes its own test and ships the wrong files.

This module is PUBLIC, which is deliberate. The published repository carries the manifest so it can
check its own boundary; carrying the twenty lines that read it costs nothing and means the check is
not a claim the reader has to take on faith.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

# `tomllib` is stdlib only from 3.11 and the supported floor is 3.10.
try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - only on 3.10
    import tomli as tomllib

PUBLIC, INTERNAL = "public", "internal"


def read_toml(path: Path) -> dict[str, Any]:
    """One place that knows how to parse TOML on both supported Pythons."""
    return tomllib.loads(path.read_text(encoding="utf-8"))


def load(root: Path) -> dict[str, Any]:
    """The manifest, parsed. `root` is the repository root, not the file."""
    return read_toml(root / "publish.toml")


def tracked(root: Path) -> tuple[str, ...]:
    """Every path git knows about HERE AND NOW, NUL-delimited so a name with a space survives.

    The git index rather than a filesystem walk: it excludes build droppings and `__pycache__`
    without a denylist, and it is case-exact on a case-insensitive volume.

    This is the boundary CHECK's source, deliberately: `tests/repo/test_public_boundary.py` must fail on
    a file the moment it is staged, before anyone commits it. The projector reads a different one --
    `tools/project.py:committed_entries` reads the tree at a revision, because what gets published
    is a function of a commit and must not depend on the disk it was produced on.
    """
    out = subprocess.run(["git", "ls-files", "-z"], cwd=root,
                         capture_output=True, text=True, check=True)
    return tuple(p for p in out.stdout.split("\0") if p)


def untracked(root: Path) -> tuple[str, ...]:
    """Every path git does NOT know about, `.gitignore` honoured, NUL-delimited like `tracked`.

    `tracked` is what the boundary check can classify; this is precisely what it cannot see. A file
    that is neither committed, nor staged, nor ignored has been through no boundary review at all,
    and a prefix rule will classify it the moment somebody stages it -- silently, because a rule
    like `docs/` matches a whole directory nobody looked at. Listing the undecided material is what
    lets `tests/repo/test_public_boundary.py` ask for the decision BEFORE the commit that publishes it.
    """
    out = subprocess.run(["git", "ls-files", "--others", "--exclude-standard", "-z"], cwd=root,
                         capture_output=True, text=True, check=True)
    return tuple(p for p in out.stdout.split("\0") if p)


def _matches(path: str, rule: str) -> bool:
    """A rule ending in `/` is a directory prefix; anything else is an exact path."""
    return path.startswith(rule) if rule.endswith("/") else path == rule


def classify(path: str, manifest: dict[str, Any]) -> tuple[str | None, str]:
    """`(side, why)` for one path. Exceptions win outright; otherwise longest prefix wins.

    Returns `(None, reason)` when the manifest cannot decide, so a caller can report WHICH of the
    two failure modes happened -- an unclassified path or an ambiguous one -- rather than only that
    one did.
    """
    exceptions = manifest["exceptions"]
    for side in (INTERNAL, PUBLIC):
        if path in exceptions.get(side, []):
            return side, "exception"

    def longest(side: str) -> str | None:
        matching = [str(r) for r in manifest[side] if _matches(path, str(r))]
        return max(matching, key=len) if matching else None

    best_public, best_internal = longest(PUBLIC), longest(INTERNAL)
    if best_public is None and best_internal is None:
        return None, "matches no rule"
    if best_public is None:
        assert best_internal is not None
        return INTERNAL, best_internal
    if best_internal is None:
        return PUBLIC, best_public
    if len(best_public) == len(best_internal):
        return None, f"ambiguous: {best_public!r} and {best_internal!r} tie"
    return (PUBLIC, best_public) if len(best_public) > len(best_internal) \
        else (INTERNAL, best_internal)


def public_paths(manifest: dict[str, Any], paths: tuple[str, ...]) -> tuple[str, ...]:
    """The published subset, in the order given, so callers are deterministic for free."""
    return tuple(p for p in paths if classify(p, manifest)[0] == PUBLIC)
