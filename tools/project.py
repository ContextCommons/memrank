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
"""Produce the public tree from `publish.toml`. The one projector.

    python -m tools.project --out /tmp/memrank-public [--rev HEAD] [--dry-run]

WHAT "THE ONE PROJECTOR" MEANS. Every consumer that needs a public tree runs THIS -- the local
command a person types, and the ops sync that opens the export pull request. A projection that
two programs can produce is a projection that can differ, and the difference is invisible precisely
where it matters: the tree that gets pushed. The export orchestration calls
`python -m tools.project --root <clone> --rev <sha> --out <snapshot>` and pushes what lands there;
it does not build a tree of its own. That command IS the contract, and `--rev` is the whole of it.

WHY A COPY AND NOT `git archive | delete`. Both produce the same tree when nothing goes wrong, and
they fail in opposite directions when something does. Archive-then-delete starts with everything
and removes 1160 paths: a rule that fails to match leaves an internal file in the output, and the
output looks fine. Copy-from-the-public-set starts with nothing and adds 429: a rule that fails to
match leaves a public file OUT, and the tree fails to import. One failure mode is a leak nobody
sees, the other is a red build. This module takes the second.

WHY COMMITTED OBJECTS AND NOT THE WORKING TREE. The projection is a function of a commit, and only
of a commit. Reading blobs at a tree-ish means the same revision projects to the same bytes from
any machine in any state: an uncommitted edit cannot reach the output, an untracked file sitting at
a public path cannot reach it, and a stale file left over from a deleted branch cannot reach it.
Copying from disk made all three possible, and none of them are visible in the result.

The classification rule itself is not reimplemented here -- see `tools/manifest.py`, and neither
is the lockfile prune -- see `tools/lockfile.py`.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tools import lockfile
from tools import manifest as mf

#: A projection smaller than this is a bug, not a small repository. The exact number is asserted by
#: `tests/repo/test_projection.py`; this is the floor that catches a rule matching nothing at all, the
#: same invariant `scripts/internal/ci_public_safe_gate.py` states as "zero files scanned is a
#: FAIL". A tool that silently emits an empty tree passes every check that runs after it.
MINIMUM_PUBLIC_PATHS = 300

#: What the projection is a function of, when the caller names nothing else.
DEFAULT_REVISION = "HEAD"

#: The two blob modes a source tree may contain. A symlink (`120000`) would be copied as its target
#: string and a submodule (`160000`) has no blob at all; both are refused rather than guessed at.
FILE_MODES = {"100644": 0o644, "100755": 0o755}


class ProjectionError(Exception):
    """The projection cannot be produced safely."""


@dataclass(frozen=True)
class Entry:
    """One committed blob: what `git ls-tree` says about a path at a revision."""

    mode: str
    sha: str
    path: str


def committed_entries(root: Path, rev: str = DEFAULT_REVISION) -> tuple[Entry, ...]:
    """Every blob in the tree at `rev`, from the object database.

    `git ls-tree` and not `git ls-files`: the index reflects what is staged on this disk right now,
    which is the working-tree dependency this projector exists to remove. `-z` so a name with a
    space survives; the record itself is `<mode> <type> <sha>\\t<path>`.
    """
    try:
        out = subprocess.run(["git", "ls-tree", "-r", "-z", rev], cwd=root,
                             capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as exc:
        raise ProjectionError(f"cannot read the tree at {rev!r}: {exc.stderr.strip()}") from exc

    entries = []
    for record in out.stdout.split("\0"):
        if not record:
            continue
        meta, path = record.split("\t", 1)
        mode, kind, sha = meta.split()
        if kind != "blob":
            raise ProjectionError(f"{path}: expected a blob at {rev}, found a {kind}")
        if mode not in FILE_MODES:
            raise ProjectionError(f"{path}: unsupported mode {mode} -- refusing to guess at it")
        entries.append(Entry(mode=mode, sha=sha, path=path))
    return tuple(entries)


def committed_paths(root: Path, rev: str = DEFAULT_REVISION) -> tuple[str, ...]:
    """The path set at `rev`, for callers that do not need the blobs."""
    return tuple(entry.path for entry in committed_entries(root, rev))


def _read_blobs(root: Path, shas: tuple[str, ...]) -> dict[str, bytes]:
    """The contents of every named blob, in one `git cat-file --batch` rather than one per file.

    The batch stream is `<sha> <type> <size>\\n<contents>\\n` repeated, and the size is what says
    where a record ends -- content is bytes and may contain anything, newlines included.
    """
    wanted = sorted(set(shas))
    if not wanted:
        return {}
    result = subprocess.run(["git", "cat-file", "--batch"], cwd=root,
                            input="\n".join(wanted).encode(), capture_output=True, check=True)

    blobs: dict[str, bytes] = {}
    stream, offset = result.stdout, 0
    while offset < len(stream):
        header_end = stream.index(b"\n", offset)
        sha, kind, size = stream[offset:header_end].split(b" ")
        if kind != b"blob":
            raise ProjectionError(f"{sha.decode()} is a {kind.decode()}, not a blob")
        start = header_end + 1
        offset = start + int(size)
        blobs[sha.decode()] = stream[start:offset]
        offset += 1                                    # the newline git writes after the contents

    missing = [sha for sha in wanted if sha not in blobs]
    if missing:
        raise ProjectionError(f"{len(missing)} object(s) missing from the repository: {missing[:5]}")
    return blobs


def _safe_destination(out: Path, relative: str) -> Path:
    """`out / relative`, refusing anything that escapes `out`.

    The manifest is trusted input today and this still runs on every path, because the cost is a
    string comparison and the failure it prevents is writing outside the output directory.
    """
    destination = (out / relative).resolve()
    if not destination.is_relative_to(out.resolve()):
        raise ProjectionError(f"path escapes the output directory: {relative}")
    return destination


def _module_path(module: str) -> str | None:
    """`memrank.ops` -> the path that classifies it. Package before module, since a console script
    names `memrank.ops`, whose file is `memrank/ops/__init__.py` and never `memrank/ops.py` -- the
    distinction the first version of this got wrong, silently keeping the entry."""
    if not module.startswith("memrank"):
        return None
    return module.replace(".", "/") + "/__init__.py"


def _drops_internal_subject(line: str, manifest: dict[str, Any]) -> bool:
    """True for a `pyproject.toml` line whose subject does not ship.

    Two shapes, both keyed on a path the manifest can classify: a per-file ruff ignore
    (`"memrank/api/app.py" = [...]`) and a console script (`memrank-ops = "memrank.ops:main"`).
    """
    stripped = line.strip()
    if stripped.startswith("#") or "=" not in stripped:
        return False
    if stripped.startswith('"'):
        target: str | None = stripped.split('"')[1]
    else:
        value = stripped.split("=", 1)[1].strip().strip('"')
        target = _module_path(value.split(":")[0]) if ":" in value else None
    return bool(target and "/" in target and mf.classify(target, manifest)[0] == mf.INTERNAL)


def _public_pyproject(text: str, manifest: dict[str, Any]) -> str:
    """`pyproject.toml` with the entries whose subjects do not ship.

    Derived from the manifest rather than from line numbers, so it cannot rot: an entry is dropped
    BECAUSE its subject classifies internal. The two things a path cannot express -- which optional
    dependency groups exist only for internal code, and the `Homepage` naming the private repo --
    are read from `[projection]` and rewritten explicitly.
    """
    extras = manifest.get("projection", {}).get("drop_optional_dependencies", [])
    kept, skipping = [], None
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if skipping is not None:                       # inside a dropped `extra = [ ... ]` block
            if stripped == "]":
                skipping = None
            continue
        if any(stripped.startswith(f"{extra} = [") for extra in extras):
            skipping = stripped.split(" =")[0]
            continue
        if any(stripped == f'"memrank[{extra}]",' for extra in extras):
            continue                                   # the dev extra's self-reference
        if _drops_internal_subject(line, manifest):
            continue
        kept.append(line)
    return "".join(kept).replace(
        'Homepage = "https://github.com/atomicstrata/memrank-internal"',
        'Homepage = "https://github.com/ContextCommons/memrank"')


def project(root: Path, out: Path, rev: str = DEFAULT_REVISION, dry_run: bool = False) -> list[str]:
    """Write the tree `rev` projects to under `out`. Returns the projected paths.

    Nothing is read from the working tree: `root` supplies the object database and the manifest,
    and every byte written comes from a committed blob.

    Two of those blobs are then rewritten in place, both from `[projection]` in the manifest and
    so from ONE statement of which extras do not ship: `pyproject.toml` loses those extras, and
    `uv.lock` loses what they held (`tools/lockfile.py`). They are rewritten together because a
    lock that disagrees with the `pyproject.toml` beside it makes `uv sync --locked` refuse the
    whole tree -- which is what the public repository got until ATO-1856.
    """
    manifest = mf.load(root)
    entries = [e for e in committed_entries(root, rev)
               if mf.classify(e.path, manifest)[0] == mf.PUBLIC]

    if len(entries) < MINIMUM_PUBLIC_PATHS:
        raise ProjectionError(
            f"only {len(entries)} public path(s) -- refusing to emit a tree this small. A prefix "
            f"rule that matches nothing produces a plausible-looking projection that is simply "
            f"wrong.")

    paths = [entry.path for entry in entries]
    if dry_run:
        return paths

    blobs = _read_blobs(root, tuple(entry.sha for entry in entries))
    if out.exists():
        shutil.rmtree(out)
    for entry in entries:
        destination = _safe_destination(out, entry.path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(blobs[entry.sha])
        destination.chmod(FILE_MODES[entry.mode])

    extras = manifest.get("projection", {}).get("drop_optional_dependencies", [])
    pyproject = out / "pyproject.toml"
    pyproject.write_text(_public_pyproject(pyproject.read_text(encoding="utf-8"), manifest),
                         encoding="utf-8")
    if "uv.lock" in paths:                             # classified like any other path, so ask
        lock = out / "uv.lock"
        lock.write_text(lockfile.public_lock(lock.read_text(encoding="utf-8"), extras),
                        encoding="utf-8")
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tools.project", description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="repository root")
    parser.add_argument("--out", type=Path, required=True, help="where to write the public tree")
    parser.add_argument("--rev", default=DEFAULT_REVISION,
                        help="the committed revision to project (default: %(default)s)")
    parser.add_argument("--dry-run", action="store_true",
                        help="list what would be written and change nothing")
    args = parser.parse_args(argv)

    try:
        paths = project(args.root, args.out, rev=args.rev, dry_run=args.dry_run)
    except ProjectionError as exc:
        print(f"projection refused: {exc}", file=sys.stderr)
        return 1

    verb = "would write" if args.dry_run else "wrote"
    where = "" if args.dry_run else f" to {args.out}"
    print(f"{verb} {len(paths)} path(s) from {args.rev}{where}")
    if args.dry_run:
        for relative in paths:
            print(f"  {relative}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
