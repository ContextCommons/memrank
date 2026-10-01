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
"""The projected `uv.lock`: this repository's lock with the dropped extras pruned out.

WHY THE LOCK CANNOT SHIP VERBATIM. `tools/project.py` rewrites `pyproject.toml` -- the optional
dependency groups named in `[projection] drop_optional_dependencies` exist only for code that does
not ship, so they go. A lock that still declares those extras disagrees with the `pyproject.toml`
beside it, and `uv sync --locked` refuses the whole tree. That was ATO-1856.

WHY A PRUNE AND NOT A RE-LOCK. Re-locking at projection time would make the output a function of
the network, of PyPI's contents on the day, and of the resolver's version -- and the projection is
meant to be a function of a commit and nothing else (see `tools/project.py`). Measured against
`uv lock` on a real projection, the correct public lock is a PURE DELETION of this one: every
package that survives keeps the byte-identical pin it already had, and the only line rewritten is
`provides-extras`. So the transform is a deletion, computed from the committed bytes.

WHY THAT IS SAFE. A deletion cannot invent a version. The failure mode of a wrong prune is a lock
uv rejects -- the `lockfile` job in `.github/workflows/public-ci.yml` goes red -- and never a
public tree pinned to something we did not pin. The one case that would be wrong-but-quiet is
covered below at `_without_dropped_groups`.

The set of extras to drop is not decided here: it is `[projection] drop_optional_dependencies` in
`publish.toml`, the same table `tools/project.py` reads to rewrite `pyproject.toml`. One input, so
the two outputs cannot drift.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

#: `[[package]]` starts a package; a line of `name = [` inside `[package.optional-dependencies]`
#: starts an extra's requirement list; `{ name = "x", extra(s) = ["y"] }` is one requirement.
PACKAGE_HEADER = "[[package]]"
OPTIONAL_HEADER = "[package.optional-dependencies]"
METADATA_HEADER = "[package.metadata]"
_NAME = re.compile(r'name = "([^"]+)"')
_GROUP = re.compile(r"^(?P<name>[A-Za-z0-9_.-]+) = \[$")
_EXTRAS = re.compile(r'extras? = \[([^\]]*)\]')
#: The project itself, as uv writes it for a lock rooted at the tree being locked.
ROOT_SOURCE = 'source = { editable = "." }'


class LockfileError(Exception):
    """The lockfile is not shaped the way this prune assumes."""


@dataclass
class Package:
    """One `[[package]]` block, split into the parts the prune has to reason about."""

    name: str
    lines: list[str]
    #: Requirements outside any extra, plus `[package.metadata]`, kept verbatim in `lines`.
    groups: dict[str, list[str]]
    is_root: bool

    @property
    def plain_requirements(self) -> list[str]:
        """Every requirement line outside `[package.optional-dependencies]`."""
        stop = len(self.lines)
        for index, line in enumerate(self.lines):
            if line.strip() in (OPTIONAL_HEADER, METADATA_HEADER):
                stop = index
                break
        return [line for line in self.lines[:stop] if line.strip().startswith("{ name = ")]


def _requirement(line: str) -> tuple[str, tuple[str, ...]]:
    """`{ name = "psycopg", extra = ["binary"] }` -> `("psycopg", ("binary",))`."""
    name = _NAME.search(line)
    if name is None:
        raise LockfileError(f"requirement names no package: {line.strip()}")
    extras = _EXTRAS.search(line)
    return name.group(1), tuple(re.findall(r'"([^"]+)"', extras.group(1)) if extras else ())


def _split_packages(text: str) -> tuple[list[str], list[list[str]]]:
    """The lines before the first `[[package]]`, then one list of lines per package."""
    lines = text.splitlines(keepends=True)
    starts = [i for i, line in enumerate(lines) if line.startswith(PACKAGE_HEADER)]
    if not starts:
        raise LockfileError("no [[package]] block in the lockfile")
    bounds = starts + [len(lines)]
    return lines[:starts[0]], [lines[a:b] for a, b in zip(bounds, bounds[1:], strict=False)]


def _groups(lines: list[str]) -> dict[str, list[str]]:
    """Each extra's requirement lines, keyed by extra, from `[package.optional-dependencies]`."""
    try:
        start = next(i for i, line in enumerate(lines) if line.strip() == OPTIONAL_HEADER)
    except StopIteration:
        return {}
    stop = next((i for i, line in enumerate(lines[start + 1:], start + 1)
                 if line.startswith("[")), len(lines))

    groups: dict[str, list[str]] = {}
    open_group: str | None = None
    for line in lines[start + 1:stop]:
        header = _GROUP.match(line.strip())
        if header:
            open_group = header.group("name")
            groups[open_group] = []
        elif open_group is not None and line.strip().startswith("{ name = "):
            groups[open_group].append(line)
        elif line.strip() == "]":
            open_group = None
    return groups


def _parse(text: str) -> tuple[list[str], list[Package]]:
    preamble, blocks = _split_packages(text)
    packages = []
    for lines in blocks:
        name = next((_NAME.search(line).group(1) for line in lines      # type: ignore[union-attr]
                     if line.startswith("name = ")), None)
        if name is None:
            raise LockfileError("a [[package]] block names no package")
        packages.append(Package(name=name, lines=lines, groups=_groups(lines),
                                is_root=any(line.strip() == ROOT_SOURCE for line in lines)))
    return preamble, packages


def _without_dropped_groups(root: Package, drop: set[str]) -> dict[str, list[str]]:
    """The root's remaining extras, with what the dropped ones contributed taken back out.

    uv FLATTENS a self-referential extra: `dev = ["memrank[api]", ...]` in `pyproject.toml` becomes
    a `dev` list here holding the `memrank[api]` reference AND every requirement `api` names. Drop
    the reference alone and the flattened copies stay, so the group still pulls in an extra that no
    longer exists.

    Subtracting the dropped group's own entries is exact unless a surviving group ALSO named one of
    them directly, in which case uv wrote one entry for both reasons and this takes it away. That
    is the single case where the prune can be wrong, and it is not quiet: the group loses a
    requirement `pyproject.toml` still states, so `uv sync --locked` refuses the tree.
    """
    contributed = {_requirement(line)[0] for name in drop for line in root.groups.get(name, ())}
    kept: dict[str, list[str]] = {}
    for name, requirements in root.groups.items():
        if name in drop:
            continue
        kept[name] = [line for line in requirements
                      if not (_requirement(line)[0] == root.name
                              and set(_requirement(line)[1]) & drop)
                      and _requirement(line)[0] not in contributed]
    return kept


def _reachable(root: Package, groups: dict[str, list[str]],
               packages: list[Package]) -> tuple[set[str], set[tuple[str, str]]]:
    """The packages, and the (package, extra) pairs, the public tree still needs.

    Plain reachability over the resolution graph. A package appearing twice under different
    markers is one node here: both versions are kept or neither is, which is what uv does too.
    """
    by_name: dict[str, list[Package]] = {}
    for package in packages:
        by_name.setdefault(package.name, []).append(package)

    seeds = [_requirement(line) for line in root.plain_requirements]
    seeds += [_requirement(line) for requirements in groups.values() for line in requirements]

    seen_packages, seen_extras, pending = {root.name}, set(), list(seeds)
    while pending:
        name, extras = pending.pop()
        fresh = name not in seen_packages
        seen_packages.add(name)
        for package in by_name.get(name, ()):
            if fresh:
                pending += [_requirement(line) for line in package.plain_requirements]
            for extra in extras:
                if (name, extra) in seen_extras:
                    continue
                seen_extras.add((name, extra))
                pending += [_requirement(line) for line in package.groups.get(extra, ())]
    return seen_packages, seen_extras


def _render(package: Package, keep_groups: dict[str, list[str]]) -> list[str]:
    """The package's lines with `[package.optional-dependencies]` replaced by `keep_groups`."""
    if not package.groups:
        return package.lines
    start = next(i for i, line in enumerate(package.lines) if line.strip() == OPTIONAL_HEADER)
    stop = next((i for i, line in enumerate(package.lines[start + 1:], start + 1)
                 if line.startswith("[")), len(package.lines))

    body: list[str] = []
    for name, requirements in keep_groups.items():
        body += [f"{name} = [\n", *requirements, "]\n"]
    if body:
        body.append("\n")                              # uv leaves one blank line after the table
    head = package.lines[:start] + ([OPTIONAL_HEADER + "\n"] if body else [])
    return head + body + package.lines[stop:]


def _public_metadata(lines: list[str], drop: set[str], root_name: str) -> list[str]:
    """`[package.metadata]` without the dropped extras: their requirements and their names.

    `requires-dist` is what uv compares against `pyproject.toml`, so this is the half that decides
    whether `uv sync --locked` accepts the tree at all.
    """
    kept = []
    for line in lines:
        if any(f"extra == '{extra}'" in line for extra in drop):
            continue
        name, extras = (_requirement(line) if line.strip().startswith("{ name = ")
                        else (None, ()))
        if name == root_name and set(extras) & drop:
            continue
        if line.startswith("provides-extras = "):
            remaining = [e for e in re.findall(r'"([^"]+)"', line) if e not in drop]
            line = "provides-extras = [" + ", ".join(f'"{e}"' for e in remaining) + "]\n"
        kept.append(line)
    return kept


def public_lock(text: str, drop_extras: list[str]) -> str:
    """`uv.lock` as the public tree needs it: the same pins, minus what the dropped extras held."""
    drop = set(drop_extras)
    preamble, packages = _parse(text)
    roots = [package for package in packages if package.is_root]
    if len(roots) != 1:
        raise LockfileError(f"expected exactly one editable root package, found {len(roots)}")
    root = roots[0]

    groups = _without_dropped_groups(root, drop)
    names, extras = _reachable(root, groups, packages)

    rendered = list(preamble)
    for package in packages:
        if package.name not in names:
            continue
        if package.is_root:
            start = next(i for i, line in enumerate(package.lines)
                         if line.strip() == METADATA_HEADER)
            package = Package(name=package.name, groups=package.groups, is_root=True,
                              lines=package.lines[:start] + _public_metadata(
                                  package.lines[start:], drop, package.name))
            rendered += _render(package, groups)
            continue
        rendered += _render(package, {name: lines for name, lines in package.groups.items()
                                      if (package.name, name) in extras})
    return "".join(rendered)
