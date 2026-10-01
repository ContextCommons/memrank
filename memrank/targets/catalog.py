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
"""Target discovery: where manifests live, and how a ref becomes a resolved Manifest.

Three sources, in increasing precedence: the manifests shipped inside the package, then every
directory on the ``targets.path`` setting, then the operator's own directory
(``${MEMRANK_CONFIG_DIR}/targets``, the same config home the secrets wallet uses). The resolved
canonical ref is what callers echo back, so a short ref never silently means something the operator
did not expect.

``targets.path`` is what lets a descriptor live beside the translator it launches, in a folder the
operator owns -- so the pair can be copied or shared as one thing. Its directories are peers, so a
name two of them both claim is refused rather than resolved. Each entry carries the directory it
came from, because a relative ``binding.root`` is resolved against it.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from memrank.targets.manifest import Manifest, ManifestError, from_dict
from memrank.targets.resolve import apply_overrides, merge, parse_overrides, parse_ref

_MAX_INHERITANCE_DEPTH = 8


class TargetNotFound(ManifestError):
    """No manifest matches the given ref."""


def builtin_dir() -> Path:
    """The manifests shipped inside the package."""
    return Path(__file__).parent / "builtin"


def user_dir() -> Path:
    """The operator's own manifest directory."""
    configured = os.environ.get("MEMRANK_CONFIG_DIR")
    root = Path(configured) if configured else Path.home() / ".config" / "memrank"
    return root / "targets"


def path_dirs() -> list[Path]:
    """The operator's own descriptor directories, from the ``targets.path`` setting.

    This is how a target file gets to live beside the translator it launches -- in a folder the
    operator owns, which may be a repo, a scratch directory, or a subfolder of an engine. Spelled
    as a search path (``os.pathsep``, like ``PATH``) because that is what it is.
    """
    from memrank import settings

    configured = settings.get("targets.path")
    if not configured:
        return []
    dirs = []
    for part in filter(None, configured.split(os.pathsep)):
        directory = Path(part).expanduser()
        # Refused rather than resolved against the cwd: a relative entry points somewhere
        # different from every folder memrank runs in, and a missing directory loads as zero
        # targets -- so the failure would be targets silently vanishing.
        if not directory.is_absolute():
            raise ManifestError(
                f"targets.path entry {part!r} is relative, so it would point at a different "
                f"directory from every folder memrank runs in; use an absolute path "
                f"(memrank config set targets.path <absolute dir>)")
        dirs.append(directory)
    return dirs


def graph_file(filename: str) -> Path:
    """Where a manifest's ``compose:`` file resolves to.

    Searched in the same order manifests are: the operator's directories first, so a local graph can
    stand in for a shipped one, then the package's own.

    Args:
        filename: The value of a manifest's ``compose:`` field.

    Returns:
        The path to use -- the first that exists, else the builtin one, so the caller reports a
        missing file at the location it was expected in.
    """
    candidates = [user_dir() / filename,
                  *(directory / filename for directory in path_dirs()),
                  builtin_dir() / filename]
    return next((path for path in candidates if path.is_file()), candidates[-1])


def _load_dir(path: Path) -> dict[str, tuple[dict[str, Any], Path]]:
    """Load every ``*.yaml`` in ``path``, keyed by name, paired with the directory it came from.

    The directory travels with the data because a descriptor's ``binding.root`` is resolved against
    it. Carried as a pair rather than stuffed into the mapping, which would reach ``from_dict`` and
    be refused as an unknown field.
    """
    if not path.is_dir():
        return {}
    out: dict[str, tuple[dict[str, Any], Path]] = {}
    for file in sorted(path.glob("*.yaml")):
        # Compose files live beside the manifests that name them and are not manifests themselves.
        if file.name.endswith(".compose.yaml"):
            continue
        data = yaml.safe_load(file.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not data.get("name"):
            raise ManifestError(f"{file} does not declare a 'name'")
        out[data["name"]] = (data, path)
    return out


def _load_path_dirs() -> dict[str, tuple[dict[str, Any], Path]]:
    """Every descriptor on ``targets.path``, refusing a name two directories both claim.

    Refused rather than resolved: the directories on a search path are peers, so there is no
    honest rule for which wins. Two setups defining ``myengine:dev`` differently is precisely what
    a quiet merge would hide -- and the run would measure whichever happened to sort first.
    """
    out: dict[str, tuple[dict[str, Any], Path]] = {}
    for directory in path_dirs():
        for name, entry in _load_dir(directory).items():
            if name in out:
                raise ManifestError(
                    f"target {name!r} is defined in both {out[name][1]} and {directory}. Neither "
                    f"directory on targets.path takes precedence over the other, so rename one or "
                    f"drop it from the path.")
            out[name] = entry
    return out


def load_all() -> dict[str, tuple[dict[str, Any], Path]]:
    """Every known manifest with the directory it was read from; later sources win."""
    return {**_load_dir(builtin_dir()), **_load_path_dirs(), **_load_dir(user_dir())}


def load_raw() -> dict[str, dict[str, Any]]:
    """Every known manifest as a raw mapping, keyed by name; the user dir wins."""
    return {name: data for name, (data, _) in load_all().items()}


def list_targets() -> list[str]:
    """Every known target name, sorted."""
    return sorted(load_raw())


def _chain(name: str, raw: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Walk the ``from:`` chain from ``name`` upward, nearest first."""
    chain: list[dict[str, Any]] = []
    seen: set[str] = set()
    current: str | None = name
    while current is not None:
        if current in seen:
            raise ManifestError(f"inheritance cycle in target {name!r} at {current!r}")
        if current not in raw:
            raise TargetNotFound(f"unknown target {current!r}; known: {', '.join(sorted(raw))}")
        seen.add(current)
        node = raw[current]
        chain.append(node)
        if len(chain) > _MAX_INHERITANCE_DEPTH:
            raise ManifestError(
                f"target {name!r} exceeds max inheritance depth {_MAX_INHERITANCE_DEPTH}")
        current = node.get("from")
    return chain


#: Fields a variant never inherits. `abstract` says "this manifest is a shared declaration", which
#: is a statement about the file it appears in -- inheriting it would make every variant of a base
#: unrunnable, which is the exact opposite of what a base is for.
NOT_INHERITED: frozenset[str] = frozenset({"abstract"})


def _flatten(name: str, raw: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Collapse a ``from:`` chain into a single mapping, nearest ancestor applied last."""
    flattened: dict[str, Any] = {}
    for node in reversed(_chain(name, raw)):
        flattened = merge(flattened, {k: v for k, v in node.items() if k not in NOT_INHERITED})
    # Only what the named manifest itself declares, never an ancestor's.
    for field in NOT_INHERITED:
        if field in raw[name]:
            flattened[field] = raw[name][field]
    return flattened


def resolve_target(ref: str, overrides: Sequence[str] = ()) -> Manifest:
    """Resolve a ref (plus optional ``key=value`` overrides) into a validated Manifest.

    Args:
        ref: A target reference, e.g. ``hindsight:matched``.
        overrides: Hydra-style ``key=value`` argv tokens.

    Returns:
        The fully-resolved, validated Manifest.

    Raises:
        RefError: On a malformed ref or override.
        TargetNotFound: When no manifest matches.
        ManifestError: When the resolved manifest is invalid.
    """
    canonical = parse_ref(ref).canonical
    entries = load_all()
    # _flatten raises TargetNotFound for an unknown name, so the lookup below is safe. The base_dir
    # is the NAMED target's own directory, never an ancestor's: only a source target declares a
    # binding, and it declares its own.
    flattened = _flatten(canonical, {name: data for name, (data, _) in entries.items()})
    return from_dict(apply_overrides(flattened, parse_overrides(overrides)),
                     base_dir=entries[canonical][1])


def origins() -> dict[str, Path]:
    """Which directory each known target was read from, for listings."""
    return {name: directory for name, (_, directory) in load_all().items()}


def required_secrets_for(target: Manifest) -> list[str]:
    """The secret env-vars this target needs to launch.

    Two sources, unioned. What the target's *providers* imply, via
    :func:`memrank.secrets.requirements.required_secrets` -- right for the engines memrank ships,
    where    `llm: {provider: anthropic}` genuinely means `ANTHROPIC_API_KEY`. And what the target
    *declares*, which is the only way to express a credential memrank could not have guessed: a
    user and a password, a client id and a tenant, or a token under a name no provider table
    contains.

    The declared side is keyed by the name memrank RESOLVES, not the variable the engine reads --
    those differ whenever a target states a rename, and the wallet knows only the former.
    """
    from memrank.secrets import requirements

    providers = requirements.keyed_providers(target.components)
    derived = requirements.required_secrets(
        target.adapter, embedder=providers.get("embedder"), llm=providers.get("llm"))
    return sorted(set(derived) | set(target.secrets))


def secret_status(target: Manifest) -> list[tuple[str, bool]]:
    """Each required secret paired with whether it currently resolves (env or wallet)."""
    from memrank import config

    return [(name, config.secret(name) is not None) for name in required_secrets_for(target)]


def checkout_status(target: Manifest) -> tuple[str, str | None] | None:
    """``(ref, path)`` for a link-bound target, ``path`` being ``None`` when it is not linked.

    ``None`` for every other target: an image-backed engine has no checkout, and a descriptor that
    states its own ``root`` has nothing this machine needs to answer. Shaped like
    :func:`secret_status` because the CLI renders the two the same way -- a requirement the target
    declares, and whether this machine currently satisfies it.
    """
    if target.binding is None or target.binding.link is None:
        return None
    return target.binding.link, target.binding.root
