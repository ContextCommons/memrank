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
"""Where this install came from -- and, when it is too old, where a newer one comes from.

``memrank version`` printing ``0.2.0`` and nothing else is useless the moment there is a
second user: the version in ``pyproject.toml`` moves rarely and the install branch moves
daily, so two people on different commits report the same string and a bug report cannot be
tied to code. This is the same failure :func:`memrank._installed_version` was written to
prevent, one level up -- there the version disagreed with the install, here the version
cannot distinguish two installs.

PEP 610 answers it without any build machinery: pip and uv both write ``direct_url.json``
beside the installed metadata, recording the URL a distribution was installed from and, for a
VCS install, the *resolved* commit. Standard metadata is deliberate -- uv's own
``uv-receipt.toml`` is private to uv and records the ref that was *requested* (``@dev``),
which is precisely the thing that cannot identify a build.

A distribution installed from a plain wheel or sdist has no ``direct_url.json`` at all; that
is a fact about the install, not an error, and it reads back as "no origin to report".
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from memrank import __version__

#: What a refusal composed somewhere else may say about upgrading. The hosted platform knows
#: that a caller's CLI is stale and nothing whatever about how that CLI was installed, so it
#: asks for the upgrade in these words and :func:`with_upgrade_instruction` fills in the how
#: on the machine that can answer. Shared as a constant because it is a seam between two
#: modules, not prose either side may reword.
UPGRADE_REQUEST = "upgrade the memrank CLI"

#: The published distribution, and what a registry upgrade names.
_DISTRIBUTION = "memrank"

#: uv keeps tool environments under ``<data dir>/uv/tools``, or under ``UV_TOOL_DIR`` when it
#: is set. Two consecutive path segments is the whole test: nothing else installs a
#: distribution beneath a directory named ``tools`` inside one named ``uv``.
_UV_TOOL_SEGMENTS = ("uv", "tools")

#: PEP 610 names the file; the spec fixes it, so it is not configurable.
_DIRECT_URL = "direct_url.json"

#: Enough of a commit to identify it in a bug report, short enough to read in a version line.
_SHORT_COMMIT = 7

#: The commit length receipts have always carried in ``memrank_version``. Kept distinct from
#: :data:`_SHORT_COMMIT` because published artifacts are compared against each other and
#: shortening the field would make an old receipt and a new one of the same build differ.
_COMMIT_IN_VERSION = 8

#: A hung ``git`` must not hold up a run; two seconds is the value receipt.py has always used.
_GIT_TIMEOUT_SECONDS = 2

#: The record the standalone build carries beside its interpreter: the commit it was built from,
#: its target, and the origin its install script is served from. Written by
#: ``tools/internal/binary/build.py``; the name is the seam between the two.
STANDALONE_RECORD = "memrank-build.json"


def standalone_build() -> dict[str, str] | None:
    """The standalone build's record of itself, or ``None`` when this is a Python install.

    The standalone build has no installer front end and no PEP 610 record worth reading -- its
    package metadata was copied in at build time and names the build machine's directory. What
    identifies it is the commit it was frozen from, and what upgrades it is its install script.
    A frozen process without the record is a broken build and fails loudly here.
    """
    if not getattr(sys, "frozen", False):
        return None
    record = Path(sys._MEIPASS) / STANDALONE_RECORD  # type: ignore[attr-defined]
    return dict(json.loads(record.read_text(encoding="utf-8")))


def _reinstall_command(build: dict[str, str]) -> str:
    """The standalone build updates and repairs itself the one way it was installed."""
    return f"curl -fsSL {build['origin']}/install.sh | sh"


def _direct_url(distribution: str) -> dict[str, Any] | None:
    """The distribution's PEP 610 record, or ``None`` when it has none.

    Absent for a wheel/sdist install and for a source checkout that was never installed --
    both ordinary. A malformed record is *not* ordinary and raises: corrupt install metadata
    is worth a loud failure, not a quietly emptier version line.
    """
    from importlib.metadata import Distribution, PackageNotFoundError

    try:
        raw = Distribution.from_name(distribution).read_text(_DIRECT_URL)
    except PackageNotFoundError:
        return None
    return None if raw is None else json.loads(raw)


def describe_origin(distribution: str = "memrank") -> str | None:
    """One phrase naming where ``distribution`` was installed from, or ``None``.

    Three shapes, because PEP 610 records three: a VCS install carries ``vcs_info`` and is
    reported by resolved commit; an editable install carries ``dir_info.editable`` and is
    reported as the working tree it tracks, since its code is whatever is on disk right now;
    anything else is reported by URL. The standalone build is reported by its own record.
    """
    build = standalone_build()
    if build is not None:
        return f"standalone {build['target']}, commit {build['commit'][:_SHORT_COMMIT]}"
    record = _direct_url(distribution)
    if record is None:
        return None
    url = record.get("url", "")
    vcs = record.get("vcs_info")
    if vcs:
        return _vcs_origin(url, vcs)
    if record.get("dir_info", {}).get("editable"):
        return f"editable, {url}"
    return url or None


def _vcs_origin(url: str, vcs: dict[str, Any]) -> str:
    """``git+<url>@<ref>, commit <sha>`` -- the ref for orientation, the commit for truth."""
    ref = vcs.get("requested_revision")
    location = f"{vcs['vcs']}+{url}@{ref}" if ref else f"{vcs['vcs']}+{url}"
    commit = vcs.get("commit_id", "")
    return f"{location}, commit {commit[:_SHORT_COMMIT]}" if commit else location


def head_commit(directory: str | Path | None = None) -> str | None:
    """HEAD of the checkout at ``directory``, or ``None`` when there is no answer.

    ``None`` covers every ordinary way this has no answer -- no ``git`` on PATH, the path is
    not a checkout, the path does not exist -- because none of them is a failure of the run
    being recorded. What it must never do is answer about some *other* repository, so callers
    are responsible for passing a directory they can vouch for; ``None`` means the process's
    working directory and is only correct for a caller that knows it runs inside this repo.
    """
    command = ["git", "rev-parse", "HEAD"]
    if directory is not None:
        command[1:1] = ["-C", str(directory)]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False,
                                timeout=_GIT_TIMEOUT_SECONDS)
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _editable_tree(url: str) -> Path | None:
    """The working tree a ``file://`` editable install points at."""
    parsed = urlparse(url)
    return Path(unquote(parsed.path)) if parsed.scheme == "file" and parsed.path else None


def build_commit(distribution: str = "memrank") -> str | None:
    """The commit of the memrank build in use, or ``None`` when no commit identifies it.

    Two installs can name a commit and one cannot. A VCS install carries the resolved commit
    in its own PEP 610 record -- no ``git`` call, and correct wherever the process runs. An
    editable install's code is whatever is in the tree it tracks, so that tree's HEAD is the
    answer, and the tree is the one the record names rather than wherever the caller happens
    to be standing. A wheel or sdist install has no commit at all: its version is its identity
    and ``None`` is the honest answer, where a commit read out of the caller's own repository
    would be a plausible lie in the artifact that exists to be reproducible. The standalone build
    was frozen from one commit and carries it.
    """
    build = standalone_build()
    if build is not None:
        return build["commit"]
    record = _direct_url(distribution)
    if record is None:
        return None
    vcs = record.get("vcs_info")
    if vcs:
        return vcs.get("commit_id") or None
    if record.get("dir_info", {}).get("editable"):
        tree = _editable_tree(record.get("url", ""))
        return head_commit(tree) if tree else None
    return None


def build_identity(distribution: str = "memrank") -> str:
    """What a receipt records as ``memrank_version``: the version, and a commit when there is one."""
    commit = build_commit(distribution)
    return f"{__version__}+{commit[:_COMMIT_IN_VERSION]}" if commit else __version__


def _installed_as_uv_tool(distribution: str) -> bool:
    """Whether ``distribution`` lives in a uv-managed tool environment.

    Which front end installed a distribution is not recorded anywhere -- PEP 610 describes
    where the code came from, never who fetched it -- but where the code *sits* is recorded by
    the filesystem, and a uv tool sits somewhere nothing else does. It matters because the two
    upgrade paths are not interchangeable: ``pip install --upgrade`` inside a uv tool
    environment upgrades a copy the ``memrank`` on PATH does not run, and ``uv tool upgrade``
    is not available to someone who never used uv.
    """
    from importlib.metadata import Distribution, PackageNotFoundError

    try:
        location = Distribution.from_name(distribution).locate_file("")
    except PackageNotFoundError:
        return False
    path = Path(str(location))
    root = os.environ.get("UV_TOOL_DIR")
    if root:
        return path.is_relative_to(Path(root))
    parts = path.parts
    return any(parts[i:i + 2] == _UV_TOOL_SEGMENTS for i in range(len(parts) - 1))


def _install_source(record: dict[str, Any] | None, distribution: str) -> str:
    """What names this install's source to an installer, for a command that re-runs it.

    A VCS install is named by the URL and ref it recorded -- the caller's own ref, never one
    named here, because no instruction of ours may move somebody off the branch they chose.
    Anything else is named by the distribution, which is what a registry resolves.
    """
    vcs = (record or {}).get("vcs_info")
    if not vcs:
        return distribution
    ref = vcs.get("requested_revision")
    url = (record or {}).get("url", "")
    return f"{vcs['vcs']}+{url}@{ref}" if ref else f"{vcs['vcs']}+{url}"


def _upgrade_from_source(source: str, distribution: str) -> str:
    """Upgrade this install in place, whether its source is a git URL or the registry.

    A git source needs ``--refresh``/``--force-reinstall`` and a registry source does not: a
    git *branch* resolves against a cache, so both installers otherwise report nothing to
    upgrade while the branch has moved (astral-sh/uv#4317, #9146, #14684). A released version
    is a new version and resolves on its own.
    """
    from_registry = source == distribution
    if _installed_as_uv_tool(distribution):
        if from_registry:
            return f"uv tool upgrade {distribution}"
        return f"uv tool install --force --refresh {source}"
    if from_registry:
        return f"pip install --upgrade {distribution}"
    return f"pip install --upgrade --force-reinstall {source}"


def dependency_instruction(package: str, extra: str | None = None,
                           distribution: str = _DISTRIBUTION) -> str:
    """How to get ``package`` into THIS install, for a refusal about a missing dependency.

    Three installs again, and the reason to read the metadata rather than name both shapes is
    the same one that made naming only ``uv sync`` wrong: an instruction that does not fit the
    reader's install is a second dead end stacked on the first. An editable install is a
    checkout and syncs; a tool environment takes the extra alongside the source it was
    installed from; anything else is a plain install of the package, or of the extra that
    carries it. The standalone build is fixed at build time: a missing package it should carry
    means it is damaged and reinstalls, and an extra it does not carry is not one it can gain.
    """
    build = standalone_build()
    if build is not None:
        if extra:
            return (f"a Python install of memrank, pip install '{distribution}[{extra}]' -- "
                    f"this standalone build does not carry it")
        return _reinstall_command(build)
    record = _direct_url(distribution)
    if record is not None and record.get("dir_info", {}).get("editable"):
        return f"uv sync --extra {extra}" if extra else "uv sync"
    source = _install_source(record, distribution)
    if _installed_as_uv_tool(distribution):
        return f"uv tool install --force --with {package} {source}"
    if extra:
        return f"pip install --upgrade '{distribution}[{extra}]'"
    return f"pip install --force-reinstall {source}"


def upgrade_instruction(distribution: str = _DISTRIBUTION) -> str:
    """How to upgrade *this* install -- the only install anyone here can speak for.

    Three shapes, and the difference between them is the point: telling someone who installed
    a release to run a git install replaces their release with a branch checkout, and telling
    someone whose install is editable to reinstall anything is telling them to do nothing at
    all, since their code is the tree on disk. An editable install therefore gets prose rather
    than a command; the working tree is the thing to move and only its owner knows how. The
    standalone build upgrades by re-running the script that installed it.
    """
    build = standalone_build()
    if build is not None:
        return _reinstall_command(build)
    record = _direct_url(distribution)
    if record is not None and record.get("dir_info", {}).get("editable"):
        tree = _editable_tree(record.get("url", ""))
        where = f" at {tree}" if tree else ""
        return f"this install is editable: update the working tree it tracks{where}"
    return _upgrade_from_source(_install_source(record, distribution), distribution)


def with_upgrade_instruction(message: str) -> str:
    """A refusal from elsewhere, answered with the upgrade that fits the install reading it.

    A no-op on every message that did not ask, so a caller can route all of them through it
    rather than deciding per code which refusals are about version skew.
    """
    if UPGRADE_REQUEST not in message:
        return message
    return f"{message}: {upgrade_instruction()}"


def version_line() -> str:
    """What ``memrank version`` and ``memrank --version`` both print, so they cannot differ."""
    origin = describe_origin()
    return f"{__version__}  ({origin})" if origin else __version__
