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
"""Which targets a machine other than this one could resolve.

A cloud task runs the platform harness, which carries only the manifests shipped inside the package.
Everything else in this machine's catalog -- the ``targets.path`` directories and the operator's own
``${MEMRANK_CONFIG_DIR}/targets`` -- exists on this laptop and nowhere else, so a ref that resolves
here can be unresolvable there.

It cost a Fargate task to learn: ``mem0:memory-arena`` (defined in a sibling repo reached through
``targets.path``) was submitted to the cloud, launched, and the container exited 1 on its first line
with ``unknown target 'mem0:memory-arena'``. Nothing was misconfigured and no newer image would have
helped -- the definition simply is not there.

This module answers the question and renders nothing: the CLI and the API refuse in different
shapes, and a guard that raised would force one of them to catch its own control flow.
"""
from __future__ import annotations

from pathlib import Path

from memrank.targets import catalog
from memrank.targets.resolve import parse_ref

#: How many ``from:`` links to follow before giving up. Deliberately the catalog's own limit: this
#: walk visits exactly the chain ``_flatten`` merges, and a cycle is that function's error to raise.
_MAX_DEPTH = catalog._MAX_INHERITANCE_DEPTH


def _chain_origins(ref: str, raw: dict[str, dict], where: dict[str, Path]) -> list[tuple[str, Path]]:
    """Every manifest ``ref`` is built from, nearest first, with the directory each came from.

    The ancestors matter as much as the ref. A builtin variant may inherit from a locally-defined
    base, and the ref alone would look shipped.
    """
    out: list[tuple[str, Path]] = []
    current: str | None = parse_ref(ref).canonical
    seen: set[str] = set()
    while current is not None and current in raw and current not in seen and len(out) <= _MAX_DEPTH:
        seen.add(current)
        out.append((current, where[current]))
        current = raw[current].get("from")
    return out


def local_only(refs: list[str]) -> dict[str, Path]:
    """The refs among ``refs`` that only this machine can resolve, and where each comes from.

    Args:
        refs: Target refs as the user typed them.

    Returns:
        Ref -> the non-builtin directory holding the manifest that makes it unportable, which is the
        ref's own when it is locally defined and an ancestor's when it inherits from one. Refs that
        do not resolve at all are absent: an unknown ref is the resolver's error to report, with its
        list of what IS known, and duplicating that here would answer a different question badly.
    """
    where = catalog.origins()
    raw = catalog.load_raw()
    builtin = catalog.builtin_dir()
    found: dict[str, Path] = {}
    for ref in refs:
        for _, directory in _chain_origins(ref, raw, where):
            if directory != builtin:
                found[ref] = directory
                break
    return found


def unportable_message(local: dict[str, Path]) -> str:
    """Why these refs cannot be submitted to the cloud, and the two ways out.

    Names the directory rather than only the ref, because the ref looks perfectly ordinary in
    ``targets ls`` -- the parenthesised origin is the only thing that distinguishes it, and a user
    who has not read the precedence rules has no reason to have noticed.
    """
    named = ", ".join(f"{ref!r} ({directory})" for ref, directory in sorted(local.items()))
    subject = "these targets are" if len(local) > 1 else "this target is"
    them = "them" if len(local) > 1 else "it"
    # NOT "move the manifest into memrank/targets/builtin/", which this used to advise. Some
    # targets live outside the package ON PURPOSE -- `mem0:voyage` and `mem0:bge-tei` were moved
    # there deliberately on 2026-08-19 -- so telling their author to move them back prescribes
    # undoing a decision rather than describing a limit.
    return (f"{subject} defined only on this machine: {named}. A cloud task runs the platform "
            f"harness, which carries only the targets shipped with memrank, so it cannot resolve "
            f"{them}. Run {them} with --on local. A target belongs in the package only if it is "
            f"memrank's to ship; one that is yours stays yours, and stays local.")
