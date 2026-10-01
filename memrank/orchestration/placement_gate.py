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
"""Where a run executes, validated and provisioned: placements and their guards.
"""
from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import typer

from memrank.evaluation.cell import AdapterFactory

PLACEMENTS: tuple[str, ...] = ("none", "local", "cloud")
# Fargate sizing. A stack target runs engine + datastore + embedder sidecars in one task; an
# in-process arm is only the harness. Same split scripts/cloud-run.sh applies.
_CLOUD_SIZE_STACK = ("4096", "8192")
_CLOUD_SIZE_IN_PROCESS = ("1024", "2048")


def _require_placement_ready(where: str, *, target=None) -> None:
    """Refuse a placement this machine cannot use, before any run is minted.

    Asked of the placement class rather than restated here: the same ``check()`` rows are what
    ``provision`` enforces on, so the refusal a user reads and the failure that would have
    happened are one code path. There is deliberately no `doctor` command wrapping this -- the
    check belongs to the run that needs it.
    """
    from memrank.placement.base import require
    if where != "local":
        return
    if target is not None and target.binding is not None:
        from memrank.placement.workspace import WorkspacePlacement

        assert target is not None
        require(WorkspacePlacement(target=target).check())
        return
    from memrank.placement.local import LocalPlacement

    require(LocalPlacement(run_id="preflight").check())


def placement_for(target, where: str, *, run_dir: Path):
    """The placement for ONE target, as a context manager.

    Chosen per target rather than per run: an in-process arm never needs provisioning whatever
    ``--on`` says, and a sweep gives each stack target its own disposable project, torn down before
    the next begins.
    """
    from memrank.placement.inprocess import InProcessPlacement

    if where not in PLACEMENTS:
        raise typer.BadParameter(f"unknown --on {where!r}; known: {', '.join(PLACEMENTS)}")
    if where == "cloud":
        # Unreachable: `run` submits and returns long before the placement loop. Asserted rather
        # than assumed, because the in-process short-circuit below would otherwise silently run a
        # cloud-flagged baseline on this machine.
        raise AssertionError("--on cloud is a submission, handled before the placement loop")
    if where == "none" or target.kind == "in-process":
        return InProcessPlacement()
    if target.binding is not None:
        from memrank.placement.workspace import WorkspacePlacement

        return WorkspacePlacement(target=target, log_path=run_dir / "engine.log")
    from memrank.placement.local import LocalPlacement

    # No registry and no tag: a local run learns both from the target's own container graph, so a
    # fresh clone can run every stack target without exporting anything first.
    return LocalPlacement(run_id=run_dir.name)


def _validate_source_target(*, explicit_on: str | None, on: str,
                            factories: list[tuple[str, AdapterFactory]],
                            overrides: list[str]) -> tuple[str, list[Any]]:
    """Choose local placement when any target is source-bound, or refuse an incoherent placement.

    A sweep may contain source targets. `placement_for` is a per-target context manager and a sweep
    tears each target down before the next begins, so two source targets never hold a port at the
    same time -- which is what the blanket refusal here used to guard against. Comparing an engine
    you just wrote a translator for against the catalog is the whole point of having one, and
    refusing it protected nothing: quality compares across all of them, and latency is already kept
    honest by the declared transport class rather than by which targets ran together.

    What stays is the placement coupling. A source target is launched from a local directory, so it
    cannot run in the cloud or against an already-running endpoint -- and that refusal IS the
    reproducibility gate, not friction.
    """
    from memrank.targets import resolve_target

    targets = [resolve_target(ref, overrides) for ref, _ in factories]
    source_targets = [target for target in targets if target.binding is not None]
    if not source_targets:
        return on, []
    if explicit_on is not None and explicit_on != "local":
        named = ", ".join(repr(target.name) for target in source_targets)
        raise typer.BadParameter(f"source-bound target(s) {named} require --on local")
    return "local", source_targets


def _source_digest_guard(source_targets: list[Any], submitted: str | None) -> str:
    """Prove the background process re-read the definitions the foreground did.

    One digest over EVERY source target in the sweep, in submission order: covering only the first
    would leave the rest free to be edited between submission and execution.

    Args:
        source_targets: The resolved source-bound targets, in submission order.
        submitted: The digest the submitting parent computed, or ``None`` in the parent itself.

    Returns:
        The digest to pass to the background process.

    Raises:
        typer.BadParameter: When a definition changed after submission.
    """
    from memrank.targets.manifest import definition_digest

    resolved = definition_digest(*source_targets)
    if submitted is not None and submitted != resolved:
        names = ", ".join(repr(target.name) for target in source_targets)
        raise typer.BadParameter(f"source target(s) {names} changed after submission; submit again")
    return resolved


@contextmanager
def _environment_patch(values: dict[str, str]) -> Iterator[None]:
    """Apply placement facts only for the cell that placement provisioned."""
    absent = object()
    previous = {key: os.environ.get(key, absent) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is absent:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(value)
