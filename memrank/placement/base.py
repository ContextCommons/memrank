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
"""The placement contract: bring a target up, hand back where it is, tear it down."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import TracebackType
from typing import Any, Protocol

from memrank.errors import MemrankError
from memrank.targets.manifest import Manifest


class PlacementError(MemrankError):
    """A target could not be provisioned."""


class CloudRenderError(MemrankError):
    """A target cannot be rendered into a task definition.

    Lives here rather than in :mod:`memrank.placement.cloud` so the Compose-to-ECS compiler can
    raise it without importing the renderer that calls it.
    """


@dataclass(frozen=True)
class Requirement:
    """One precondition a placement needs, and what was actually found.

    Reported rather than raised, so a caller can name EVERY missing precondition in one pass
    instead of failing at the first in resolution order. The knowledge belongs to the placement
    that needs it, and the same rows are what enforcement raises on, so a diagnosis cannot drift
    from the thing it diagnoses.
    """

    name: str
    ok: bool
    detail: str
    fix: str = ""


def require(requirements: list[Requirement]) -> None:
    """Raise unless every requirement is met, naming all of them and how to fix each.

    All, not the first: failing in resolution order tells a user to install one thing, then the
    next, which is the round-trip a single honest refusal removes.

    Raises:
        PlacementError: If any requirement is unmet. A ``MemrankError``, so the CLI boundary
            prints it verbatim -- an unmet precondition is the user's machine, not a memrank bug,
            and it must never reach the "internal error" path that says otherwise.
    """
    missing = [row for row in requirements if not row.ok]
    if not missing:
        return
    lines = [f"{row.name}: {row.detail}" + (f" -- {row.fix}" if row.fix else "")
             for row in missing]
    raise PlacementError("this machine cannot run here.\n  " + "\n  ".join(lines))


@dataclass(frozen=True)
class Endpoint:
    """Where a provisioned target is reachable, and what was actually run.

    ``base_url`` is ``None`` for in-process targets, which have no address at all.
    ``adapter_env`` is what the adapter needs in order to talk to this instance -- the discovered
    host and port, never a guessed one. ``image_digests`` records what was really pulled, so the
    receipt can say more than a tag.
    """

    base_url: str | None = None
    adapter_env: dict[str, str] = field(default_factory=dict)
    image_digests: dict[str, str] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)


class Placement(Protocol):
    """Provision a target somewhere, then always tear it down.

    Implementations are context managers so teardown survives an exception. That is deliberate:
    ``scripts/local-eval.sh`` used a shell ``EXIT`` trap which reads a ``local`` from ``main()``, so
    under ``set -u`` a failed ``compose up`` died with "project: unbound variable" and hid the real
    docker error -- losing both the containers and the reason.
    """

    def check(self) -> list[Requirement]:
        """Report what this placement needs and whether it has it. NEVER raises.

        The half every placement implements, including ones that materialize by submitting
        rather than provisioning. ``provision`` calls it first, so what a user is told and what
        actually stops a run are the same rows.
        """
        ...

    def provision(self, target: Manifest) -> Endpoint:
        """Bring the target up and return where it is reachable."""
        ...

    def teardown(self) -> None:
        """Remove everything provisioned. Must be safe to call twice."""
        ...

    def __enter__(self) -> Placement:
        ...

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None,
                 tb: TracebackType | None) -> None:
        ...
