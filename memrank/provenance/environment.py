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
"""The conditions a run was measured under -- the half of comparability that is not the question.

The interface model's first principle: quality compares across places, latency compares only
within like environments, and the interface never lies about which. That rule needs the
conditions to be a *value* someone can compare, not prose. What this replaces was
``receipt.host`` -- three strings, one of them ``"macOS-26.5.2-arm64-arm-64bit"``, which no
consumer could act on without matching substrings, and which nothing read.

**Recorded, never hashed.** ``config_hash`` defines a run's IDENTITY: two runs with the same hash
asked the same question. Environment must stay out of it, or the same question asked on a laptop
and in the cloud becomes two questions and the runs can never be compared at all -- the exact
opposite of what recording it is for. It lives beside the hash in the receipt, never inside it.

**Nothing here identifies a person or a machine**, deliberately. Hostname was the one field of
the old blob that did, and it is gone rather than carried and stripped: the leaderboard's
``public_provenance`` is an allowlist, and every surface that had to remember to drop one field
is a surface that would eventually forget. An Environment is publishable by construction. The
machine that produced a run is still identifiable from the run directory it lives in.

**Place is detected, never passed.** A cloud run executes this harness *inside* the Fargate task,
and ``--on`` is deliberately not forwarded there (a task that inherited it would submit another
task), so the container is never told where it is. ECS supplies its own markers; reading them
cannot drift from reality the way a flag threaded through three layers can.

Known gap: there is no tenancy field. Whether a task had a host to itself is not something
anything available here can answer honestly, and a field that is wrong half the time is worse
than an absent one. It belongs with the metric invariance classes, which are what will actually
consume it.
"""

from __future__ import annotations

import os
import platform
from dataclasses import asdict, dataclass
from typing import Any

#: Set by ECS in every task's environment. Both are checked: the first names the launch type and
#: is the direct answer, the second is present on Fargate platform 1.4+ regardless and catches a
#: task whose launch type is reported differently.
_ECS_MARKERS = ("AWS_EXECUTION_ENV", "ECS_CONTAINER_METADATA_URI_V4")

#: What ``platform.machine()`` calls an architecture, mapped to what an image index calls it --
#: the same normalisation :meth:`memrank.placement.local.LocalPlacement.daemon_platform` applies
#: to the daemon's answer, so a receipt and a purl spell one architecture one way.
_ARCH_ALIASES = {"aarch64": "arm64", "x86_64": "amd64", "AMD64": "amd64"}

CLOUD, LOCAL = "cloud", "local"


@dataclass(frozen=True)
class Environment:
    """Where and on what a run was measured. Every field is non-identifying.

    ``memory_bytes`` and ``cpu_count`` are what the OS reports to THIS process, which under a
    container is the container's view -- which is the honest number, because it is what the run
    actually had.
    """

    place: str
    arch: str
    os: str
    os_version: str
    python: str
    cpu_count: int | None
    memory_bytes: int | None

    def as_dict(self) -> dict[str, Any]:
        """The recorded form. Plain types only -- a receipt is JSON on disk."""
        return asdict(self)


def detect() -> Environment:
    """The environment this process is running in, asked of the machine rather than declared."""
    return Environment(
        place=CLOUD if any(os.environ.get(marker) for marker in _ECS_MARKERS) else LOCAL,
        arch=normalise_arch(platform.machine()),
        os=platform.system().lower(),
        os_version=platform.release(),
        python=platform.python_version(),
        cpu_count=os.cpu_count(),
        memory_bytes=_memory_bytes(),
    )


def normalise_arch(reported: str) -> str:
    """``aarch64`` and ``arm64`` are one architecture; record one spelling of it."""
    return _ARCH_ALIASES.get(reported, reported)


def _memory_bytes() -> int | None:
    """Total memory, or ``None`` where the platform will not say.

    ``None`` rather than 0: a run that did not learn its memory has not got zero of it, and a
    consumer comparing environments must be able to tell "unknown" from "tiny".
    """
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, ValueError, OSError):
        return None
