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
"""Placement for targets that have nowhere to be placed."""

from __future__ import annotations

from types import TracebackType

from memrank.placement.base import Endpoint, Requirement
from memrank.targets.manifest import Manifest


class InProcessPlacement:
    """A no-op for ``kind: in-process`` targets -- baseline and the control arms.

    They run inside the harness, so there is no container to start, no port to discover, and nothing
    to tear down. Having a placement for them at all keeps the caller uniform: every target is
    provisioned the same way, and only this class knows the answer is "nothing to do".
    """

    def check(self) -> list[Requirement]:
        """Nothing is required: the harness is already running, which is the whole placement."""
        return []

    def provision(self, target: Manifest) -> Endpoint:
        """Provision nothing, and say nothing about where the target is.

        Two cases share this: a ``kind: in-process`` target, which runs inside the harness; and
        ``--on none`` against a stack whose engine someone already started (the pre-M5 default,
        where the adapter finds it at its configured URL).
        """
        return Endpoint()

    def teardown(self) -> None:
        return None

    def __enter__(self) -> InProcessPlacement:
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None,
                 tb: TracebackType | None) -> None:
        self.teardown()
