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
"""What a run has got to so far, kept as it goes, so its ending can say where it stopped.

An exception says what went wrong; it does not say that the run already has an id and a page,
that every question was answered, or that 221 answers failed because the agent's provider
refused them. The runner writes those facts here as they become true, and the ending
(:mod:`memrank.loop.ending`) reads them whichever way the run ends. Nothing here changes what
the run does.
"""

from __future__ import annotations

import threading
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from memrank.service.protocol import RunResult


class Stage(Enum):
    """Where the run is, in the order a run passes through them."""

    SETUP = "setup"  # signing in, reading the agent, creating the run
    CHECK = "check"  # starting the agent and the check that it keeps conversations apart
    STEPS = "steps"  # feeding and asking
    JUDGING = "judging"
    UPLOAD = "upload"


@dataclass
class RunTrace:
    """The facts an ending needs, written by the runner as each becomes true."""

    stage: Stage = Stage.SETUP
    run_id: str | None = None
    url: str | None = None
    folder: Path | None = None
    #: The run as recorded once every step is done, before judging.
    recorded: RunResult | None = None
    #: The run as judged, before it is uploaded.
    judged: RunResult | None = None
    #: Failed steps seen in this session, by cause (:func:`memrank.loop.explain.cause`).
    failures: Counter[str] = field(default_factory=Counter)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def failed(self, cause: str) -> None:
        """Count one failed step; called from whichever lane's worker carried it."""
        with self._lock:
            self.failures[cause] += 1
