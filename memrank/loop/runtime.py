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
"""How long a run was going: its total evaluation runtime, counted across resumes (ATO-2698).

A run can be stopped and resumed, and the time it sat stopped is not the evaluation's. So each
``memrank run`` session appends a ``start`` line to ``sessions.jsonl`` in the run folder when it
begins and an ``end`` line, with how long it was going by the runner's own clock, when it stops
for any reason. The total is the sum over the sessions, the last one counted up to its last
graded answer.

It is exact or absent, never estimated: a ledger whose first session is not the one that created
the run (a run begun before this was recorded, or on another machine), or with a session that
never wrote its ``end`` (the process was killed), gives no total at all.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

#: The ledger's file name in a run folder.
SESSIONS_FILE = "sessions.jsonl"


def _ms(start: float, end: float) -> float:
    return max(0.0, (end - start) * 1000.0)


def _earlier_ms(path: Path) -> float | None:
    """The ended sessions' time, or None when the ledger cannot account for the whole run."""
    if not path.exists():
        return None
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
             if line.strip()]
    if not lines or not lines[0].get("created"):
        return None
    total, open_session = 0.0, False
    for line in lines:
        if line["event"] == "start":
            if open_session:
                return None  # a session that never recorded its end
            open_session = True
        else:
            total += float(line["active_ms"])
            open_session = False
    return None if open_session else total


@dataclass
class Session:
    """One ``memrank run`` session of a run: started on :func:`begin`, ended on :meth:`end`."""

    path: Path
    clock: Callable[[], float]
    started: float
    #: The earlier sessions' time, read before this one was added; None when not exact.
    earlier_ms: float | None

    def total_ms(self) -> float | None:
        """The run's total runtime up to now, or None when it cannot be known exactly."""
        if self.earlier_ms is None:
            return None
        return self.earlier_ms + _ms(self.started, self.clock())

    def end(self) -> None:
        """Record how long this session was going."""
        self._append({"event": "end", "active_ms": _ms(self.started, self.clock())})

    def _append(self, line: dict[str, object]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(line) + "\n")


def begin(folder: Path, *, created: bool, clock: Callable[[], float],
          started: float, settings: dict[str, int | bool] | None = None) -> Session:
    """Start a session in ``folder``'s ledger; ``created`` when this session created the run.

    ``started`` is the clock reading when the session began, taken before the run folder existed.
    ``settings`` are recorded on its ``start`` line: how this session ran, which a resume may
    change -- ``ingest_concurrency``, ``answer_concurrency``, ``judge_concurrency`` and
    ``questions_in_order`` (:meth:`memrank.loop.stages.Ceilings.settings`). No score depends on
    the judge's; answers depend on the others only for an agent that stores what it is asked.
    """
    path = folder / SESSIONS_FILE
    earlier = 0.0 if created else _earlier_ms(path)
    session = Session(path=path, clock=clock, started=started, earlier_ms=earlier)
    session._append({"event": "start", "created": created, **(settings or {})})
    return session
