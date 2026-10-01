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
"""Trace -- everything observed while one task ran.

Exactly one per task per attempt, always, including on failure: "what ran" is answerable even
when nothing worked, and a failure is a row with a reason rather than a gap in the table. A
trace interprets nothing. It is also the store a measure thought of after the run is applied
over, which is why it holds the provider's own payload and not just the text.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from memrank.contract import Recall
from memrank.instrument.task import Task

#: The trace fields a measure may declare it reads. A measure naming anything else is refused
#: before the run rather than raising halfway through one.
TRACE_FIELDS: frozenset[str] = frozenset({
    "given", "recalled", "answered", "timings_ms", "declared", "error",
})


class Given(BaseModel):
    """What was given to the system before the prompt."""

    model_config = ConfigDict(frozen=True)

    document_ids: tuple[str, ...] = ()
    count: int = 0


class Answered(BaseModel):
    """The answer, and who produced it: the system itself, or the answer writer."""

    model_config = ConfigDict(frozen=True)

    text: str
    #: ``"system"`` or the writer's name. Never invented.
    produced_by: str


class Declared(BaseModel):
    """What the system said about itself. Declared, never verified, and never the wall clock."""

    model_config = ConfigDict(frozen=True)

    version: str | None = None
    #: What a provider billed it, per `MemoryAdapter.token_metrics`. `None` means it declared
    #: nothing, which is not zero.
    tokens: dict[str, float | None] | None = None
    #: Samples in ms of time only the engine can see, per `MemoryAdapter.declared_latency`.
    engine_timings: dict[str, list[float]] | None = None
    #: `MemoryAdapter.state_fingerprint`, when the engine offers one.
    state: str | None = None


class Failure(BaseModel):
    """Why a task has no result: the step that broke and what it said."""

    model_config = ConfigDict(frozen=True)

    #: ``prepare`` | ``ingest`` | ``retrieve`` | ``answer`` | ``complete`` | ``respond`` | ``rank``
    step: str
    message: str


class Trace(BaseModel):
    """One task, one attempt, everything observed. Complete when the task finished or broke."""

    model_config = ConfigDict(frozen=True)

    #: The task as it was put, minus its context: what was given is recorded by id in
    #: `given`, and carrying the corpus again in every trace would multiply it by the group.
    #: The task travels with the trace so a measure thought of after the run -- which reads a
    #: saved result and nothing else -- can still see what a correct outcome looks like.
    task: Task
    group: str | None = None
    #: 1-based. `attempts=3` gives three traces per task, not one averaged.
    attempt: int = 1
    given: Given = Given()
    #: What one retrieve gave back: the documents in the order the system ranked them --
    #: the order IS the rank -- and whatever the system declared about the call. `None` when
    #: the system was never asked to recall (it broke first, or it is not a kind that
    #: recalls); a `Recall` with no documents is a system that recalled nothing, which is a
    #: different thing. The type is `memrank.contract.Recall`, the same one retrieve returns,
    #: so nothing is re-shaped between the call and the record of it.
    recalled: Recall | None = None
    answered: Answered | None = None
    #: memrank's own measurement, at memrank's own call boundary. Keys are steps
    #: (``ingest``, ``retrieve``, ``answer``). A group's ingest is recorded on the trace of the
    #: task it ran for, which is the group's first.
    timings_ms: dict[str, float] = Field(default_factory=dict)
    declared: Declared = Declared()
    error: Failure | None = None
    started: datetime
    finished: datetime

    @property
    def task_id(self) -> str:
        """The task this trace is of. One trace per task per attempt, always."""
        return self.task.id
