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
"""Putting one task to a system, and reading back what the system declares.

Each kind is asked in its own verb, because the kind is the base class. Nothing here decides
anything: it produces the observed halves of a trace and the timings memrank took at its own
call boundary.
"""

from __future__ import annotations

import time
from typing import Any

from memrank.core import MemoryAdapter, Recall
from memrank.instrument.system import Assistant, Model, Retriever, System
from memrank.instrument.task import Task
from memrank.instrument.trace import Answered, Declared

#: How many passages a run asks for when the caller states nothing.
DEFAULT_K = 10


class Timer:
    """Wall-clock milliseconds around one call, taken by memrank at its own boundary."""

    def __init__(self, timings: dict[str, float], step: str) -> None:
        self._timings = timings
        self._step = step

    def __enter__(self) -> Timer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self._timings[self._step] = (time.perf_counter() - self._start) * 1000.0


def recall(system: MemoryAdapter, task: Task, k: int, timings: dict[str, float]) -> Recall:
    """Ask a memory for what is relevant. What it gives back is what the trace records."""
    with Timer(timings, "retrieve"):
        return system.retrieve(
            task.prompt, k, task.group or task.id, task.metadata.get("query_timestamp"))


def rank(system: Retriever, task: Task, k: int, timings: dict[str, float]) -> Recall:
    """Ask a retriever to rank its own corpus. A retriever declares nothing about the call."""
    with Timer(timings, "retrieve"):
        documents = list(system.rank(task.prompt, k))
    return Recall(documents=documents)


def answer_itself(system: Model | Assistant, task: Task, timings: dict[str, float]) -> Answered:
    """Ask a system that answers for itself. The answer is the system's, and says so."""
    if isinstance(system, Model):
        with Timer(timings, "complete"):
            text = system.complete(task.prompt)
    else:
        with Timer(timings, "respond"):
            text = system.respond([{"role": "user", "content": task.prompt}])
    return Answered(text=text, produced_by="system")


def _non_empty(mapping: dict[str, Any] | None) -> bool:
    """A declaration that is all `None` (or empty) declared nothing, and is recorded as such."""
    return bool(mapping) and any(value is not None for value in (mapping or {}).values())


def declarations(system: System, group: str) -> Declared:
    """What the system says about itself. Declared, never verified, and never invented.

    Read through both spellings while the rename runs: the kind-neutral `declared_*` verbs on
    `System`, and the memory contract's own `token_metrics` / `declared_latency` /
    `state_fingerprint`, which the registered adapters already implement.
    """
    version = system.declared_version()
    if version is None:
        stated = getattr(type(system), "engine_version", None)
        version = stated if isinstance(stated, str) and stated != "unknown" else None
    tokens = system.declared_tokens()
    timings = system.declared_timings()
    state = system.declared_state()
    if isinstance(system, MemoryAdapter):
        tokens = tokens if _non_empty(tokens) else system.token_metrics()
        timings = timings if _non_empty(timings) else system.declared_latency()
        state = state if state is not None else system.state_fingerprint(group)
    return Declared(
        version=version,
        tokens=tokens if _non_empty(tokens) else None,
        engine_timings=timings if _non_empty(timings) else None,
        state=state,
    )
