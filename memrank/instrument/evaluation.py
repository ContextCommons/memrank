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
"""Evaluation -- a named, versioned bundle: tasks, the measures it ships with, the rule
for when the system's state is cleared, and the verb that applies it to a system.

The thing a person brings, reuses or publishes. memrank's evaluations and private ones are the
same kind of thing, which is why `memrank.evaluation("demo")` returns exactly what a person
writes by hand.

`evaluation.run(system=...)` is how an evaluation is applied. The verb lives on the evaluation
because that is where a reader holding one goes looking for it, and because the import path
then carries the model: a free function taking both objects says nothing about which argument
is which, which is the confusion it was observed to cause (decision 0008).

Scoring is not here. An evaluation bundles measures; the measures decide.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from memrank import deprecation
from memrank.instrument.asking import DEFAULT_K
from memrank.instrument.measure import Measure
from memrank.instrument.task import Task

if TYPE_CHECKING:  # the run imports this module, so the verb's types cannot be imported here
    from memrank.instrument.result import Result
    from memrank.instrument.run import Answerer
    from memrank.instrument.system import System

#: What an evaluation states instead of a version when its material cannot be frozen. The same
#: distinction `memrank.quality.UNFREEZABLE` draws for a benchmark's dataset: a gap and an
#: honest statement are different values.
UNFREEZABLE = "cannot-be-frozen"


class Clearing(str, Enum):
    """When the system's state is cleared. Closed, so two runs can be compared on it."""

    PER_TASK = "per-task"
    PER_GROUP = "per-group"
    AT_END = "at-end"


class Evaluation(BaseModel):
    """Tasks, measures, and a clearing rule, under a name and a version."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    name: str
    #: The version of what this evaluation asks, or :data:`UNFREEZABLE` when it cannot say.
    version: str
    tasks: tuple[Task, ...] = ()
    #: The measures it ships with. A person may run others over the traces afterwards.
    measures: tuple[Measure, ...] = ()
    clearing: Clearing = Clearing.PER_GROUP
    #: Kept out of the way of the seven: what this evaluation wants to record about itself.
    metadata: dict[str, object] = Field(default_factory=dict)

    def run(self, system: System, *, answerer: Answerer | None = None,
            k: int = DEFAULT_K, attempts: int = 1) -> Result:
        """Apply this evaluation to ``system``, here, in this process.

        Returns a :class:`~memrank.instrument.result.Result`: either a refusal with no traces
        and a stated reason, or one trace per task per attempt plus the values this
        evaluation's measures produced over them.

            from memrank.evaluations import Demo
            from memrank.systems import TFIDF

            result = Demo().run(system=TFIDF())

        ``answerer`` writes the answer for a system that only recalls; ``k`` is how many
        documents to ask a memory or a retriever for; ``attempts`` repeats every task. Each
        default is spelled out rather than hidden behind ``None``, so the signature a reader
        prints is the contract.
        """
        # Imported here and not at module scope: `run` imports this module for `Evaluation` and
        # `Clearing`, so a module-level import would be a cycle. The verb is the only thing in
        # this module that needs it. `DEFAULT_K` comes from `asking`, which imports neither.
        from memrank.instrument.run import run as _run

        deprecation.warn("evaluation.run(system=...)")
        return _run(system, self, answerer=answerer, k=k, attempts=attempts)

    def group_of(self, task: Task) -> str:
        """The unit of state a task belongs to. A task with no group is its own."""
        return task.group or task.id

    def groups(self) -> list[str]:
        """The groups, in the order their first task appears."""
        seen: list[str] = []
        for task in self.tasks:
            name = self.group_of(task)
            if name not in seen:
                seen.append(name)
        return seen
