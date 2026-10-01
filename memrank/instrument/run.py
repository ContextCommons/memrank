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
"""run -- the act: a system and an evaluation produce traces, then the measures are applied.

One verb, and one place where refusal happens. What the loop guarantees:

- it refuses BEFORE the first `prepare` when the run cannot be set up (`refusal.py`);
- every task yields exactly one trace per attempt, whether it succeeded or failed;
- it never stops on a task's failure;
- state is cleared where the evaluation's rule says, and the result records the rule that was
  in force and whether the clearing could be observed.

Scoring is not in here. The measures run after the traces exist, which is what lets a measure
thought of after the run be applied to the same traces (`memrank.measure`).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field

from memrank.core import Document, MemoryAdapter, Recall
from memrank.instrument import asking
from memrank.instrument.evaluation import Clearing, Evaluation
from memrank.instrument.measure import Measure, Value
from memrank.instrument.refusal import refusal
from memrank.instrument.result import EvaluationRecord, Result, SystemRecord, now
from memrank.instrument.system import Assistant, Model, Retriever, System, kind_of
from memrank.instrument.task import Task
from memrank.instrument.trace import Answered, Declared, Failure, Given, Trace


class Answerer(ABC):
    """Writes the answer from the task and what was recalled. A parameter of the run.

    Not one of the seven: when a system only recalls, something must write an answer before a
    measure can read one, and whose writer that is belongs to the person running (H1).
    """

    #: What this writer calls itself. Recorded verbatim in `Answered.produced_by`.
    name: str = "answerer"

    @abstractmethod
    def write(self, task: Task, recalled: Recall) -> str:
        """Return the answer to ``task`` from what the system recalled."""


@dataclass
class _Unit:
    """One stretch of the run between two clearings of the system's state."""

    id: str
    tasks: list[Task] = field(default_factory=list)

    def documents(self) -> list[Document]:
        """This unit's material, in first-seen order, deduplicated by document id."""
        seen: dict[str, Document] = {}
        for task in self.tasks:
            for document in task.context:
                seen.setdefault(document.id, document)
        return list(seen.values())


def _units(evaluation: Evaluation) -> list[_Unit]:
    """Split the tasks where the evaluation's clearing rule says state is cleared."""
    if evaluation.clearing is Clearing.AT_END:
        return [_Unit(id=evaluation.name, tasks=list(evaluation.tasks))]
    if evaluation.clearing is Clearing.PER_TASK:
        return [_Unit(id=task.id, tasks=[task]) for task in evaluation.tasks]
    units: dict[str, _Unit] = {}
    for task in evaluation.tasks:
        name = evaluation.group_of(task)
        units.setdefault(name, _Unit(id=name)).tasks.append(task)
    return list(units.values())


class _Clearing:
    """Clears the system's state at each unit boundary, and records what happened."""

    def __init__(self, system: System) -> None:
        self.system = system
        self.observed = False
        self.note: str | None = None

    def open(self, unit: _Unit) -> None:
        """Prepare the system for a unit. Raises; the caller records the failure per task."""
        if isinstance(self.system, MemoryAdapter):
            self.system.prepare(unit.id)

    def close(self) -> None:
        """Clear the system's state, recording a failure rather than ending the run."""
        if not isinstance(self.system, MemoryAdapter):
            return
        try:
            self.system.cleanup()
        except Exception as failure:  # recorded on the result, never swallowed and never fatal
            self.note = self.note or f"{type(failure).__name__}: {failure}"
            return
        self.observed = True


def _feed(system: System, unit: _Unit, timings: dict[str, float]) -> None:
    """Give the unit's material to a system that can be told things."""
    documents = unit.documents()
    if not documents or not isinstance(system, MemoryAdapter):
        return
    with asking.Timer(timings, "ingest"):
        system.ingest(documents)


def _ask(system: System, task: Task, k: int, timings: dict[str, float]
         ) -> tuple[Recall | None, Answered | None, str]:
    """Put one task to the system in its kind's own verb. Returns the step it reached."""
    if isinstance(system, MemoryAdapter):
        return asking.recall(system, task, k, timings), None, "retrieve"
    if isinstance(system, Retriever):
        return asking.rank(system, task, k, timings), None, "retrieve"
    if isinstance(system, (Model, Assistant)):
        return None, asking.answer_itself(system, task, timings), "complete"
    raise AssertionError("refusal() admitted a system of no kind")  # pragma: no cover


def _written(answerer: Answerer | None, task: Task, recalled: Recall | None,
             timings: dict[str, float]) -> Answered | None:
    """The answer the writer produced, when there is a writer and something to write from."""
    if answerer is None or recalled is None:
        return None
    with asking.Timer(timings, "answer"):
        text = answerer.write(task, recalled)
    return Answered(text=text, produced_by=answerer.name)


def _trace(task: Task, unit: _Unit, attempt: int, *, error: Failure | None = None,
           recalled: Recall | None = None, answered: Answered | None = None,
           timings: dict[str, float] | None = None, declared: Declared | None = None,
           started=None, finished=None) -> Trace:
    """One trace. Built in one place, so a failed task's row has the same shape as any other."""
    documents = unit.documents()
    return Trace(
        task=task.model_copy(update={"context": ()}), group=task.group, attempt=attempt,
        given=Given(document_ids=tuple(d.id for d in documents), count=len(documents)),
        recalled=recalled, answered=answered, timings_ms=dict(timings or {}),
        declared=declared or Declared(), error=error,
        started=started or now(), finished=finished or now())


def _one_task(system: System, task: Task, unit: _Unit, attempt: int, *,
              k: int, answerer: Answerer | None) -> Trace:
    """One task, one attempt, one trace -- including when it broke."""
    started = now()
    timings: dict[str, float] = {}
    step = "retrieve"
    try:
        recalled, answered, step = _ask(system, task, k, timings)
        answered = answered or _written(answerer, task, recalled, timings)
    except Exception as failure:
        return _trace(task, unit, attempt, started=started,
                      error=Failure(step=step, message=f"{type(failure).__name__}: {failure}"),
                      timings=timings)
    return _trace(task, unit, attempt, recalled=recalled, answered=answered, timings=timings,
                  declared=asking.declarations(system, unit.id), started=started)


def _run_unit(system: System, unit: _Unit, attempt: int, clearing: _Clearing, *,
              k: int, answerer: Answerer | None) -> list[Trace]:
    """One unit: open it, feed it, put every task to the system, then clear."""
    timings: dict[str, float] = {}
    step = "prepare"
    try:
        clearing.open(unit)
        step = "ingest"
        _feed(system, unit, timings)
    except Exception as failure:
        broke = Failure(step=step, message=f"{type(failure).__name__}: {failure}")
        return [_trace(task, unit, attempt, error=broke) for task in unit.tasks]
    traces = []
    for position, task in enumerate(unit.tasks):
        trace = _one_task(system, task, unit, attempt, k=k, answerer=answerer)
        if position == 0 and "ingest" in timings:
            # The unit's ingest belongs to the unit; it is recorded once, on the trace of the
            # task it ran for, so pooling it per step counts one ingest per unit.
            trace = trace.model_copy(update={"timings_ms": {**timings, **trace.timings_ms}})
        traces.append(trace)
    clearing.close()
    return traces


def apply_measures(measures: Sequence[Measure], traces: Sequence[Trace],
                   values: Sequence[Value] = ()) -> list[Value]:
    """Run measures over traces, in order, each seeing what the ones before produced."""
    produced = list(values)
    for m in measures:
        produced.extend(m.measure(traces, tuple(produced)))
    return produced


def _records(system: System, evaluation: Evaluation, clearing: _Clearing
             ) -> tuple[SystemRecord, EvaluationRecord]:
    declared = asking.declarations(system, evaluation.name)
    return (
        SystemRecord(name=type(system).__name__, kind=kind_of(system),
                     version=declared.version, address=system.address),
        EvaluationRecord(name=evaluation.name, version=evaluation.version,
                         clearing=evaluation.clearing.value,
                         task_count=len(evaluation.tasks),
                         cleared=clearing.observed, clearing_note=clearing.note),
    )


def run(system: System, evaluation: Evaluation, *, answerer: Answerer | None = None,
        k: int = asking.DEFAULT_K, attempts: int = 1) -> Result:
    """Measure one system with one evaluation, here, in this process.

    Returns a :class:`~memrank.instrument.result.Result`: either a refusal with no traces and a
    stated reason, or one trace per task per attempt plus the values the evaluation's measures
    produced over them.
    """
    started = now()
    reason = refusal(system, evaluation, answerer)
    if reason is not None:
        empty = _Clearing(system) if isinstance(system, System) else None
        return Result(
            system=SystemRecord(name=type(system).__name__,
                                kind=kind_of(system) if isinstance(system, System) else None,
                                address=getattr(system, "address", None)),
            evaluation=EvaluationRecord(name=evaluation.name, version=evaluation.version,
                                        clearing=evaluation.clearing.value,
                                        task_count=len(evaluation.tasks),
                                        cleared=bool(empty and empty.observed)),
            refusal=reason, started=started, finished=now())
    clearing = _Clearing(system)
    traces: list[Trace] = []
    for attempt in range(1, attempts + 1):
        for unit in _units(evaluation):
            traces.extend(_run_unit(system, unit, attempt, clearing, k=k, answerer=answerer))
    values = apply_measures(evaluation.measures, traces)
    system_record, evaluation_record = _records(system, evaluation, clearing)
    return Result(system=system_record, evaluation=evaluation_record, traces=tuple(traces),
                  values=tuple(values), started=started, finished=now())


def measure(result: Result, *measures: Measure) -> Result:
    """Apply more measures to a result's stored traces. Nothing is run again.

    Returns a new result carrying the values it already had plus the new ones, so measuring
    after the fact and measuring in the run produce the same values in the same order.
    """
    if result.refused:
        raise ValueError(
            f"{result.evaluation.name} refused and has no traces to measure: {result.refusal}")
    return result.model_copy(update={
        "values": tuple(apply_measures(measures, result.traces, result.values))})
