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
"""Result -- what a run produced, in the form a person reads.

The traces, the values (each carrying its measure's name and its decider), and the record of
which system and which evaluation version. Never a bare number, and never a verdict: a result
does not say "better".

`print(result)` is the whole of the read: `__str__` lays out what is above, and `__repr__`
stays pydantic's, so a debugger still shows the fields. Nothing in an example formats a result
itself -- a person who copies a line out of one gets the same lines memrank prints.

This is the READ form of the NEW run. It is not the stored artifact the CLI and the cloud
read -- `memrank.evaluation.result.EvalResult` is still that, unchanged, and
`memrank.evaluation.api.run` still produces it.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from memrank.instrument.measure import Value
from memrank.instrument.trace import Trace

#: Bumped when a stored result's field names change. A reader can tell what it is holding.
SCHEMA_VERSION = "instrument-1"


def now() -> datetime:
    """The one clock the run reads, in UTC, so a stored result is unambiguous."""
    return datetime.now(timezone.utc)


#: How wide a measure's name is padded to, so the values below it line up as a column.
_MEASURE_WIDTH = 20


def _shown(value: float | bool | None) -> str:
    """A value as a reader reads it: three decimals for a number, otherwise what it is."""
    return f"{value:.3f}" if isinstance(value, float) else repr(value)


def _value_lines(value: Value) -> list[str]:
    """One value: its measure, its number, who decided it, and what it is about."""
    where = value.task_id or value.group or "whole run"
    lines = [f"  {value.measure:<{_MEASURE_WIDTH}} {_shown(value.value):>7}  "
             f"decided by {value.decider.value:<7} [{where}]"]
    if value.why:
        lines.append(f"    {value.why}")
    return lines


class SystemRecord(BaseModel):
    """Which system ran, and what memrank could establish about it. Nothing is invented."""

    model_config = ConfigDict(frozen=True)

    #: The class memrank observed.
    name: str
    #: `memory`, `model`, `retriever`, `assistant` -- or `None` when the class declared none.
    kind: str | None = None
    #: What the system said its version is. `None` is "did not state", never "0".
    version: str | None = None
    #: Where it was reached. `None` means in-process.
    address: str | None = None


class EvaluationRecord(BaseModel):
    """Which evaluation ran, at which version, under which clearing rule."""

    model_config = ConfigDict(frozen=True)

    name: str
    #: The version, or the evaluation's own statement that its material cannot be frozen.
    version: str
    #: The rule actually in force for this run.
    clearing: str
    task_count: int = 0
    #: Whether the system's own clearing was observed to run, not whether it was asserted.
    cleared: bool = False
    #: Why clearing could not be observed, when it could not. `None` when nothing went wrong.
    clearing_note: str | None = None


class Result(BaseModel):
    """One run's whole record: a refusal, or the traces and the values."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = SCHEMA_VERSION
    system: SystemRecord
    evaluation: EvaluationRecord
    #: Why the run refused, stated before the system was touched. When set, `traces` is empty.
    refusal: str | None = None
    traces: tuple[Trace, ...] = ()
    values: tuple[Value, ...] = ()
    started: datetime
    finished: datetime

    @property
    def refused(self) -> bool:
        return self.refusal is not None

    def __str__(self) -> str:
        """The result as a person reads it: which run, every value, then the traces."""
        header = [
            f"system:     {self.system.name} ({self.system.kind}), "
            f"version {self.system.version}",
            f"evaluation: {self.evaluation.name} at {self.evaluation.version}, "
            f"{self.evaluation.task_count} task(s), cleared {self.evaluation.clearing}",
        ]
        if self.refused:
            return "\n".join(
                [*header, f"REFUSED before the system was touched: {self.refusal}"])
        values = [line for value in self.values for line in _value_lines(value)]
        broken = sum(1 for trace in self.traces if trace.error is not None)
        traces = (f"traces:     {len(self.traces)} recorded, {broken} with errors "
                  f"-- one per task per attempt, always")
        return "\n".join([*header, "", *values, "", traces])

    def values_of(self, measure: str) -> tuple[Value, ...]:
        """Every value one measure produced. Values are named; this is how a reader picks one."""
        return tuple(v for v in self.values if v.measure == measure)

    def traces_of(self, task_id: str) -> tuple[Trace, ...]:
        """Every attempt's trace for one task -- where a reader digs when a value is low."""
        return tuple(t for t in self.traces if t.task_id == task_id)

    def save(self, path: str | Path) -> Path:
        """Write this result as JSON. The traces persist, typed, and can be measured again."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> Result:
        """Read a saved result back. Refuses a schema it does not know, rather than guessing."""
        stored = json.loads(Path(path).read_text(encoding="utf-8"))
        found = stored.get("schema_version")
        if found != SCHEMA_VERSION:
            raise ValueError(
                f"{path} carries schema_version {found!r}; this memrank reads "
                f"{SCHEMA_VERSION!r}. It was written by another version of the instrument.")
        return cls.model_validate(stored)
