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
"""What happened to each unit -- the record a run keeps whether or not the unit worked.

A cell used to end at the first unit that raised: the run was marked failed and every unit
already measured was discarded, however many hours it had cost. The unit failure RATE is
itself a signal -- an engine that drops one unit in fifty is telling us something a run that
merely dies does not -- so every unit now records an outcome and the failures are counted.

The modelling follows the judge stage's precedent one level up: `_judge_one_query` records an
unparseable verdict and skips that query, and only "nothing graded at all" aborts. Here a
failed unit contributes no `per_unit` score row and no `per_query` drill rows, so it is
excluded from judging and from the composite's denominator by construction rather than by a
filter somewhere -- and `units_total`/`units_failed` beside the composite are what say so.
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

#: The two outcomes a unit can have. A unit that was never attempted has no record at all --
#: `fail_fast` and the fatal conditions stop the loop, and what did not run is not reported.
OK = "ok"
FAILED = "failed"


class UnitStageError(Exception):
    """Internal: an exception raised inside one unit, tagged with the stage it came from.

    Raised by :func:`stage_guard` and never escapes the eval loop -- the loop either turns it
    into a :class:`UnitOutcome` or re-raises the original cause. It carries the cause rather
    than replacing it so that the fatal paths (``fail_fast``, ``RateLimitExhausted``) raise the
    exception the caller would have seen before this existed.
    """

    def __init__(self, stage: str, cause: BaseException) -> None:
        super().__init__(f"{stage}: {type(cause).__name__}: {cause}")
        self.stage = stage
        self.cause = cause


@contextmanager
def stage_guard(stage: str) -> Iterator[None]:
    """Tag any ``Exception`` raised in the block with the stage that raised it.

    ``BaseException`` that is not an ``Exception`` -- ``KeyboardInterrupt`` above all -- passes
    through untagged and therefore un-survivable: an operator interrupting a run is not one
    unit failing, and a loop that recorded it as one would swallow the interrupt fifty times.
    """
    try:
        yield
    except UnitStageError:
        raise                      # already tagged by an inner guard; the inner stage is the true one
    except Exception as exc:
        raise UnitStageError(stage, exc) from exc


@dataclass(frozen=True)
class UnitOutcome:
    """One unit's fate: that it ran, or where and how it did not.

    ``error`` is the exception's CLASS NAME and ``message`` its text. They are separate fields
    because they are not equally publishable: a class name is ours, while a message quotes
    whatever the engine or the dataset said and belongs with the judge rationales that the
    public-safe surfaces already drop.
    """

    unit_id: str
    outcome: str
    stage: str | None = None
    error: str | None = None
    message: str | None = None

    @classmethod
    def ok(cls, unit_id: str) -> UnitOutcome:
        return cls(unit_id=unit_id, outcome=OK)

    @classmethod
    def failed(cls, unit_id: str, *, stage: str, cause: BaseException) -> UnitOutcome:
        return cls(unit_id=unit_id, outcome=FAILED, stage=stage,
                   error=type(cause).__name__, message=str(cause))

    @property
    def is_ok(self) -> bool:
        return self.outcome == OK

    def to_dict(self) -> dict[str, Any]:
        """The artifact form. A successful unit carries two keys, not three nulls."""
        if self.is_ok:
            return {"unit_id": self.unit_id, "outcome": self.outcome}
        return {"unit_id": self.unit_id, "outcome": self.outcome, "stage": self.stage,
                "error": self.error, "message": self.message}


def failure_summary(outcomes: list[UnitOutcome]) -> dict[str, Any]:
    """The counts that travel beside the composite, so nobody reads it over the wrong n.

    ``unit_failure_rate`` is ``None`` for a cell with no units rather than 0.0 -- the same
    None-vs-0.0 rule the composite follows, since a rate over nothing is not a rate of zero.
    """
    failed = sum(1 for o in outcomes if not o.is_ok)
    return {
        "units_total": len(outcomes),
        "units_failed": failed,
        "unit_failure_rate": (failed / len(outcomes)) if outcomes else None,
    }
