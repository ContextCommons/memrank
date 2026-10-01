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
"""Measure -- a named rule from traces to values, and the values it produces.

A measure declares three things before it runs: its scope, which trace fields and value names
it reads, and who decides. A measure that reads a name nothing produces is refused before the
run rather than raising in the middle of one.

The shipped measures live in `memrank.instrument.measures`. They are ordinary measures: a judge
is a measure whose decider is a model, the word-match proxy is one whose decider is a rule, and
latency and failure rate are ones whose decider is memrank's own clock and bookkeeping.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from enum import Enum

from pydantic import BaseModel, ConfigDict

from memrank.instrument.trace import Trace


class Scope(str, Enum):
    """What one value of this measure is about."""

    #: One value per trace.
    TASK = "task"
    #: One value (or one per group) for the whole run.
    RUN = "run"


class Decider(str, Enum):
    """Who decided a value. Carried on every value, so no number is anonymous."""

    #: memrank's own clock and bookkeeping.
    MEMRANK = "memrank"
    #: A fixed rule -- string matching, a structural check.
    RULE = "rule"
    #: A model adjudicated it.
    MODEL = "model"
    #: The system's own word, recorded as its word.
    SYSTEM = "system"


class Value(BaseModel):
    """One named value. Never a bare number: it carries its measure and its decider."""

    model_config = ConfigDict(frozen=True)

    #: The measure that produced it.
    measure: str
    decider: Decider
    #: The task this value is about. `None` for a run-scope value.
    task_id: str | None = None
    #: The group this value is about, when a measure works at a group's granularity -- a
    #: benchmark's own `score()` does, because its unit IS the group. `None` otherwise.
    group: str | None = None
    value: float | bool | None = None
    #: Why, when the measure can say: a rationale, a matched span, a reason it is `None`.
    why: str | None = None


class Measure(ABC):
    """A named rule from traces to values. Subclass it and declare what it reads."""

    #: The name every value of this measure carries. Also the value name other measures read.
    name: str = "measure"
    scope: Scope = Scope.TASK
    #: Trace fields (see `memrank.instrument.trace.TRACE_FIELDS`) and other measures' names.
    #: A measure that combines others says so here.
    reads: tuple[str, ...] = ()
    decider: Decider = Decider.RULE

    @abstractmethod
    def measure(self, traces: Sequence[Trace], values: Sequence[Value]) -> list[Value]:
        """Read the traces and whatever has been measured so far; produce values.

        ``values`` is what earlier measures produced in this pass, so a measure that declares
        it reads another's name gets it. It is never mutated.
        """


def value_names(measures: Sequence[Measure]) -> frozenset[str]:
    """Every value name a set of measures will produce."""
    return frozenset(m.name for m in measures)
