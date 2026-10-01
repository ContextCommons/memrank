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
"""Your own measure, applied to a run that already happened.

    uv run python examples/04-your-own-measure/run.py

A measure is a named rule from traces to values, and it declares three things
before it runs: its scope, which trace fields it reads, and who decides.
Scoring sits outside the run loop, so a measure thought of afterwards runs
over stored traces and the system is never touched.
"""

import tempfile
from collections.abc import Sequence
from pathlib import Path

import memrank
from memrank import Decider, Measure, Result, Scope, Trace, Value


class PassagesRecalled(Measure):
    """How many passages came back for each question."""

    name = "passages-recalled"
    scope = Scope.TASK
    # Both are declared, and both are checked before the run rather than
    # during it. RULE means a fixed rule decided: nobody judged, no clock.
    reads = ("recalled",)
    decider = Decider.RULE

    def measure(self, traces: Sequence[Trace],
                values: Sequence[Value]) -> list[Value]:
        """Returns one value per trace."""
        measured = []
        for trace in traces:
            recalled = trace.recalled
            measured.append(Value(
                measure=self.name,
                decider=self.decider,
                task_id=trace.task_id,
                value=float(len(recalled.documents)) if recalled else None,
                why=f"of the {trace.given.count} passage(s) the run gave it",
            ))
        return measured


system = memrank.system("word-overlap")
evaluation = memrank.evaluation("demo")
passages_recalled = PassagesRecalled()

with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "run.json"
    result = evaluation.run(system=system)
    result.save(path)
    # A different process, days later, starts exactly here.
    stored = Result.load(path)

print(memrank.measure(stored, passages_recalled))
