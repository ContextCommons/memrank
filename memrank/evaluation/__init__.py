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
"""Deprecated: the in-process eval library behind `memrank.evaluation.api.run` and `memrank
submit`. Evaluate an agent with `memrank run` instead; see the README. `constants` and
`result` are used by the current path (`loop/judge`, `runs/registry`) and do not warn.

The eval library: run one (adapter x benchmark) cell and hand back its result.

What belongs here: the measurement loop (`cell`), what an engine may declare about
itself and what memrank measures for it (`measurement`), the aggregation into the
artifact dict (`aggregate`), the judge stage (`judge_stage`), the provenance receipt
(`receipt`), and the observer contract the loop narrates through (`observer`).

What does not: terminals, run directories, heartbeats, placement, cloud, Typer.
Nothing under `memrank.evaluation` may import `memrank.cli`, `memrank.placement`,
`memrank.accounts`, or `memrank.runs.registry` -- the run lifecycle wires those
concerns in from outside, through an `EvalObserver`. (`runs.status` and `term.style`
are a TEMPORARY exception carried by `observer._GlobalsObserver`, the pre-split
default; it leaves with the CLI split.)
"""
from memrank.evaluation.api import run
from memrank.evaluation.case import Answer, CaseRow, Passage, Subscore
from memrank.evaluation.cell import cell_applicable, run_cell
from memrank.evaluation.observer import EvalObserver, EvalPlan
from memrank.evaluation.result import EvalResult
from memrank.evaluation.score import Score

__all__ = ["Answer", "CaseRow", "EvalObserver", "EvalPlan", "EvalResult", "Passage",
           "Score", "Subscore", "cell_applicable", "run", "run_cell"]
