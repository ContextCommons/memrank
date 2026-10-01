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
"""How the eval loop narrates -- the observer contract, and silence as the default.

`EvalObserver` is the loop's ONLY outward voice: no terminal, no heartbeat file, no
globals. The loop calls it and nothing else, so what a run reports is exactly what its
caller wired in -- the CLI's terminal + heartbeat implementations live in
`memrank.orchestration.observers`, and a library caller who wired nothing gets
`NULL_OBSERVER`'s nothing.
"""
from __future__ import annotations

from dataclasses import dataclass

from memrank.judging.judge import JudgeConfig
from memrank.judging.shape import GENERIC_BINARY_CATEGORIES, BinaryJudgeShape, JudgeShape


def _progress_step(n: int, verbose: bool = False) -> int:
    """Emit at most ~10 progress lines for ``n`` items (every item when ``n`` is small,
    or every item under ``verbose``)."""
    return 1 if verbose else max(1, n // 10)

@dataclass(frozen=True)
class EvalPlan:
    """What ``run_cell`` is about to do, sized before any of it happens.

    Judging is sized by the SHAPE's judgeability, the same predicate that decides what
    ``_apply_judge`` actually grades and what ``assert_judge_coverage`` gates on -- so a
    progress bar's denominator cannot disagree with the work. ``retrievals`` is multiplied
    by ``repeats``; ``judgements`` is not: extra passes are retrieval only, and each query
    is judged once.
    """
    units: int
    documents: int
    retrievals: int
    judgements: int
    adapter: str = ""
    benchmark: str = ""


def _eval_plan(units, repeats: int, judge: JudgeConfig | None,
               shape: JudgeShape | None = None, *,
               adapter: str = "", benchmark: str = "") -> EvalPlan:
    if shape is None:
        shape = BinaryJudgeShape(GENERIC_BINARY_CATEGORIES)
    return EvalPlan(
        units=len(units),
        documents=sum(len(u.documents) for u in units),
        retrievals=sum(len(u.queries) for u in units) * repeats,
        judgements=0 if judge is None else
        sum(1 for u in units for q in u.queries if shape.is_judgeable(q)),
        adapter=adapter, benchmark=benchmark)


class EvalObserver:
    """How the eval loop reports what it is doing -- every method a no-op.

    The loop calls these and nothing else: no terminal, no heartbeat file, no globals. A
    caller that wants narration subclasses what it cares about; the default is silence.

    One object threaded through the loop rather than loose callbacks or module state --
    ``runs.status`` records how per-call wiring decayed last time -- and a base class rather
    than a Protocol so a new event is not a breaking change. Methods MAY be called from
    worker threads (``workers>1``, ``judge_workers>1``); implementations must be
    thread-safe.
    """

    def planned(self, plan: EvalPlan) -> None:
        """The cell's totals, before the first document -- a bar can size itself now."""

    def unit_started(self, *, index: int, total: int, documents: int) -> None:
        """A unit's ingest begins (sequential path only, as before the seam)."""

    def unit_finished(self, *, label: str, index: int, total: int,
                      queries_done: int, queries_total: int) -> None:
        """A unit's retrieval is done (sequential path only)."""

    def stage_started(self, stage: str) -> None:
        """A stage is entered before any of its items complete (currently: judge)."""

    def item_done(self, stage: str, *, seconds: float, done: int = 0, total: int = 0,
                  label: str = "", pass_index: int = 0, passes: int = 1) -> None:
        """One document ingested / query retrieved / query judged, with its measured
        seconds. Called for EVERY item -- throttling an echo is the observer's business."""

    def warning(self, message: str) -> None:
        """Something survivable but worth a human's eye (rate-limit pause, unjudged query)."""


#: The library default: a run nobody is listening to says nothing, anywhere.
NULL_OBSERVER = EvalObserver()
