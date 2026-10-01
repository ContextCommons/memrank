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
"""The CLI's ears on the eval loop: terminal echoes and the run heartbeat.

`ConsoleObserver` narrates to the terminal exactly as the loop always has -- same
strings, same ``max(1, total // 10)`` throttle. `RunStatusObserver` adds the heartbeat:
the run's progress model, ``status.json`` updates, and the (env-gated) remote publish a
cloud task performs. Together they are the CLI's replacement for the module-global
"current run" -- one explicit object per run instead of process state.

Thread-safety: methods are called from worker threads (``workers>1``,
``judge_workers>1``). ``RunStatus`` and ``RunProgress`` carry their own locks, and the
observer serializes snapshots through local and remote publication so an older snapshot
cannot overwrite a newer one. ``planned`` runs before any pool starts.
"""
from __future__ import annotations

import threading

from memrank.evaluation.observer import EvalObserver, EvalPlan, _eval_plan, _progress_step
from memrank.runs import status as run_status
from memrank.runs.progress import RunProgress
from memrank.term import style


def _progress_from_plan(plan: EvalPlan) -> RunProgress:
    progress = RunProgress()
    progress.plan(units=plan.units, documents=plan.documents,
                  retrievals=plan.retrievals, judgements=plan.judgements)
    return progress


def _planned_progress(units, repeats, judge, shape=None) -> RunProgress:
    """The work model for one cell -- ``EvalPlan``'s counts as a ``RunProgress``."""
    return _progress_from_plan(_eval_plan(units, repeats, judge, shape))


class ConsoleObserver(EvalObserver):
    """Terminal narration only -- what `ops compare` wants: echoes without a run dir."""

    def __init__(self, *, verbose: bool = False) -> None:
        self._verbose = verbose

    def unit_started(self, *, index: int, total: int, documents: int) -> None:
        # Announced so long ingests (e.g. BEAM) are attributable.
        style.say(f"ingesting unit {index}/{total} ({documents} docs) ...")

    def unit_finished(self, *, label: str, index: int, total: int,
                      queries_done: int, queries_total: int) -> None:
        style.say(f"{label} unit {index}/{total} ({queries_done}/{queries_total} queries)")

    def item_done(self, stage: str, *, seconds: float, done: int = 0, total: int = 0,
                  label: str = "", pass_index: int = 0, passes: int = 1) -> None:
        if stage != "judge" and self._echo_due(done, total):
            self._say_item(stage, done=done, total=total, label=label,
                           pass_index=pass_index, passes=passes)

    def warning(self, message: str) -> None:
        style.warn(message)

    def _echo_due(self, done: int, total: int) -> bool:
        return done % _progress_step(total, self._verbose) == 0 or done == total

    def _say_item(self, stage: str, *, done: int, total: int, label: str,
                  pass_index: int, passes: int) -> None:
        if stage == "ingest":
            style.say(f"{label} ingested {done}/{total} docs")
        else:
            suffix = f" (pass {pass_index + 1}/{passes})" if passes > 1 else ""
            style.say(f"{label} retrieved {done}/{total} queries{suffix}")


class RunStatusObserver(ConsoleObserver):
    """ConsoleObserver plus the heartbeat -- the full recorded-run behavior.

    Publishing mirrors what the module-global ``note_progress`` did, event for event:
    record every item into the progress model, and on the same throttle as the echo,
    write the record into ``status.json`` (with the stage's coarse state) and offer it
    to the env-gated remote publisher. The judge stage publishes per query and never
    echoes, exactly as before.
    """

    def __init__(self, status: run_status.RunStatus, *, verbose: bool = False) -> None:
        super().__init__(verbose=verbose)
        self._status = status
        self._progress: RunProgress | None = None
        self._publish_lock = threading.Lock()

    def planned(self, plan: EvalPlan) -> None:
        self._progress = _progress_from_plan(plan)

    def unit_started(self, *, index: int, total: int, documents: int) -> None:
        super().unit_started(index=index, total=total, documents=documents)
        self._publish(stage="ingest", unit=index)

    def unit_finished(self, *, label: str, index: int, total: int,
                      queries_done: int, queries_total: int) -> None:
        super().unit_finished(label=label, index=index, total=total,
                              queries_done=queries_done, queries_total=queries_total)
        self._publish(stage="retrieve", unit=index)

    def stage_started(self, stage: str) -> None:
        self._publish(stage=stage)

    def item_done(self, stage: str, *, seconds: float, done: int = 0, total: int = 0,
                  label: str = "", pass_index: int = 0, passes: int = 1) -> None:
        if self._progress is not None:
            self._progress.record(stage, seconds=seconds)
        if stage == "judge":
            # Per query, not per batch: each one is five sequential LLM calls, so this is
            # the finest granularity that exists -- still far coarser than the 10s remote
            # publish throttle.
            self._publish(stage="judge")
            return
        if self._echo_due(done, total):
            self._say_item(stage, done=done, total=total, label=label,
                           pass_index=pass_index, passes=passes)
            self._publish(stage=stage)

    def _publish(self, *, stage: str | None = None, unit: int | None = None) -> None:
        if self._progress is None:
            return
        # Keep snapshot order through both destinations; locking only the status write
        # lets a delayed worker replace completed progress with its older snapshot.
        with self._publish_lock:
            if unit is not None:
                self._progress.enter_unit(unit)
            if stage is not None:
                # Announcing a stage makes it the current one before any of its work has
                # completed -- a stage entered but not yet advanced must not render 0/0.
                self._progress.enter_stage(stage)
            record = self._progress.as_dict()
            self._status.set_progress_record(
                record, state=run_status.stage_state(stage) if stage else None)
            run_status.publish_remote(record, run_id=self._status.data["run_id"])
