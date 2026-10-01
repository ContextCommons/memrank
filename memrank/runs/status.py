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
"""Per-run progress heartbeat -- so `memrank ps` can see how a running eval is going.

Each ``memrank submit`` writes/updates a ``runs/<run-id>/status.json`` heartbeat as it progresses.
A run is one target x one benchmark: a comma sweep is N runs, each with its own heartbeat, executed
sequentially by one process (the not-yet-started ones sit in ``queued``). This is engine-agnostic:
it tracks the *eval script* (units ingested/retrieved), not how the engine executes (docker
container, fetched binary, or in-process baseline). The eval loop reaches this file through
`memrank.orchestration.observers.RunStatusObserver` -- one explicit object per run, handed to
`run_cell` -- never through module state.

Liveness in :func:`active_runs` is a fact-check (``os.kill(pid, 0)`` + heartbeat freshness), not a
timing-based wait.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from memrank.atomic_json import write_json

STATUS_FILE = "status.json"
_STALE_SECONDS = 90  # a run whose heartbeat is older than this (and no live pid) is "stale"
_TERMINAL = ("done", "failed")

#: What a run reports when nothing recorded its state. Never displayed as terminal -- a run
#: folder with results but no heartbeat predates the heartbeat, and inferring "done" from the
#: presence of files is exactly the lie the interface must not tell.
UNKNOWN_STATE = "unknown"

#: The platform's state names in the vocabulary :func:`classify` produces. The translation
#: lives beside the vocabulary it targets, because one condition spelled two ways in one
#: column is the same defect as a wrong value: a reader comparing rows would take `done` and
#: `stopped-success` for different things.
PLATFORM_STATES = {
    "submitted": "submitted",
    "running": "running",
    "stopped-success": "done",
    "stopped-failed": "failed",
    # A local agent run that stopped before it finished; `memrank run --resume` continues it.
    "stopped-incomplete": "incomplete",
    "unknown": UNKNOWN_STATE,
}


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class RunStatus:
    """A live status heartbeat for one run, persisted to ``<run_dir>/status.json``."""

    run_dir: Path
    data: dict[str, Any] = field(default_factory=dict)

    #: Held across mutate-then-write, so a heartbeat published from one eval worker can never
    #: serialise a record another worker is halfway through updating. Reentrant because the
    #: progress path mutates and then calls :meth:`update`, which takes it again.
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False, compare=False)

    @classmethod
    def create(cls, run_dir: Path, *, target: str, benchmark: str,
               slice_: str | None, eval_ref: str | None = None) -> RunStatus:
        """Create and persist the initial ``initializing`` status for one target x evaluation.

        ``eval_ref`` names WHICH evaluation -- `beam:100k-smoke`, not `beam`. Without it `runs ls`
        showed two runs of different evaluations as identical rows, distinguishable only by
        decoding the run id.
        """
        now = _utcnow()
        existing = read(run_dir) or {}
        preserved = {key: existing[key] for key in (
            "experiment_id", "experiment_label", "plan_hash", "experiment_spec", "log_run"
        ) if key in existing}
        rs = cls(run_dir, {
            "run_id": run_dir.name, "pid": os.getpid(), "target": target,
            "benchmark": benchmark, "eval_ref": eval_ref or benchmark,
            "slice": slice_, "state": "initializing",
            "progress": {}, "message": "", "started_at": now, "updated_at": now, "error": None,
            **preserved,
        })
        rs._write()
        return rs

    def update(self, *, state: str | None = None, message: str | None = None,
               error: str | None = None, **progress: Any) -> None:
        """Update state/message/progress and stamp the heartbeat (``updated_at``)."""
        with self._lock:
            if state is not None:
                self.data["state"] = state
            if message is not None:
                self.data["message"] = message
            if error is not None:
                self.data["error"] = error
            if progress:
                self.data.setdefault("progress", {}).update(progress)
            self.data["updated_at"] = _utcnow()
            self._write()

    def mark_remote(self, **facts: Any) -> None:
        """Record that this run executes elsewhere, with the handles needed to find it.

        Top-level rather than under ``progress``: :func:`classify` reads ``pid`` and ``placement``
        to decide whether a local liveness check even applies, and ``memrank ps``/``status`` read
        the task handles. Burying them in the progress dict would leave both looking at nothing.

        The pid is cleared deliberately. A remote run's process id belongs to a container on
        another machine; keeping it would invite an ``os.kill`` check that is meaningless at best
        and a collision with an unrelated local process at worst.
        """
        with self._lock:
            self.data["pid"] = None
            self.data.update(facts)
            self.data["updated_at"] = _utcnow()
            self._write()

    def annotate(self, **facts: Any) -> None:
        """Record top-level facts (e.g. ``log_run``) without touching pid or state.

        Like :meth:`mark_remote`, these live top-level rather than under ``progress`` because
        readers (``memrank logs``) look for them there; unlike it, the run still executes here,
        so the pid stays.
        """
        with self._lock:
            self.data.update(facts)
            self.data["updated_at"] = _utcnow()
            self._write()

    def set_progress_record(self, record: dict[str, Any], *, state: str | None) -> None:
        """Replace the whole progress record and stamp the heartbeat, as one atomic step.

        The record is assigned rather than merged (see :func:`note_progress`), and the assignment
        plus the write happen under one lock hold -- otherwise a second worker's record can land
        between this one's assignment and its write, publishing a state that belongs to neither.
        """
        with self._lock:
            self.data["progress"] = record
            self.update(state=state)

    def _write(self) -> None:
        with self._lock:
            write_json(self.run_dir / STATUS_FILE, self.data)


# The module-global "current run" (`_CURRENT`/`_PROGRESS`, `note_progress`) is gone: the
# run lifecycle hands `memrank.orchestration.observers.RunStatusObserver` to the eval loop
# instead, and everything those globals did lives there -- one explicit object per run.


def stage_state(stage: str) -> str:
    """The coarse run state a stage corresponds to, for `ps`'s STATE column."""
    return {"ingest": "ingesting", "retrieve": "retrieving",
            "judge": "judging"}.get(stage, stage)


#: How often a cloud run's progress is published. Every S3 write is a billed PutObject and the
#: watcher polls at 15s, so anything finer buys nothing a poll-path viewer could see. It is only
#: the STARTING interval now: the ingest route answers each POST with the interval to use next
#: (`memrank/api/run_progress.py`), so the server can slow a fleet down without a new harness.
REMOTE_PUBLISH_INTERVAL_S = 10.0

#: How long to wait on the progress POST. Short and deliberate: this call sits between two units
#: of real evaluation work, so a slow API must cost a stale bar rather than the run's throughput.
_POST_TIMEOUT_S = 5.0

_last_published = 0.0
_interval_s = REMOTE_PUBLISH_INTERVAL_S
_sequence = 0
#: Set once the server has said this run is no longer running -- see `_post_progress`. A stopped
#: task keeps evaluating until ECS kills it, and there is nobody left to show its bar to.
_reporting = True


def publish_remote(record: dict[str, Any], *, run_id: str) -> None:
    """Publish the progress record when this process IS a cloud run, on the server's interval.

    TWO destinations, and the split is deliberate rather than transitional.

    S3 is the floor. The task holds ``s3:PutObject`` on ``cloud-runs/*`` and that grant already
    exists, so the record lands whether or not an API is reachable, whether or not the deployment
    wires an event transport, and whether or not this build knows about the ingest route. Every
    reader that polls -- `ps`, `watch` against an older server -- keeps working from it alone.

    The POST is the live path. It carries the same record to
    :mod:`memrank.api.run_progress`, which is the only thing that publishes to the event bus, so
    the browser and a streaming `watch` see a sample in about a second instead of on the next
    poll. The task holds a RUN TOKEN for this and nothing else: it names one run, expires, and
    reaches exactly one route. It is emphatically not a bus credential -- the reasons are in
    :mod:`memrank.api.run_tokens`, and the short version is that a producer able to write the
    stream directly is a producer able to forge any event on it.

    BEST-EFFORT BY DESIGN, and the only deliberate swallow in this path: a progress write that
    fails must not fail an evaluation that is otherwise fine. Losing a progress update costs a
    stale bar for one interval; raising here would discard a run that has done real work. The
    caller's own failures still propagate -- this catches nothing but its own publishing.
    """
    global _last_published
    bucket = os.environ.get("MEMRANK_PROGRESS_BUCKET")
    url = os.environ.get("MEMRANK_PROGRESS_URL")
    token = os.environ.get("MEMRANK_RUN_TOKEN")
    if not bucket and not (url and token):
        return
    now = time.monotonic()
    if now - _last_published < _interval_s:
        return
    _last_published = now
    if bucket:
        try:
            from memrank.runs.artifacts import put_progress

            put_progress(bucket=bucket, run_id=run_id, record=record)
        except Exception as exc:  # noqa: BLE001 - see the docstring; a stale bar beats a lost run
            logging.getLogger(__name__).debug("progress publish failed: %s", exc)
    if url and token and _reporting:
        _post_progress(url, token, record)


def _post_progress(url: str, token: str, record: dict[str, Any]) -> None:
    """Report one sample to the ingest route and obey what it answers.

    The response is a control channel, so this reads it rather than discarding it: an interval
    lets the server throttle every running task at once, and a state that is not ``running`` says
    this run has been killed or already observed to have stopped. Both are things only the server
    knows, arriving on a connection that had to be made anyway.

    Note what is NOT sent: no run id, no org id. The route derives both from the token, so there
    is nothing here for a wrong value to be read from.
    """
    global _interval_s, _reporting, _sequence
    _sequence += 1
    try:
        import httpx

        answer = httpx.post(url, timeout=_POST_TIMEOUT_S,
                            headers={"Authorization": f"Bearer {token}"},
                            json={"progress": record, "sequence": _sequence})
        if answer.status_code >= 400:
            # Logged, not raised, and not retried: the S3 record above already landed, so the
            # bar is at worst one interval stale. A 401 here is a token that expired under a
            # long run, which nothing this process can do anything about.
            logging.getLogger(__name__).debug("progress POST refused: %s %s",
                                              answer.status_code, answer.text[:200])
            return
        body = answer.json()
        interval = float(body.get("interval_s", _interval_s))
        state = body.get("state", "running")
    except Exception as exc:  # noqa: BLE001 - see `_publish_remote`; a stale bar beats a lost run
        logging.getLogger(__name__).debug("progress POST failed: %s", exc)
        return
    if state not in ("submitted", "running") or interval <= 0:
        _reporting = False
        logging.getLogger(__name__).debug("server reports run is %s; stopping progress posts",
                                          state)
        return
    # Floored at the built-in interval rather than taken as given: the server may slow this task
    # down, but a bad answer must not turn a progress channel into a request loop.
    _interval_s = max(interval, REMOTE_PUBLISH_INTERVAL_S)


# --- reading / classifying (for `memrank ps`) ------------------------------------------------- #
def read(run_dir: Path) -> dict[str, Any] | None:
    """Parse ``<run_dir>/status.json``, or ``None`` if the run has no status file."""
    path = run_dir / STATUS_FILE
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def pid_alive(pid: int | None) -> bool:
    """Whether ``pid`` names a live process (a fact check, not a wait)."""
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists but owned by another user
    return True


def classify(data: dict[str, Any]) -> str:
    """Return ``queued`` | ``running`` | ``done`` | ``failed`` | ``stale`` | ``unknown``.

    ``unknown`` is reserved for a record this machine genuinely cannot read: a cloud run whose
    heartbeat nobody is restamping. Everything else is a claim about a process here.
    """
    state = data.get("state")
    if state in _TERMINAL:
        return state
    # A queued sibling's heartbeat is deliberately NOT stamped while an earlier target of the
    # same sweep executes, so freshness would misread it as stale mid-sweep. The pid is the fact
    # that decides: the process that will start it is alive, or nobody will.
    if state == "queued":
        return "queued" if pid_alive(data.get("pid")) else "stale"
    # A pid means the run executes HERE, and then it is the whole answer -- alive is running,
    # dead is stale -- exactly as it already is for a queued sibling above. Liveness used to be
    # consulted only to convict: a live pid fell through to the heartbeat check below and any
    # phase quieter than 90 seconds was reported dead anyway. A judged run spends ~15 minutes
    # in LLM calls, so `watch` announced a healthy run as a corpse and `runs show` froze at the
    # stage before it (2026-08-04). Establishing a fact and then discarding it is not a check.
    #
    # A run executing elsewhere has no pid to check -- `mark_remote` clears it precisely so that
    # none of this applies. Its pid, if it had one, would belong to a process inside a Fargate
    # task: os.kill against it locally is meaningless at best, and at worst matches an unrelated
    # local process and reports a finished run as running.
    pid = data.get("pid")
    if pid is not None:
        return "running" if pid_alive(pid) else "stale"
    # No pid, so the only evidence left is the heartbeat's age -- which for a cloud run is written
    # by `watch`, an optional attachment. An unwatched task goes cold in 90 seconds while running
    # perfectly well, and calling that stale reported a fleet of healthy runs as corpses the first
    # time the platform was unreachable (2026-08-03). We do not know, and the platform record is
    # what does; say so rather than guessing terminally. A local record with no pid at all is a
    # heartbeat nobody claimed, and stale remains the honest reading of it.
    cold = UNKNOWN_STATE if data.get("placement") == "cloud" else "stale"
    updated = data.get("updated_at")
    if not isinstance(updated, str):
        return cold
    try:
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(updated)).total_seconds()
    except (TypeError, ValueError):
        # ValueError: not a timestamp at all. TypeError: a NAIVE one, which cannot be
        # subtracted from an aware now() -- every heartbeat this code writes is aware, but a
        # hand-edited or pre-timezone record would otherwise take the whole listing down with
        # it. An unusable timestamp is the same answer either way.
        return cold
    return "running" if age <= _STALE_SECONDS else cold


def active_runs(runs_root: Path) -> list[dict[str, Any]]:
    """Every run's status under ``runs_root``, newest first, each annotated with a ``status`` field."""
    if not runs_root.exists():
        return []
    out: list[dict[str, Any]] = []
    for run_dir in runs_root.iterdir():
        data = read(run_dir) if run_dir.is_dir() else None
        if data is not None:
            data["status"] = classify(data)
            out.append(data)
    out.sort(key=lambda d: d.get("started_at", ""), reverse=True)
    return out
