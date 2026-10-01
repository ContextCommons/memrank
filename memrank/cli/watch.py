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
"""Deprecated: the hidden `memrank watch`/`kill` commands that follow `submit` runs. Evaluate an
agent with `memrank run` instead; see the README.

`memrank watch` / `kill` -- the lifecycle verbs of the detached-by-default model.

Every submission returns immediately with run ids; blocking is a VERB, not a flag (the
interface model's rejected shapes: `--wait`, `--detach`). `watch <id...>` attaches until the
runs end -- late, again, or from CI -- polling a whole sweep TOGETHER rather than serialising
it. A local run is watched through its heartbeat (:mod:`memrank.runs.status`); a cloud run
through the platform API, inheriting `--wait`'s old promise: on success its artifacts come
down and it becomes indistinguishable from a local run.

`kill <id>` ends a local background run: one process executes a whole sweep, so the signal
ends every sibling, and the record must say so for each of them.
"""
from __future__ import annotations

import json
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, NamedTuple

import typer

from memrank import (
    settings,
)
from memrank.placement import run_api_client
from memrank.runs import reconcile, registry
from memrank.runs import status as run_status
from memrank.term import key_input, progress, style

#: How long `watch` blocks before giving up: a long benchmark outruns any per-call cap, and
#: giving up is not failure -- the run keeps going and `watch` re-attaches. Exit 2 says so.
WATCH_TIMEOUT_S = 2700.0
#: How often a NON-INTERACTIVE watch re-reads the run. Coarse on purpose: `_append` prints a line
#: only when it changes, and a 45-minute CI watch should not emit 180 of them.
POLL_INTERVAL_S = 15.0
#: The live progress a stream has delivered, per run id, and the lock that guards it. Module
#: state because the streams run on their own threads and the display reads from the main one;
#: a dict of small records replaced whole, so a reader never sees a half-written sample.
_LIVE: dict[str, dict[str, Any]] = {}
_LIVE_LOCK = threading.Lock()

#: How often an INTERACTIVE watch re-reads it. tqdm separates measurement from display cadence --
#: `mininterval` defaults to 0.1s -- and the same split applies here: the run writes its heartbeat on
#: every document, so a terminal redraw costs one file read and nothing to the run. At 15s a
#: single-worker run whose documents take ~12s could show the same numbers twice and read as frozen,
#: which is what prompted this.
POLL_INTERVAL_TTY_S = 2.0


def watch(run_ids: list[str] = typer.Argument(..., help="run id(s), as printed by "
                                              "`memrank submit` / `memrank runs ls`")) -> None:
    """Attach to run(s) until they end: exit 0 all done, 1 any failed, 2 still going at timeout."""
    root = registry.runs_root()
    for rid in run_ids:
        if run_status.read(root / rid) is None:
            _adopt_from_org(root / rid, rid)
    outcomes = _watch_all(root, run_ids)
    worst = max(outcomes.values())
    if worst:
        raise typer.Exit(worst)


def _adopt_from_org(run_dir: Path, run_id: str) -> None:
    """Write a local heartbeat for a run only the ORG knows about.

    A run submitted from another directory, another machine, or by a colleague has no local record
    here -- and `watch` used to refuse it, so a finished cloud run whose directory was gone had
    permanently unreachable results even though the org knew its state and its artifacts.

    Adopted rather than special-cased: everything after this point reads one source, the local
    record, exactly as it would for a run submitted here.

    Raises:
        typer.BadParameter: If neither this machine nor the org has heard of it.
    """
    from memrank import settings

    org = settings.get("defaults.org")
    record = None
    if org:
        try:
            with run_api_client.authenticated_client() as http:
                record = run_api_client.get_run(http, org, run_id)
        except Exception:                    # noqa: BLE001 - absence is an answer, not a failure
            record = None
    if record is None:
        raise typer.BadParameter(
            f"no run {run_id!r} recorded here or in your org (`memrank runs ls`)")
    run_dir.mkdir(parents=True, exist_ok=True)
    status = run_status.RunStatus.create(
        run_dir, target=record.get("target_ref") or "?", benchmark=record.get("benchmark") or "?",
        slice_=record.get("slice"))
    status.mark_remote(placement="cloud", task_arn=record.get("task_arn") or "",
                       region=record.get("region") or "", cluster=record.get("cluster") or "",
                       log_group=record.get("log_group") or "",
                       artifact_bucket=(record.get("artifact") or {}).get("bucket") or "")
    style.say(f"{run_id}: adopted from your org (it was not recorded on this machine)")


class ProgressDisplay:
    """Draws the runs being watched -- a redrawn block on a terminal, appended lines elsewhere.

    Owns a :class:`~memrank.term.progress.LiveRegion` on a terminal, and with it the cursor for
    the whole watch. EVERY line this command emits while that region is live goes through
    :meth:`say`, because a bare `style.say` moves the cursor without the region knowing, and from
    then on the region erases rows that hold something else. That was one of the ways the block
    used to dissolve into fragments partway through a cloud run.
    """

    def __init__(self, stream: Any = None) -> None:
        self._region = progress.LiveRegion(stream if stream is not None else sys.stderr)
        self._last: dict[str, str] = {}
        #: Whether a quit key is actually being read. Advertising `q` where nothing listens -- a
        #: pipe, a background job -- is worse than saying nothing at all.
        self.listening = False

    @property
    def region(self) -> Any:
        """The live region, for the one other thing that draws during a watch: a download."""
        return self._region

    def say(self, text: str) -> None:
        """Emit a narration line without disturbing the block -- above it, permanently."""
        self._region.print_above(text)

    def close(self) -> None:
        """Give the cursor back. Safe on every exit path, and safe to call twice."""
        self._region.close()

    def show(self, root: Path, run_ids: list[str], *, cloud: _Cloud | None) -> None:
        found = [(rid, record) for rid in run_ids
                 if (record := _progress_of(root / rid, cloud=cloud))]
        if not found:
            return
        if self._region.live:
            self._redraw(found)
        else:
            self._append(found)

    def _redraw(self, found: list[tuple[str, dict]]) -> None:
        lines: list[str] = []
        for rid, record in found:
            lines.extend(progress.block(_title(rid, record), record))
        if self.listening:
            lines.append(progress.QUIT_HINT)
        self._region.draw(lines)

    def _append(self, found: list[tuple[str, dict]]) -> None:
        """One line per run, and only when it has changed -- a 45-minute watch at 15s polls would
        otherwise put 180 identical lines in a CI log."""
        for rid, record in found:
            rendered = progress.line(_title(rid, record), record)
            if self._last.get(rid) != rendered:
                style.say(rendered)
                self._last[rid] = rendered


def _say(display: Any, text: str) -> None:
    """Narrate through the live block when there is one, and plainly when there is not.

    The block hides the cursor and counts the rows it drew; a `style.say` landing in the middle of
    that scrolls the screen underneath it and every subsequent frame erases the wrong rows. Going
    through the display makes the line land ABOVE the block and stay there, which is what these
    lines were always for -- a record of what happened, not part of the live view.
    """
    if display is None:
        style.say(text)
        return
    display.say(text)


def _title(run_id: str, record: dict[str, Any]) -> str:
    """What names a run in the display: its cell if known, else its id."""
    return record.get("_title") or run_id


class _Cloud(NamedTuple):
    """The API client and the org to poll under, as ONE value.

    `_cloud_client` hands both back together or raises, so they were never independently
    optional -- but threaded as two arguments each consumer could test one and use the other,
    which is precisely what a `None` org reaching a `/orgs/{org}/...` URL would be. Bundling
    them makes "are we watching a cloud run" a single question with a single answer.
    """

    http: Any
    org: str


def _progress_of(run_dir: Path, *, cloud: _Cloud | None) -> dict[str, Any] | None:
    """One run's progress record -- from its local heartbeat, or from S3 for a cloud run.

    A cloud run's heartbeat here says only what the submitter knew; the live record is what the
    task itself published (`run_status._publish_remote`). Absence is normal and silent: a task
    that has not published yet, or one from before this existed, simply has no bar.
    """
    data = run_status.read(run_dir) or {}
    record = data.get("progress") or None
    if data.get("placement") == "cloud" and cloud is not None:
        # The stream first, then the artifact. Both describe the same run, but the stream's copy
        # is at most one publish old while the artifact's is at most one poll old on top of that
        # -- and asking S3 for something already delivered is a request per run per tick that
        # tells the viewer nothing new.
        record = _live_progress(run_dir.name) or _remote_progress(cloud, run_dir.name) or record
    if record and data.get("target"):
        record = {**record, "_title": f"{data['target']} × {data.get('benchmark')}"}
    return record


class _EventStream:
    """One background thread reading a cloud run's live events into :data:`_LIVE`.

    A thread rather than an async rewrite of the watch loop, deliberately: `watch` polls a whole
    sweep together on one clock, and every reader below it -- `_poll_one`, `reconcile`, the
    display -- is synchronous. Turning that inside out to gain a progress bar would be a large
    change to the part of this command that already works, in order to improve the part that is
    cosmetic. A daemon thread per watched cloud run costs a socket and reads a dict.

    Failure is SILENT AND TERMINAL for this thread, never for the watch. A deployment with no
    event transport answers 503, an older server answers 404, and a dropped connection is just
    the end of the stream. In every case this stops and the poll path below keeps drawing the
    bar exactly as it did before this existed -- which is why nothing here retries: a stream that
    reconnected on a server that has no such route would be a loop nobody sees.
    """

    def __init__(self, http: Any, org: str, run_id: str) -> None:
        self._http = http
        self._org = org
        self._run_id = run_id
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"watch:{run_id}", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        """Ask the reader to finish after its next frame. Not joined: the request is blocked in
        a read with no deadline, and a watch that is exiting must not wait on a quiet server."""
        self._stop.set()

    def _run(self) -> None:
        try:
            run_api_client.stream_run_events(self._http, self._org, self._run_id, self._on_event)
        except Exception:  # noqa: BLE001 - see the class docstring; the poll path still draws
            return

    def _on_event(self, kind: str, payload: dict[str, Any]) -> bool:
        if kind == "run.progressed":
            record = payload.get("progress")
            if isinstance(record, dict):
                with _LIVE_LOCK:
                    _LIVE[self._run_id] = record
        return not self._stop.is_set()


def _live_progress(run_id: str) -> dict[str, Any] | None:
    """The newest sample a stream has delivered for this run, if any."""
    with _LIVE_LOCK:
        return _LIVE.get(run_id)


def _remote_progress(cloud: _Cloud, run_id: str) -> dict[str, Any] | None:
    """The progress a cloud task published, or None if it has not published any.

    Swallows its own failures deliberately: a watch that cannot read a progress file must keep
    watching. The run's OUTCOME comes from `_poll_cloud`, which does not swallow anything, so a
    real failure is still reported -- this only decides whether a bar is drawn.
    """
    from memrank.runs.artifacts import PROGRESS_FILE

    try:
        raw = run_api_client.fetch_artifact(cloud.http, cloud.org, run_id, PROGRESS_FILE)
        return json.loads(raw)
    except Exception:  # noqa: BLE001 - see the docstring; absence of a bar is not an error
        return None


def _watch_all(root: Path, run_ids: list[str]) -> dict[str, int]:
    """Poll every id each pass -- a sweep is watched together, never serialised."""
    from contextlib import nullcontext

    cloud_ids = [rid for rid in run_ids
                 if (run_status.read(root / rid) or {}).get("placement") == "cloud"]
    client, org = _cloud_client() if cloud_ids else (nullcontext(None), None)
    outcomes: dict[str, int | None] = dict.fromkeys(run_ids)
    deadline = time.monotonic() + WATCH_TIMEOUT_S
    display = ProgressDisplay()
    streams: list[_EventStream] = []
    with client as http, key_input.raw_mode() as listening:
        display.listening = listening
        # Both come from `_cloud_client`, which returns them together or raises. Joined into one
        # value here so everything below asks a single question instead of two that were never
        # actually independent.
        cloud = _Cloud(http, org) if http is not None and org is not None else None
        if cloud is not None:
            # Started before the first draw so a run already in flight fills its bar from the
            # stream rather than waiting a poll for it. One per cloud run: a sweep is watched
            # together, and a shared stream would have to demultiplex what the server already
            # separates by route.
            streams = [_EventStream(cloud.http, cloud.org, rid) for rid in cloud_ids]
            for stream in streams:
                stream.start()
        try:
            while True:
                for rid, settled in outcomes.items():
                    if settled is None:
                        outcomes[rid] = _poll_one(root / rid, cloud=cloud, display=display)
                pending = [rid for rid, settled in outcomes.items() if settled is None]
                display.show(root, pending, cloud=cloud)
                if not pending:
                    break
                if time.monotonic() >= deadline:
                    _detach(pending, outcomes, "the watch elapsed", display)
                    break
                try:
                    # `listening` is true only on a terminal that could be put in raw mode, which
                    # is exactly the case where a human is reading the redraw rather than a log.
                    interval = POLL_INTERVAL_TTY_S if listening else POLL_INTERVAL_S
                    quit_pressed = key_input.wait_for_quit(interval, listening=listening)
                except KeyboardInterrupt:
                    # Ctrl-C ends the WATCH, not the run. Unhandled it printed a traceback, which
                    # reads like the evaluation crashed when nothing did -- only the viewer left.
                    _detach(pending, outcomes, "you interrupted the watch", display)
                    break
                if quit_pressed:
                    _detach(pending, outcomes, "you quit the watch", display)
                    break
        finally:
            # The cursor is hidden for the duration of the block, and a hidden cursor OUTLIVES the
            # process: skipping this on the exception path leaves the user's shell looking broken
            # in a way that quitting the program does not fix.
            display.close()
        for stream in streams:
            stream.stop()
    return {rid: outcome for rid, outcome in outcomes.items() if outcome is not None}


def _detach(pending: list[str], outcomes: dict[str, int | None], why: str,
            display: Any = None) -> None:
    """End the watch while the runs continue -- timeout, `q` and Ctrl-C are one situation.

    Outcome 2 for each, the code `watch` already documents as "still going": the watch ended and
    the run did not. Giving a deliberate quit its own code would make every caller learn a fourth
    one to describe a state they already handle.
    """
    for rid in pending:
        _say(display, f"{rid}: still running -- {why}; it keeps going, and "
                      f"`memrank watch {rid}` re-attaches")
        outcomes[rid] = 2


def _poll_one(run_dir: Path, *, cloud: _Cloud | None, display: Any = None) -> int | None:
    """One run's outcome -- 0/1 when it has ended, ``None`` while it is still worth waiting."""
    # `or {}` as everywhere else in this module: a run directory with no status file has an
    # empty record, not an absent one, and `classify` already reads that as `stale` -- which is
    # the honest verdict on a heartbeat nobody ever wrote.
    data = run_status.read(run_dir) or {}
    if data.get("placement") == "cloud":
        if cloud is None:
            # The watch decided which runs were cloud runs before it signed in, from this same
            # field; a run that became a cloud run since cannot be polled under no org. Said
            # plainly rather than sent to `/orgs/None/runs`, which returns a 404 about a run.
            raise RuntimeError(
                f"{run_dir.name} became a cloud run after the watch started; re-run "
                f"`memrank watch {run_dir.name}` so it signs in for it")
        return _poll_cloud(run_dir, data, cloud=cloud, display=display)
    outcome = run_status.classify(data)
    if outcome == "done":
        _say(display, f"{run_dir.name}: {style.good('done')}")
        return 0
    if outcome in ("failed", "stale"):
        # A stale run's process died without recording an outcome; waiting on it would hang
        # a CI job forever on a run nobody is executing.
        reason = data.get("error") or ("the process died without recording an outcome"
                                       if outcome == "stale" else "failed")
        _say(display, f"{run_dir.name}: {style.bad(outcome)} ({reason})")
        return 1
    return None


def _poll_cloud(run_dir: Path, data: dict[str, Any], *, cloud: _Cloud,
                display: Any = None) -> int | None:
    """One cloud run's outcome via the platform API; artifacts come down on success.

    The reconciling itself belongs to :mod:`memrank.runs.reconcile`, which `runs show` and
    `runs sync` also call. Watching is a way of ASKING for a run to be reconciled, not the only
    place that knows how -- which it was, and which is why a run nobody watched kept its numbers
    in the cloud forever.
    """
    record = run_api_client.get_run(cloud.http, cloud.org, run_dir.name)
    state = reconcile.reconcile(
        cloud.http, cloud.org, run_dir, record,
        report=lambda what: _say(display, f"{run_dir.name}: fetching {what}"),
        # So the byte meter becomes a footer UNDER the block instead of a second thing moving the
        # cursor on the same stream -- which is what it was, and what made a cloud run's block come
        # apart exactly when its artifacts started landing.
        region=display.region if display is not None else None)
    if state == "running":
        return None
    if state == "failed":
        reason = record.get("stopped_reason") or record["state"]
        _say(display, f"{run_dir.name}: {style.bad('task FAILED')} "
                      f"(exit {record.get('exit_code')}; {reason})")
        return 1
    _say(display, f"{run_dir.name}: run recorded -> {run_dir}")
    return 0


def _cloud_client() -> tuple[Any, str]:
    """An authenticated API client + the org to poll under, or a refusal to guess."""
    org = settings.get("defaults.org")
    if org is None:
        raise typer.BadParameter("watching a cloud run polls the memrank API under an org: "
                                 "`memrank config set defaults.org <slug>` (or `memrank auth login`)")
    return run_api_client.authenticated_client(), org


#: What a target that could not be stopped costs the batch. 2 is Click's usage error (the id
#: names nothing killable); 1 is "asked, and the answer was no" -- the same split the retired
#: names and the surviving commands already use.
KILL_UNUSABLE_ID = 2
KILL_REFUSED = 1


def kill(run_ids: list[str] = typer.Argument(..., help="run id(s), as printed by "
                                             "`memrank submit` / `memrank ps`")) -> None:
    """Stop background run(s): exit 0 all stopped, non-zero if any target could not be.

    Variadic in `watch`'s shape, because a comma sweep mints one run per target and hands back
    a list of ids: stopping that list should not take N invocations when watching it takes one.
    A target that cannot be stopped is reported and the rest still are -- a batch does not
    abandon its remaining work over one bad id.
    """
    root = registry.runs_root()
    worst = max(_kill_one(root, rid) for rid in run_ids)
    if worst:
        raise typer.Exit(worst)


def _kill_one(root: Path, run_id: str) -> int:
    """Stop one run: signal its process group; the unwinding child records the end."""
    data = run_status.read(root / run_id)
    if data is None:
        style.error(f"no run {run_id!r} recorded here (`memrank runs ls`)")
        return KILL_UNUSABLE_ID
    if data.get("state") in ("done", "failed"):
        style.say(f"{run_id} already ended ({data['state']}); nothing to kill")
        return 0
    if data.get("placement") == "cloud":
        return _kill_cloud(root / run_id, data)
    if run_status.classify(data) == "stale":
        # Nobody is left to write this run's end, so kill is the writer -- recording the
        # crash, never "killed by user", which would relabel the corpse.
        run_status.RunStatus(root / run_id, data).update(
            state="failed", error="the process died without recording an outcome")
        style.say(f"{run_id} was already dead; recorded failed")
        return 0
    pid = data.get("pid")
    if not pid:
        style.error(f"{run_id} records no pid to signal")
        return KILL_UNUSABLE_ID
    # The child is a session leader (start_new_session=True), so pid == pgid and the group
    # signal reaches an in-flight `docker compose` too. The child's SIGTERM handler unwinds
    # its placements and writes every terminal record -- kill never writes a live run's state,
    # so there is no second writer to race.
    os.killpg(pid, signal.SIGTERM)
    style.say(f"stop signalled -- {run_id} unwinds and records failed "
              f"(`memrank watch {run_id}` observes it end)")
    return 0


def _kill_cloud(run_dir: Path, data: dict[str, Any]) -> int:
    """Place-transparent kill: the API stops the task under the org's identity.

    Marking the mirror here races nobody -- the harness's own status.json lives inside the
    dying container -- and a later `watch` poll reads the org record, which the API's
    read-repair settles to the same terminal state, kill reason in ``stopped_reason``.
    """
    http, org = _cloud_client()
    try:
        with http:
            run_api_client.kill_run(http, org, run_dir.name)
    except run_api_client.RunApiError as exc:
        # Already phrased for a person (the sibling paths' pattern, e.g. runs sync); a
        # refused kill changed nothing, so the mirror is deliberately left as it was.
        style.error(str(exc))
        return KILL_REFUSED
    run_status.RunStatus(run_dir, data).update(state="failed", error="killed by user")
    style.say(f"stop requested -- {run_dir.name}'s task is draining; "
              f"the org record settles on its next read")
    return 0
