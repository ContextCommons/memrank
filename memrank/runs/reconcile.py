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
"""Bring one cloud run's local record in line with the org's -- in exactly one place.

A cloud run's numbers live in S3 until something downloads them, and its local heartbeat says
whatever it said at submission until something restamps it. This module is that something.

WHY IT IS ITS OWN MODULE. It used to live inside `watch`, reachable only by watching. A run
submitted and not watched to completion therefore kept its numbers in the cloud forever AND stayed
stamped `running` locally -- 60 of 101 cloud runs on the machine where this was found, 47 of them
still claiming to run hours after their task exited. `runs show` made the split visible by printing
a live `done` beside a local `(none recorded yet)`: one run, two stores, disagreeing.

So reconciliation is a function three callers share (`watch`, `runs show`, `runs sync`) rather
than a step one of them owns. Adding a fourth caller must mean calling this, which
`test_cloud_reconcile.py` enumerates and asserts.

IDEMPOTENT BY CONSTRUCTION. Artifacts overwrite, the heartbeat is last-write-wins, and a run
already reconciled costs one listing. Callers may run it whenever they are unsure.
"""
from __future__ import annotations

import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from memrank.placement import run_api_client
from memrank.runs import artifacts
from memrank.runs import status as run_status
from memrank.term import progress

#: Platform states that mean the task has not finished, so there is nothing to fetch yet.
STILL_RUNNING = ("submitted", "running")

#: How many artifacts come down at once. The population is bimodal: thousands of ~20 KB
#: `detail/<cell>/<query>.json` files, where the cost is one round trip each and throughput scales
#: with concurrency, and a handful of 40-330 MB cells, where the cost is bandwidth and 2-4 streams
#: already saturate it. 8 is above the knee for the first and short of thrashing on the second,
#: and sits well inside httpx's default pool of 100 connections.
ARTIFACT_WORKERS = 8

#: At or above this an artifact is named to ``report``; below it, it is part of a count. A locomo
#: cell emits one detail file per question -- 1540-1986 of them -- so naming every file is thousands
#: of lines that bury the four worth waiting for.
ANNOUNCE_ABOVE_BYTES = 10_000_000

#: Artifact directories that exist for the SERVICE to serve, not for this machine to read.
#: `detail/<cell>/<query>.json` and `corpus/<cell>.json` back the web drilldown, which reads them
#: from S3 through `memrank/api/results_repository.py`; no local path opens them -- `runs show` renders
#: from `registry.cell_files`, a NON-recursive glob of the run directory's top level.
#:
#: Skipping them is the difference between a `runs show` that waits two minutes and one that waits
#: five seconds, because the cost of a fetch is a round trip per file rather than its bytes: a
#: locomo smoke run listed 158 artifacts, of which 153 were these, and the API serves them roughly
#: one at a time. The API's own `get_run_questions` already states the rule this restores -- the
#: evidence is "fetched only when somebody actually opens the drilldown", so the cost "falls on
#: the reader who wants it". The uploader honoured that; the fetcher did not.
#:
#: Taken from the writer's own constants so this cannot drift from `deploy/upload_results.py`.
SERVICE_ONLY_DIRS = (artifacts.CORPUS_DIRNAME, artifacts.DETAIL_DIRNAME)

#: `_mirror_to_mlflow` sets process-global MLflow state (tracking URI, active run), so the reduce
#: that follows a concurrent fetch is serialised. Uncontended by default: the mirror is a no-op
#: unless MEMRANK_MLFLOW_ENABLED=1.
_MIRROR_LOCK = threading.Lock()


def fetch_artifacts(http: Any, org: str, run_dir: Path,
                    report: Callable[[str], None] | None = None,
                    show_progress: bool = True, region: Any = None,
                    workers: int = ARTIFACT_WORKERS, everything: bool = False) -> list[Path]:
    """Download a finished run's results through the API into ``run_dir``.

    Listed first, then streamed ``workers`` at a time. Both halves of that matter at real sizes: a
    full locomo cell measured 83 MB, so one combined response would tell the caller nothing while
    it waited, and buffering a whole cell in memory to write it once is a minute of silence that
    reads as a hang. ``report`` is how a CLI says what is about to arrive before it starts.

    CONCURRENT because a run is not a handful of files. `deploy/upload_results.py` writes one
    detail JSON per question per cell, so a locomo run is thousands of small transfers whose cost
    is a round trip each; taken one at a time that is the whole wait. The result does not depend
    on the width -- see :func:`_download_all` for why.

    SELECTIVE for the same reason, and it matters more: concurrency cannot help while the API
    serves these roughly one at a time, so the only way to stop paying for a file is not to ask
    for it. Everything under :data:`SERVICE_ONLY_DIRS` is skipped because nothing here reads it.
    ``everything`` restores the full mirror for a caller that wants every byte the run uploaded
    rather than everything this machine can render.

    Through the API, not S3. Fetching used to need AWS read credentials while submitting needed
    none -- so a tester could submit a run, watch it succeed and read its logs, then be unable to
    see its numbers. The API holds the identity; the CLI no longer reaches around it.

    Raises:
        RunApiError: If the API refuses. A run reported successful whose results did not arrive
            must not read as recorded -- that is the state `fetch_run` refused for the same reason.
    """
    listed = run_api_client.list_artifacts(http, org, run_dir.name)
    entries = listed if everything else [e for e in listed if _is_renderable(e["name"])]
    run_dir.mkdir(parents=True, exist_ok=True)
    _prepare_dirs(run_dir, entries)
    # Before a byte moves, and on THIS thread: two of the four callers are writing into a live
    # `LiveRegion`, which only one writer may hold.
    _announce(entries, len(listed) - len(entries), report)
    meter = _meter(run_dir.name, entries, region) if show_progress and entries else None
    try:
        return _download_all(http, org, run_dir, entries, meter, workers)
    finally:
        if meter is not None:
            meter.finish()


def _meter(title: str, entries: list[dict[str, Any]], region: Any) -> Any:
    """One meter for the whole fetch, sized from the listing.

    Bytes as they land, not just the announcement. A 271 MB cell printed its name and then nothing
    for minutes, which reads as a hang -- and some of those transfers really do die partway, so
    silence hid WHERE.

    An entry with no size makes the total a lie, so an unsized listing gets no bar and no ETA at
    all: `DownloadProgress` already holds that a percentage invented from an unknown denominator
    is worse than admitting the size is not known.

    ``region`` is a live block already holding the cursor -- `watch`'s, in practice. Given one, the
    meter draws through it as a footer instead of running its own cursor arithmetic on the same
    stream, where the two erase each other's rows.
    """
    sizes = [entry.get("size") for entry in entries]
    total = 0 if any(not size for size in sizes) else sum(sizes)  # type: ignore[arg-type]
    return progress.DownloadProgress(title, total, region=region, files=len(entries))


def _is_renderable(name: str) -> bool:
    """Whether this machine has anything that reads this artifact.

    A prefix test on the directory, not a suffix or a guess: an artifact is service-only when it
    LIVES under one of those directories, which is exactly how `deploy/upload_results.py` writes
    them and how `repository.py` addresses them.
    """
    return not any(name.startswith(f"{directory}/") for directory in SERVICE_ONLY_DIRS)


def _announce(entries: list[dict[str, Any]], skipped: int,
              report: Callable[[str], None] | None) -> None:
    """Say what is about to arrive: the whole fetch, then the files big enough to wait on.

    Naming every file was the contract when a run meant a handful of cells. A run now uploads one
    detail file per question, so that contract prints 1540 lines and hides the 83 MB one among
    them. The summary line keeps the count and the size honest; the named lines keep the defect
    this callback exists for closed.

    ``skipped`` is how many the selection left in the org, reported rather than swallowed.
    """
    if report is None:
        return
    total = sum(entry.get("size") or 0 for entry in entries)
    # What stayed behind is said out loud. A fetch that lists 158 and downloads 5 must not read as
    # one that silently lost 153 -- the drilldown artifacts are still in the org, and still served.
    left = f" ({skipped} drilldown artifact(s) stay in the org)" if skipped else ""
    report(f"{len(entries)} artifact(s), {_megabytes(total)}{left}")
    for entry in entries:
        if (entry.get("size") or 0) >= ANNOUNCE_ABOVE_BYTES:
            report(f"{entry['name']} ({_megabytes(entry.get('size'))})")


def _prepare_dirs(run_dir: Path, entries: list[dict[str, Any]]) -> None:
    """Create every artifact's parent once, here, rather than once per worker."""
    for parent in {(run_dir / entry["name"]).parent for entry in entries}:
        parent.mkdir(parents=True, exist_ok=True)


def _download_all(http: Any, org: str, run_dir: Path, entries: list[dict[str, Any]],
                  meter: Any, workers: int) -> list[Path]:
    """Every artifact, ``workers`` at a time, in listing order whatever the width.

    ``ex.map`` and deliberately NOT ``as_completed`` -- the same guarantee `_run_units_concurrent`
    and the judge stage rest on. It buys two things here: the returned paths are the listing's
    order at any worker count, and the exception that escapes is the EARLIEST-indexed failure
    rather than whichever thread happened to lose first. A fetch that reports a different file as
    the reason on every attempt is not one anybody can act on.

    Leaving the `with` block cancels whatever has not started and waits for what is in flight, so
    nothing writes into ``run_dir`` after the raise.
    """
    if workers <= 1 or len(entries) <= 1:
        # A pool of one is a needless difference between what a debugger walks and what most runs
        # do -- and `max_workers=0` for an empty listing is an error rather than a no-op.
        return [_download_one(http, org, run_dir, entry, meter) for entry in entries]
    with ThreadPoolExecutor(max_workers=min(workers, len(entries))) as pool:
        return list(pool.map(
            lambda entry: _download_one(http, org, run_dir, entry, meter), entries))


def _download_one(http: Any, org: str, run_dir: Path, entry: dict[str, Any],
                  meter: Any) -> Path:
    """One artifact to its place under ``run_dir``, or a refusal that names it.

    The name is re-raised into the message because a fetch is now thousands of files: "forbidden
    (403)" on its own says nothing a user can act on. ``code`` survives because it is
    machine-dispatched.
    """
    name = entry["name"]
    try:
        written = run_api_client.download_artifact(
            http, org, run_dir.name, name, run_dir / name,
            on_chunk=meter.advance if meter else None)
    except run_api_client.RunApiError as exc:
        raise run_api_client.RunApiError(f"{name}: {exc}", code=exc.code) from exc
    if meter is not None:
        meter.file_done()
    return written


def _megabytes(size: int | None) -> str:
    """A byte count as something a person reads before deciding to wait for it."""
    if not size:
        return "size unknown"
    return f"{size / 1_000_000:.1f} MB" if size >= 1_000_000 else f"{size / 1_000:.0f} KB"


#: The state a run ages into once ECS can no longer be asked about it. ECS stops answering
#: DescribeTasks for a task that stopped roughly an hour ago, so any run that finishes while
#: nothing is watching lands here -- permanently. It does NOT mean "we don't know yet"; it means
#: the source the platform was asking has forgotten, and asking again will never help.
_FORGOTTEN = "unknown"

#: Uploaded continuously while a run works, so its presence proves nothing about completion.
_PROGRESS_ARTIFACT = "progress.json"


def _finished_by_artifact(http: Any, org: str, run_id: str, state: str) -> bool:
    """Whether a run the platform lost track of nevertheless left results behind.

    The harness uploads its cells and summary as the last thing it does, so a result artifact is
    proof it reached the end -- durable evidence, where the ECS task status is transient. Consulting
    only the transient source is why four overnight runs finished, uploaded everything, and were
    then unreadable through the CLI while their numbers sat in the bucket.

    Applied ONLY to the forgotten state. A run the platform actively reports as stopped or failed
    is reported that way: this closes a gap in knowledge, it does not overrule an answer.
    """
    if state != _FORGOTTEN:
        return False
    try:
        entries = run_api_client.list_artifacts(http, org, run_id)
    except run_api_client.RunApiError:
        # No artifact listing is not evidence of failure either way; leave the caller's verdict
        # alone rather than converting an API hiccup into a permanent "failed".
        return False
    return any(e.get("name") != _PROGRESS_ARTIFACT for e in entries)


def reconcile(http: Any, org: str, run_dir: Path, record: dict[str, Any],
              report: Callable[[str], None] | None = None, region: Any = None,
              workers: int = ARTIFACT_WORKERS, show_progress: bool = True) -> str:
    """Make ``run_dir`` say what the org says, fetching results when there are results to fetch.

    Args:
        http: An authenticated API client.
        org: The org slug to fetch under.
        run_dir: The local run directory. Created if the run was never seen here.
        record: The org's record for this run, as returned by ``run_api_client.get_run`` or one
            row of ``list_runs`` -- both carry live-merged state.
        report: Called with a one-line description of the fetch, and of each artifact big
            enough to be worth waiting for, before anything downloads. These reach 83 MB; a
            command that says nothing for a minute reads as a hung one.
        workers: How many of this run's artifacts download at once.
        show_progress: Whether to draw a byte meter. False for a caller reconciling SEVERAL runs
            at once, where one meter per run would redraw over the others.

    Returns:
        The local state now recorded: ``"running"``, ``"done"`` or ``"failed"``.

    Raises:
        RunApiError: If the artifacts were refused. Deliberately NOT caught here -- a caller that
            swallows it would report a successful run as having produced nothing, which is the
            defect this module exists to close.
    """
    data = run_status.read(run_dir) or {}
    status = run_status.RunStatus(run_dir, data)
    state = record["state"]
    if state in STILL_RUNNING:
        # Restamp the heartbeat so a healthy task does not read stale in `memrank ps`
        # (classify() calls a heartbeat older than 90s stale; a cloud record has no pid).
        status.update(state="running", message=f"task {state}")
        return "running"
    if state != "stopped-success" and not _finished_by_artifact(http, org, run_dir.name, state):
        reason = record.get("stopped_reason") or state
        status.update(state="failed", error=f"exit {record.get('exit_code')}: {reason}")
        return "failed"
    from memrank.orchestration.sweep import _mirror_to_mlflow

    fetched = fetch_artifacts(http, org, run_dir, report, show_progress=show_progress,
                              region=region, workers=workers)
    # Once the artifacts are down, a cloud run IS a local run as far as everything downstream
    # is concerned -- including MLflow. Serialised: `runs sync` reconciles several runs at once,
    # and the mirror writes process-global MLflow state.
    with _MIRROR_LOCK:
        _mirror_to_mlflow(run_dir)
    status.update(state="done", message=f"fetched {len(fetched)} artifact(s)", pct=100)
    return "done"
