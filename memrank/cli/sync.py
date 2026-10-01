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
"""``memrank runs sync`` -- this machine and the org's one universe are made to agree.

A run is a run wherever it ran, so a finished local run belongs in the org universe exactly
as a cloud-born one does. Two paths put it there and they share every decision: the auto-sync
that follows each finished run (:func:`memrank.runs.push.auto_sync`, silent where there is no
hosted side to sync to) and this verb, which reconciles everything pending. Reconciling ALL of
it is the default because the listing already shows what is pending -- nobody should have to
*know* which runs the org lacks.

**THE PUSH IS NOT HERE.** It is :mod:`memrank.runs.push`, and this module is one of its two
callers. It lived here, which made the automatic report a finished sweep sends reachable only
by importing the command line -- so a library sweep pulled every verb and the argument parser
into a caller who asked for an evaluation. That module's docstring has the rest.

**BOTH DIRECTIONS.** Push was once the whole verb, and it left the other half of the disagreement
standing: a cloud run's results stay in S3 and its local heartbeat stays at `running` until
something reconciles it, and until :mod:`memrank.runs.reconcile` existed only `watch` did. So a
sweep nobody watched read as permanently running with no numbers -- 47 such runs on the machine
where this was found. Pulling (:func:`_pull`) is that half, and it delegates the per-run work to
the same function `watch` and `runs show` call, so three callers cannot drift into three answers.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import typer

from memrank import settings
from memrank.placement import run_api_client
from memrank.runs import reconcile, registry
from memrank.runs import status as run_status
from memrank.runs.push import push_one
from memrank.term import style

#: The states a run can be synced in. A run still going has no record yet; it syncs when it
#: finishes, which is what "as it happens" means for something whose results are the payload.
_SYNCABLE = ("done", "failed")

#: How many runs reconcile at once. Multiplies with the per-run artifact width, so the default
#: pair is 3 x `reconcile.ARTIFACT_WORKERS` = 24 transfers in flight -- enough to keep a home link
#: busy across several runs, short of the point where the big cells start fighting for the disk.
SYNC_RUN_WORKERS = 3


def is_pending(data: dict | None) -> bool:
    """Whether this run is one the org lacks: finished, born here, not yet synced.

    The listing's SYNCED column and the bare verb must never disagree about what "pending"
    means, so both ask this. A cloud-born run was born in the universe; a run still going has
    nothing to send yet.
    """
    return (data is not None
            and data.get("placement") != "cloud"
            and not data.get("synced_org")
            and run_status.classify(data) in _SYNCABLE)


def pending_run_dirs() -> list[tuple[Path, dict]]:
    """Every local run the org lacks, oldest first -- the order they happened in."""
    root = registry.runs_root()
    if not root.exists():
        return []
    found = []
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        data = run_status.read(run_dir)
        if data is not None and is_pending(data):
            found.append((run_dir, data))
    return found


def is_unreconciled(data: dict | None) -> bool:
    """Whether this cloud run's local record may lag the org's.

    A cloud run's numbers stay in S3 until something fetches them and its heartbeat says whatever
    it said at submission until something restamps it -- and until this verb existed, only `watch`
    did either. So a run is a candidate while it still reads as going, and whenever it has no
    results here: the two ways the lag shows up.
    """
    return data is not None and data.get("placement") == "cloud" and (
        run_status.classify(data) not in _SYNCABLE or not data.get("_has_cells"))


def unreconciled_run_dirs() -> list[tuple[Path, dict]]:
    """Every cloud run here whose local record may lag the org's, oldest first."""
    root = registry.runs_root()
    if not root.exists():
        return []
    found = []
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        data = run_status.read(run_dir)
        if data is None:
            # A run directory with no status file is not a lagging cloud run; it is not a run
            # this machine ever recorded. `is_unreconciled(None)` already said False -- said
            # here instead so the record that reaches the caller is one, not one-or-nothing.
            continue
        data = {**data, "_has_cells": bool(registry.cell_files(run_dir))}
        if is_unreconciled(data):
            found.append((run_dir, data))
    return found


def _org_records(http, org: str, run_ids: list[str]) -> dict[str, dict]:
    """The org's record for each id: one listing, then `get_run` for whatever it did not cover.

    The listing is newest-first across the whole org, so a limit generous enough for today is not
    a guarantee for an old run -- and a run silently missing from the map would be a run this verb
    reports as untouched when it simply never asked about it.
    """
    records: dict[str, dict] = {}
    wanted = set(run_ids)
    try:
        listing = run_api_client.list_runs(http, org, mine=False, limit=len(wanted) + 100)
        records = {row["id"]: row for row in listing["runs"] if row["id"] in wanted}
    except run_api_client.RunApiError as exc:
        style.error(f"listing the org's runs failed ({exc}); asking run by run")
    for run_id in run_ids:
        if run_id not in records:
            try:
                records[run_id] = run_api_client.get_run(http, org, run_id)
            except run_api_client.RunApiError as exc:
                # One unanswerable id must not end the batch. A run this machine remembers and the
                # org does not (deleted, or from before it was recorded) used to raise here and
                # abort the whole reconciliation -- so a single 404 from days ago made every later
                # run permanently unreadable, including finished ones with results waiting.
                style.error(f"{run_id}: {exc}")
    return records


#: How a buffered line is replayed. Kept as a name rather than a bare string so the reduce below
#: cannot silently print a refusal as ordinary narration.
_SAY, _ERROR = "say", "error"


def _reconcile_one(http, org: str, run_dir: Path, record: dict | None,
                   workers: int, show_progress: bool) -> tuple[str, list[tuple[str, str]]]:
    """One run reconciled, with everything it wanted to say buffered rather than printed.

    Buffered because runs reconcile concurrently: printing from a worker interleaves two runs'
    lines into one unreadable stream, and the caller replays these in listing order instead.

    Returns the state recorded and the ``(how, text)`` lines to print. Refusals are caught HERE,
    per run, exactly as they were when this ran serially -- one run the org has forgotten must not
    end the batch.
    """
    said: list[tuple[str, str]] = []
    if record is None:
        # Already reported by _org_records; counted so the summary line accounts for every
        # target rather than quietly covering fewer runs than it claimed to reconcile.
        return "unknown to the org", said
    try:
        state = reconcile.reconcile(http, org, run_dir, record,
                                    report=lambda what: said.append(
                                        (_SAY, f"    fetching {what}")),
                                    workers=workers, show_progress=show_progress)
    except run_api_client.RunApiError as exc:
        said.append((_ERROR, f"{run_dir.name}: {exc}"))
        return "refused", said
    said.append((_SAY, f"  {run_dir.name}  -> {state}"))
    return state, said


def _pull(http, org: str, workers: int = reconcile.ARTIFACT_WORKERS,
          run_workers: int = SYNC_RUN_WORKERS) -> None:
    """Bring every lagging cloud run's local record in line with the org's, several at a time.

    ``pool.map`` over the targets, which are already oldest-first, so the output is the same
    sequence of lines at any width -- the reduce below reads results in listing order, not in the
    order threads happened to finish.
    """
    targets = unreconciled_run_dirs()
    if not targets:
        return
    style.say(f"{len(targets)} cloud run(s) to reconcile, {run_workers} at a time")
    records = _org_records(http, org, [run_dir.name for run_dir, _ in targets])
    # A meter belongs to one transfer. Reconciling several runs at once, each run's would redraw
    # over the others on the same stream, so the batch keeps its per-run summary lines instead --
    # and a lone run, which has nobody to collide with, keeps its bar.
    show_progress = len(targets) == 1 or run_workers <= 1

    def work(target: tuple[Path, dict]) -> tuple[str, list[tuple[str, str]]]:
        run_dir = target[0]
        return _reconcile_one(http, org, run_dir, records.get(run_dir.name), workers,
                              show_progress)

    counts: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=max(1, min(run_workers, len(targets)))) as pool:
        for state, said in pool.map(work, targets):
            for how, line in said:
                (style.error if how == _ERROR else style.say)(line)
            counts[state] = counts.get(state, 0) + 1
    style.say("reconciled " + ", ".join(f"{n} {state}" for state, n in sorted(counts.items())))


def _org_or_refuse() -> str:
    """The org a sync lands in, or a refusal naming how to set one."""
    org = settings.get("defaults.org")
    if org is None:
        raise typer.BadParameter("syncing writes into an org: "
                                 "`memrank config set defaults.org <slug>` "
                                 "(or `memrank auth login`)")
    return org


def _selected(run_ids: list[str] | None) -> list[tuple[Path, dict]]:
    """The runs to sync: everything pending, or exactly the ones named.

    A named run is pushed even if it is already marked synced -- naming it IS the user saying
    "send this one", and the route is idempotent, so that is the repair path for a record the
    org somehow lacks. What a name cannot do is invent a run, or send one that has nothing
    to send: those are refused individually, by name.
    """
    if not run_ids:
        return pending_run_dirs()
    root = registry.runs_root()
    chosen = []
    for run_id in run_ids:
        data = run_status.read(root / run_id)
        if data is None:
            raise typer.BadParameter(f"no run {run_id!r} recorded here (`memrank runs ls`)")
        if data.get("placement") == "cloud":
            raise typer.BadParameter(
                f"{run_id} ran in the cloud -- it was born in the org, so it PULLS rather than "
                f"pushes: `memrank runs sync` with no arguments, or `memrank runs show {run_id}`")
        if run_status.classify(data) not in _SYNCABLE:
            raise typer.BadParameter(f"{run_id} is still going here; it syncs when it finishes")
        chosen.append((root / run_id, data))
    return chosen


def sync(run_ids: list[str] = typer.Argument(None, help="run id(s); default syncs everything "
                                                        "this machine has that the org lacks"),
         workers: int = typer.Option(
             reconcile.ARTIFACT_WORKERS,
             help="Download this many of ONE run's artifacts at once. A run uploads one file per "
                  "question, so the wait is round trips rather than bytes. What arrives is "
                  "unaffected: the fetch is ordered by the listing, not by which thread finished."),
         run_workers: int = typer.Option(
             SYNC_RUN_WORKERS, "--run-workers",
             help="Reconcile this many RUNS at once. Separate from --workers because they bound "
                  "different resources: this is runs, that is transfers within one. Multiplies "
                  "with it, so the defaults put 24 downloads in flight."),
         ) -> None:
    """Make this machine and your org agree: push finished local runs, pull finished cloud ones.

    Both directions, because a run is a run wherever it ran and "sync" that only pushed left half
    the disagreement in place -- cloud runs stamped `running` here long after they finished, with
    their numbers still in the org. Naming run ids selects the push side only; a cloud run is
    pulled by the bare verb, which is the only form that can know what the org has.

    The two width options apply to the pull side, which is the slow one: a cloud run's results are
    thousands of small files. Lower them on a link that cannot take the concurrency; the records
    they produce are identical at any value.
    """
    with _client() as http:
        org = _org_or_refuse()
        if not run_ids:
            _pull(http, org, workers=workers, run_workers=run_workers)
        targets = _selected(run_ids)
        if not targets:
            style.say("nothing pending -- the org has every finished run here")
            return
        failed = 0
        for run_dir, data in targets:
            try:
                push_one(http, org, run_dir, data)
            except run_api_client.RunApiError as exc:
                # One run's refusal is not the batch's: a reconciler does all the work it can,
                # and the ones it could not do are named rather than counted.
                style.error(f"{run_dir.name}: {exc}")
                failed += 1
                continue
            style.out(run_dir.name)
            style.say(f"synced {run_dir.name} -> org {org}")
    if failed:
        style.say(f"synced {len(targets) - failed} of {len(targets)} -- "
                  f"`memrank runs sync` retries the rest")
        raise typer.Exit(1)


def _client():
    """The authenticated API client, or a loud exit -- ``sync`` is an explicitly remote verb."""
    try:
        return run_api_client.authenticated_client()
    except run_api_client.RunApiError as exc:
        style.error(str(exc))
        raise typer.Exit(1) from exc
