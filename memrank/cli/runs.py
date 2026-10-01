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
"""``memrank runs`` -- the central entity, in both its tenses.

A run is one thing across its whole life: born at submission, live while executing, a durable
record when finished. "Live" is a status, not a different noun, so one listing shows both and
the state column tells them apart.

**A row is a run, and a run is one target.** The predecessor (``list-runs``) listed cells under
a shared sweep id, so a two-target sweep printed one run id twice -- an id that names two rows is
not a handle, and every verb the model adds later (``show``, ``watch``, ``kill``) takes an id as
its handle. Submission now closes the same defect from the other side: a comma sweep mints one
run per target (locally and in the cloud alike), so an id names exactly one target's row.

Two local stores answer, each for what it actually knows:

* :mod:`memrank.runs.status` -- the heartbeat, authoritative from the moment a run is born
  (targets, eval, state, timestamps, where it executes).
* :mod:`memrank.runs.registry` -- the recorded cells, authoritative once results exist
  (composite, config hash, artifact path).

A run folder with results but no heartbeat predates the heartbeat; its state reads ``unknown``
rather than being inferred from the presence of files. Guessing a terminal state is exactly the
lie the interface must not tell.

The listing merges the local stores with the org's universe over the platform API; ``show``
lives in :mod:`memrank.cli.runs_show`, which answers about one run rather than many.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import datetime, timezone

import typer

from memrank import (
    settings,
)
from memrank.cli import deprecated as deprecated_cli
from memrank.cli import monitor as monitor_cli
from memrank.cli import runs_show as runs_show_cli
from memrank.cli import sync as sync_cli
from memrank.cli import watch as watch_cli
from memrank.metrics import headline
from memrank.placement import hosted, run_api_client
from memrank.runs import registry
from memrank.runs import status as run_status
from memrank.runs.status import PLATFORM_STATES, UNKNOWN_STATE
from memrank.term import style, table

runs_app = typer.Typer(help="Browse and inspect runs -- the record of every evaluation.")
# `logs`, `watch`, `kill` and `sync` follow and reconcile the background runs `submit` starts,
# hidden with it (ATO-2343); `memrank run` is foreground and uploads its own run.
# Deprecated with `submit`. `sync` is not: it also uploads finished local runs.
runs_app.command("logs", hidden=True)(deprecated_cli.deprecated("runs logs", monitor_cli.logs))
runs_app.command("watch", hidden=True)(deprecated_cli.deprecated("runs watch", watch_cli.watch))
runs_app.command("kill", hidden=True)(deprecated_cli.deprecated("runs kill", watch_cli.kill))
runs_app.command("show")(runs_show_cli.show)
runs_app.command("sync", hidden=True)(sync_cli.sync)

#: A run nobody can still attach to or stop is not live. ``submitted`` and ``queued`` are (a
#: live process will start them, and they are killable); ``stale`` and ``unknown`` are not,
#: whatever they might yet turn out to have been.
_LIVE_STATES = ("submitted", "queued", "running")

#: How many runs a bare listing shows. A working set, not an archive -- `gh run list`'s number,
#: for the same kind of listing. Whatever is dropped is announced; a cap the reader cannot see
#: makes a partial answer look like a complete one.
DEFAULT_LIMIT = 20

#: What `--limit 0` asks the platform for. The API pages with a cursor and has no "everything"
#: mode, so uncapped is bounded here rather than pretended.
_NO_CAP_FETCH = 500

LOCAL, CLOUD = "local", "cloud"

#: What the SYNCED column says. ``pending`` is the load-bearing one: the model's promise is
#: that a reader SEES which runs the org lacks and never has to know which, so the word for
#: "this machine has it and the org does not" has to appear in the listing itself. ``—`` is
#: for runs with nothing to send yet -- still going, or no record at all.
SYNCED, PENDING, NOTHING_TO_SYNC = "yes", "pending", "—"

#: Narrowest the id column gets. Today's ids fit; the legacy ones (42 chars) do not, so the
#: printed width comes from the rows and this is only the floor.
_MIN_ID_WIDTH = 38
_MIN_TARGET_WIDTH = 22


def _age(started_at: str) -> str:
    """How long ago a run started -- ``45s``, ``12m``, ``5h``, ``9d``.

    Deliberately not :func:`monitor_cli.elapsed`, which measures how long a run *took*. Both are
    honest numbers answering different questions, and a listing sorted newest-first is asking
    "when", not "how long". The coarse step also keeps old runs readable: a run from last week
    is ``9d``, not ``14400m00s``.

    A run id stands in for the timestamp when a receipt never recorded one, so anything
    unparseable renders as no answer rather than raising.
    """
    try:
        start = datetime.fromisoformat(started_at)
    except (TypeError, ValueError):
        return "—"
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    seconds = int((datetime.now(timezone.utc) - start).total_seconds())
    if seconds < 0:
        return "—"
    for limit, unit, size in ((60, "s", 1), (3600, "m", 60), (86400, "h", 3600)):
        if seconds < limit:
            return f"{seconds // size}{unit}"
    return f"{seconds // 86400}d"


@dataclass
class RunRow:
    """One run, as listed: identity from whichever store recorded it, results if it has any."""

    run_id: str
    targets: list[str]
    eval_name: str
    state: str
    age: str
    started_at: str
    cells: list[registry.RunInfo]
    place: str = LOCAL
    synced: str = NOTHING_TO_SYNC
    #: How far through, when the run is reporting it. Carried across from the retired flat `ps`,
    #: whose one advantage over this listing was saying "89%" instead of just "running" -- the
    #: difference between a six-hour run that is working and one that is wedged.
    progress: str = ""
    #: The score the ORG holds for this run, when the platform listed it. The API computes it
    #: with the same projection this command uses, so trusting it is not trusting a second
    #: opinion. Kept because a run's artifacts live on the machine that ran it: without these,
    #: every run submitted from another laptop -- and every cloud run -- listed as "—" here while
    #: the GUI showed its score.
    org_composite: float | None = None
    org_cell_count: int | None = None
    org_score_kind: str | None = None
    #: ``agent`` for a ``memrank run`` run, whose record is one result rather than engine cells:
    #: the API sends its score with ``cell_count`` 0, so the count cannot say it is one score.
    org_kind: str | None = None

    def as_dict(self) -> dict:
        """The machine-readable form -- every column, plus the per-cell results."""
        return {
            "run_id": self.run_id, "targets": self.targets, "eval": self.eval_name,
            "progress": self.progress,
            "state": self.state, "age": self.age, "started_at": self.started_at,
            "place": self.place, "synced": self.synced,
            "score": _score_value(self), "score_kind": _score_kind(self),
            "cells": [{"target": c.adapter, "eval": c.benchmark, "composite": c.composite,
                       "headline": c.headline, "headline_kind": c.headline_kind,
                       "config_hash": c.config_hash, "path": str(c.path)} for c in self.cells],
        }


def _synced(data: dict) -> str:
    """Whether the org has this run -- asking :mod:`memrank.cli.sync` what pending means.

    Not re-derived here: a column that called a run synced while `runs sync` would send it
    (or the reverse) is a listing that lies about work the tool then does anyway.
    """
    if data.get("placement") == CLOUD or data.get("synced_org"):
        return SYNCED
    return PENDING if sync_cli.is_pending(data) else NOTHING_TO_SYNC


def _progress(data: dict) -> str:
    """``89%`` when the run reports it, else empty. Only a live run has anything to say."""
    pct = (data.get("progress") or {}).get("pct")
    return f"{pct}%" if pct is not None else ""


def _row(run_id: str, data: dict | None, cells: list[registry.RunInfo]) -> RunRow:
    """Assemble one row, preferring the heartbeat for identity -- it exists from the start."""
    if data is not None:
        data["status"] = run_status.classify(data)
        started = data.get("started_at") or run_id
        return RunRow(run_id=run_id,
                      targets=[data["target"]] if data.get("target") else [],
                      eval_name=data.get("eval_ref") or data.get("benchmark") or "?", state=data["status"],
                      age=_age(started), started_at=started, cells=cells,
                      place=CLOUD if data.get("placement") == CLOUD else LOCAL,
                      synced=_synced(data), progress=_progress(data))
    started = min(c.timestamp for c in cells)
    return RunRow(run_id=run_id, targets=sorted({c.adapter for c in cells}),
                  eval_name=cells[0].benchmark, state=UNKNOWN_STATE, age=_age(started),
                  started_at=started, cells=cells)


def _platform_row(payload: dict) -> RunRow:
    """One row from the org universe, in the local vocabulary.

    A payload without ``place`` came from a deployment older than the sync model, and that
    deployment can only hold cloud-born runs -- nothing else could have written one. So the
    absence is a fact about the server's version, not a value being guessed at.
    """
    created = payload.get("created_at") or ""
    return RunRow(run_id=payload["id"], targets=[payload.get("target_ref") or "?"],
                  eval_name=payload.get("benchmark") or "?",
                  state=PLATFORM_STATES.get(payload.get("state", ""), UNKNOWN_STATE),
                  age=_age(created), started_at=created, cells=[],
                  place=LOCAL if payload.get("place") == LOCAL else CLOUD,
                  # Listed by the org IS being in the org.
                  synced=SYNCED,
                  # The org's own projection of this run's score, kept rather than discarded.
                  # It was thrown away here, so a run whose artifacts are not on THIS machine
                  # listed as "—" while the GUI -- reading the same record -- showed its number.
                  # A deployment too old to send these leaves them None, which lands on the
                  # same "—" as before rather than on a wrong number.
                  org_composite=payload.get("composite"),
                  org_cell_count=payload.get("cell_count"),
                  org_score_kind=payload.get("score_kind"),
                  org_kind=payload.get("kind"))


def _merge(local: list[RunRow], platform: list[RunRow]) -> list[RunRow]:
    """One list of runs from two stores, deduplicated by id.

    A cloud submission writes a local record under the SERVER-minted id, so the same run is in
    both stores and concatenating would double it. The platform is authoritative for state -- it
    watches the task, the laptop only remembers submitting -- while the local record holds the
    cells, which exist only once artifacts were fetched. Take each from the store that knows.
    """
    by_id = {row.run_id: row for row in local}
    for row in platform:
        known = by_id.get(row.run_id)
        by_id[row.run_id] = (row if known is None
                             else replace(row, cells=known.cells,
                                          started_at=known.started_at or row.started_at))
    return sorted(by_id.values(), key=lambda r: (r.started_at, r.run_id), reverse=True)


def _rows() -> list[RunRow]:
    """Every local run, newest first."""
    root = registry.runs_root()
    if not root.exists():
        return []
    grouped = registry.cells_by_run()
    rows = []
    for run_dir in root.iterdir():
        if not run_dir.is_dir():
            continue
        data = run_status.read(run_dir)
        cells = grouped.get(run_dir.name, [])
        if data is None and not cells:
            continue  # a directory that recorded neither state nor results is not a run
        rows.append(_row(run_dir.name, data, cells))
    rows.sort(key=lambda r: (r.started_at, r.run_id), reverse=True)
    return rows


def _platform_rows(*, mine: bool,
                   limit: int) -> tuple[list[RunRow], hosted.Unavailable | None]:
    """The org universe's rows, plus why it could not be consulted, if it could not be.

    Both whether that second value is worth printing and how it is phrased belong to
    :mod:`memrank.placement.hosted`, not here.
    """
    try:
        # The session is resolved before the org, because signing in also configures the org:
        # telling someone who never logged in to set `defaults.org` sends them to do by hand
        # what `auth login` would have done for them.
        with run_api_client.authenticated_client() as http:
            org = settings.get("defaults.org")
            if org is None:
                return [], hosted.no_org()
            body = run_api_client.list_runs(http, org, mine=mine, limit=limit)
    except run_api_client.RunApiError as exc:
        return [], hosted.refused(exc)
    except Exception as exc:  # noqa: BLE001 - a network fault must not break a local listing
        return [], hosted.unreachable(exc)
    rows = [_platform_row(p) for p in body.get("runs", [])]
    if mine and body.get("scope") != "mine":
        # The server ignored `mine` -- it predates the parameter. Captioning these as this
        # user's would be a lie produced by a version skew, so say what they actually are.
        # Broken rather than absent: this listing DID reach an org, and misattributing other
        # people's runs is exactly the kind of thing a local-only user must still be told.
        return rows, hosted.Unavailable(
            "this deployment cannot scope a listing to you, so every submitter's runs are shown")
    return rows, None


def _listing(*, org_wide: bool, fetch: int) -> tuple[list[RunRow], list[str]]:
    """The rows a listing should show, and everything the user must be told about them.

    ``fetch`` bounds what the PLATFORM is asked for; the local store is always read whole, since
    reading it costs a directory walk and filtering must see everything (a `--target` match may
    be older than the newest N). The display cap is applied last, by the caller.
    """
    local = _rows()
    # No pre-flight "am I signed in" probe: that read the stored token to answer a boolean, and
    # then the call read it again -- two credential reads, one of them never spent (and on macOS,
    # two Keychain prompts). The call itself already reports a missing session as a typed
    # refusal, so the outcome of trying is the answer.
    platform, problem = _platform_rows(mine=not org_wide, limit=fetch)
    if problem is None:
        return _merge(local, platform), []
    if org_wide:
        # Explicitly remote: answering it from local records would be fabrication, so the
        # problem is fatal here -- and the consequence is the opposite of the bare listing's.
        raise typer.BadParameter(f"{problem}\n`--org` has nothing to read without it")
    if not hosted.worth_saying(problem):
        # A machine with no hosted side answered a local question locally. Nothing happened.
        return _merge(local, platform), []
    return _merge(local, platform), [f"{problem}; showing this machine's runs"]


def _score_value(row: RunRow) -> float | None:
    """The run's headline number, from whichever store actually holds its results.

    Local cells first: this machine has the whole artifact, so it can project the judged block
    and the composite alike. Falling back to the org's number is what makes a run submitted from
    another laptop -- or run in the cloud, whose artifacts never touch this disk -- list with the
    score the GUI shows it with. Both numbers come out of ``memrank.metrics.headline``, so the
    fallback is the same measurement rather than a second opinion.
    """
    local = [c.headline for c in row.cells if c.headline is not None]
    if local:
        return local[0] if len(row.cells) == 1 else None
    return row.org_composite if _one_org_score(row) else None


def _score_kind(row: RunRow) -> str | None:
    """What :func:`_score_value` measured, or ``None`` when it published nothing."""
    local = [c for c in row.cells if c.headline is not None]
    if local:
        return local[0].headline_kind if len(row.cells) == 1 else None
    return row.org_score_kind if _one_org_score(row) else None


def _one_org_score(row: RunRow) -> bool:
    """Whether the org's number is one run's own score: a single engine cell, or an agent run.

    An agent run's record holds one result and no cells, so the API counts it as zero cells
    while sending its judged score; reading the count alone listed every judged ``memrank run``
    run as "—" (ATO-2365).
    """
    return row.org_kind == "agent" or row.org_cell_count == 1


def _composite(row: RunRow) -> str:
    """The run's headline score as a column: one cell's number, or a count when it swept several.

    Mirrors ``memrank.api.runs._score`` exactly -- the same run listed by this command and by
    the GUI must not read two different ways. `tests/cli/test_runs_cli_composite.py` drives both.

    A judged score is marked, because a judged answer-correctness and a substring recall are
    different measurements and a bare number cannot say which one it is.
    """
    value = _score_value(row)
    if value is None:
        cells = len(row.cells) or (row.org_cell_count or 0)
        return f"{cells} cells" if cells > 1 else "—"
    suffix = " j" if _score_kind(row) == headline.JUDGED else ""
    return f"{value:.4f}{suffix}"


def _nothing_matched(filters: list[str]) -> str:
    """Why the listing is empty -- an unfiltered message after filtering reports a lie.

    Naming the filters that were applied is what lets the reader know which one to drop.
    """
    if not filters:
        return "(no runs recorded yet)"
    return f"(no runs match {' '.join(filters)})"


def legacy_notice() -> str | None:
    """A one-line pointer when this directory holds runs from before the registry moved.

    Runs used to land in ``./runs``, so they are scattered across every directory anything was ever
    launched from. Saying where they are beats an empty listing that reads as data loss -- and this
    only reports: merging two checkouts' id spaces is deliberate work, not a side effect of `ls`.

    Shown only while the registry root holds nothing, which is the moment the question "where did
    my results go" is actually being asked. Once anything has been recorded there -- migrated or
    freshly run -- repeating it on every listing would be noise.
    """
    root = registry.runs_root()
    if root.is_dir() and any(child.is_dir() for child in root.iterdir()):
        return None
    legacy = registry.legacy_runs()
    if legacy is None:
        return None
    return (f"{legacy} holds runs from before the registry moved to {registry.runs_root()}. "
            f"Copy the run directories across to keep them; an id already present there belongs "
            f"to a different run, so do not overwrite it.")


#: The listing's shape. Widths are MINIMUMS -- a legacy id is four characters wider than a
#: current one, and a fixed width pushes every column after it out of line on exactly the rows
#: a reader is least likely to recognise, so `table` grows a column to its widest cell.
#: `ID` truncates on a terminal only: the handle is what a script copies, and a script gets the
#: plain rendering (or `--json`), where nothing is cut.
_COLUMNS = (
    table.Column("ID", width=_MIN_ID_WIDTH, styler=style.unit, ellipsis=True),
    table.Column("TARGET", width=_MIN_TARGET_WIDTH, styler=style.accent),
    table.Column("EVAL", width=12),
    table.Column("PLACE", width=6, styler=style.unit),
    table.Column("STATE", width=9),
    table.Column("SYNCED", width=7, styler=style.unit),
    table.Column("AGE", width=5, align=">", styler=style.unit),
    table.Column("DONE", width=5, align=">"),
    table.Column("SCORE"),
)


def _print_table(rows: list[RunRow]) -> None:
    """Print the listing -- a bordered table on a terminal, plain rows down a pipe.

    Ids are dimmed because they are handles rather than content, so the eye goes to the target
    and the state; the state carries the one colour that means something. `table` pads before
    it styles on the plain path, which is the alignment rule this listing used to enforce by
    hand at every cell.
    """
    # Annotated rather than inferred: `Cell` is a NamedTuple, so a list literal mixing it with
    # bare strings joins to `Sequence[str | Styler | None]` -- the two elements' common supertype,
    # which is not what `emit` takes. `CellLike` is the type this row actually is.
    body: list[list[table.CellLike]] = [
        [row.run_id,
         ",".join(row.targets) or "?",
         row.eval_name,
         row.place,
         table.Cell(row.state, style.state_styler(row.state)),
         row.synced,
         row.age,
         row.progress or "—",
         _composite(row)]
        for row in rows
    ]
    table.emit(_COLUMNS, body, title="Runs")


def cli_runs_ls(
    target: str | None = typer.Option(None, "--target", help="only runs of this target"),
    eval_name: str | None = typer.Option(None, "--eval", help="only runs of this eval"),
    live: bool = typer.Option(False, "--live", help="only runs that are still going"),
    org_wide: bool = typer.Option(False, "--org", help="every submitter's runs in your org, "
                                  "not just yours"),
    limit: int = typer.Option(DEFAULT_LIMIT, "--limit", min=0,
                              help="show at most this many, newest first (0 = no cap)"),
    json_out: bool = typer.Option(False, "--json", help="machine-readable JSON output"),
) -> None:
    """List runs -- yours wherever they ran, newest first.

    Registered twice: as ``runs ls`` and as the flat ``ps`` (which defaults ``--live`` on), because
    the interface model keeps flat aliases for the hot path. ONE implementation, deliberately:
    `ps` used to read local ``status.json`` files only, so it and `runs ls` disagreed about what was
    running -- a cloud sweep submitted from another directory was invisible to one and 89% complete
    according to the other.
    """
    # Ask the platform for a little more than we will show, so the cap survives the merge:
    # a platform row that duplicates a local one collapses into it, and fetching exactly `limit`
    # would leave the listing short by however many collapsed.
    rows, notices = _listing(org_wide=org_wide, fetch=(limit * 2 if limit else _NO_CAP_FETCH))
    filters: list[str] = []
    if target:
        rows = [r for r in rows if target in r.targets]
        filters.append(f"--target {target}")
    if eval_name:
        rows = [r for r in rows if r.eval_name == eval_name]
        filters.append(f"--eval {eval_name}")
    if live:
        rows = [r for r in rows if r.state in _LIVE_STATES]
        filters.append("--live")
    # Capped AFTER filtering: capping first would let a `--target` match older than the newest
    # N vanish, so the filter would answer "none" when the answer is "one, further down".
    if limit and len(rows) > limit:
        rows = rows[:limit]
        notices.append(f"showing the {limit} most recent -- `--limit 0` for all")
    legacy = legacy_notice()
    if legacy:
        notices.append(legacy)
    for notice in notices:
        style.note(notice)
    if json_out:
        style.out(json.dumps([r.as_dict() for r in rows], indent=2, sort_keys=True))
        return
    if not rows:
        style.say(_nothing_matched(filters))
        return
    _print_table(rows)


def cli_ps(
    all_runs: bool = typer.Option(False, "--all", help="include finished/failed/stale runs"),
    target: str | None = typer.Option(None, "--target", help="only runs of this target"),
    limit: int = typer.Option(DEFAULT_LIMIT, "--limit", min=0,
                              help="show at most this many, newest first (0 = no cap)"),
    json_out: bool = typer.Option(False, "--json", help="machine-readable JSON output"),
) -> None:
    """List running evals; ``--all`` also shows finished/failed/stale."""
    cli_runs_ls(target=target, eval_name=None, live=not all_runs, org_wide=False,
                limit=limit, json_out=json_out)


runs_app.command("ls")(cli_runs_ls)
