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
"""``memrank runs show`` -- one run in full, from whichever stores know it.

Split out of :mod:`memrank.cli.runs` because the listing and the detail view answer different
questions and neither needs the other's helpers: ``ls`` merges two stores into rows, ``show``
prints one run's state and results. The shared vocabulary lives with what owns it --
:mod:`memrank.runs.status` names the states, :mod:`memrank.runs.registry` groups the cells -- so
this module and the listing cannot drift into two spellings of one condition.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import typer

from memrank import (
    settings,
)
from memrank.cli import monitor as monitor_cli
from memrank.cli import runs_show_agent
from memrank.metrics import headline
from memrank.placement import run_api_client
from memrank.runs import reconcile, registry
from memrank.runs import status as run_status
from memrank.term import detail, fmt, style


def platform_record(run_id: str) -> dict | None:
    """The org's record for ``run_id``, or ``None`` when there is none to be had.

    Silent about every reason it might be absent -- not signed in, no default org, the run is
    local-only, the API is unreachable. ``show`` still has the local record to print, and the
    caller did not ask a remote question; `auth status` is where a broken session is diagnosed.
    """
    org = settings.get("defaults.org")
    if org is None:
        return None
    try:
        with run_api_client.authenticated_client() as http:
            return run_api_client.get_run(http, org, run_id)
    except Exception:  # noqa: BLE001 - absence is an answer here, not a failure
        return None


def _field(name: str, rendered: str) -> None:
    """One ``label: value`` line printed OUTSIDE the status panel -- the results line.

    The panel sizes its own labels; this is the one pair that follows it, so it keeps the width
    the panel would have given it rather than shrinking to its own single label.
    """
    style.out(f"  {style.pad(name, 9, style.label)}  {rendered}")


def _render_platform(payload: dict) -> None:
    """The org's view of a run: state, where it ran, and how to reach its output."""
    state = run_status.PLATFORM_STATES.get(payload.get("state", ""), run_status.UNKNOWN_STATE)
    fields = [
        ("state", f"{style.state(state)} {style.unit('(' + str(payload.get('state')) + ')')}"),
        ("agent" if runs_show_agent.is_agent(payload) else "cell",
         f"{style.accent(str(payload.get('target_ref')))} × "
         f"{style.accent(str(payload.get('benchmark')))}"),
    ]
    # A synced local run is in the org universe but never had a cluster; naming one would be
    # the interface lying about where a number came from, which is the whole point of place.
    if payload.get("place") == "local":
        fields.append(("place", f"local {style.unit('(synced from the machine that ran it)')}"))
    else:
        where = f"(cluster {payload.get('cluster', '?')}, {payload.get('region', '?')})"
        fields.append(("place", f"cloud {style.unit(where)}"))
    fields.append(("started", style.unit(str(payload.get("created_at")))))
    if payload.get("exit_code") is not None:
        code = payload["exit_code"]
        shown = style.good(str(code)) if code == 0 else style.bad(str(code))
        fields.append(("exit",
                       f"{shown} {style.unit(str(payload.get('stopped_reason', '')))}".rstrip()))
    artifact = payload.get("artifact") or {}
    if artifact.get("bucket"):
        fields.append(("artifact",
                       style.unit(f"s3://{artifact['bucket']}/{artifact.get('prefix', '')}")))
    detail.emit(detail.panel(payload["id"], fields))


def pull_results(run_id: str, record: dict) -> str | None:
    """Bring a finished cloud run's results down, so `show` can print what it just called done.

    Returns the reason there are still no results, or ``None`` once they are here.

    `show` reads results from disk and state from the org, and a cloud run submitted without being
    watched has the second and not the first -- which used to print a live ``done`` directly above
    ``(none recorded yet)``. Fetching here is what `watch` would have done; it is idempotent, and
    the network trip already existed to get the state.

    Deliberately OUTSIDE :func:`platform_record`'s silent try/except: absence of a record is an
    answer, but a refused download is not, and swallowing it would report a successful run as
    having produced nothing.
    """
    org = settings.get("defaults.org")
    if org is None:
        return "signed out -- `memrank auth login`"
    run_dir = registry.runs_root() / run_id
    try:
        with run_api_client.authenticated_client() as http:
            state = reconcile.reconcile(
                http, org, run_dir, record,
                report=lambda what: style.say(f"  fetching: {what}"))
    except run_api_client.RunApiError as exc:
        return f"the org refused them ({exc})"
    return None if state == "done" else f"the run is {state}"


#: Org states worth attempting a results fetch for. ``stopped-success`` is the platform saying the
#: run succeeded; ``unknown`` is the platform saying it can no longer tell, which is where every
#: unwatched cloud run ends up once ECS forgets the task. reconcile() settles the second case from
#: the artifacts themselves.
_FETCHABLE_STATES = ("stopped-success", "unknown")


def _render_metrics(groups: dict) -> None:
    """One cell's metrics, grouped. Absent fields never appear -- see `cell_metrics`.

    Labels dimmed, values plain: everything around a number is quieter than the number, so the
    default foreground becomes the emphasis without picking a colour that fails on light or dark
    terminals. Padding happens through `style.pad`, which pads before styling -- dimming a padded
    cell the other way round misaligns every column while every substring test still passes.
    """
    for group, fields in groups.items():
        detail.emit(detail.rule(group, indent=6, width=56))
        noun = registry.RATE_NOUNS.get(group, "item")
        detail.emit(detail.fields(
            [(name, style.value(fmt.value(raw, kind, noun=noun)))
             for name, (raw, kind) in fields.items()], indent=8))


@dataclass(frozen=True)
class _Gathered:
    """What every store knows about one run -- read once, rendered either way.

    ``unfetched`` is the reason results are still not here for a run the org calls finished; it
    is a fact about the answer, not a rendering detail, so it belongs in both forms.
    """

    run_id: str
    heartbeat: dict | None
    platform: dict | None
    cells: list
    unfetched: str | None


def _gather(run_id: str) -> _Gathered:
    """Every store's view of ``run_id``, fetching results the org has and this machine does not.

    Raises:
        typer.BadParameter: When neither this machine nor the org has heard of it.
    """
    run_dir = registry.runs_root() / run_id
    heartbeat = run_status.read(run_dir) if run_dir.is_dir() else None
    cells = registry.cells_by_run().get(run_id, [])
    platform = platform_record(run_id)
    if heartbeat is None and not cells and platform is None:
        raise typer.BadParameter(f"no run {run_id!r} recorded here or in your org")
    unfetched = None
    # `unknown` is included deliberately: it is what a cloud run ages into once ECS forgets the
    # task, which every unwatched or overnight run eventually does. Gating on `stopped-success`
    # alone left finished runs unreadable with their results already in the bucket. reconcile()
    # decides from the artifacts whether such a run actually finished, so asking is cheap and
    # honest -- a run that has produced nothing still renders its reason rather than results.
    # A `memrank run` run has no cells to fetch: its result is in the org's record itself.
    if (not cells and platform is not None and not runs_show_agent.is_agent(platform)
            and platform.get("state") in _FETCHABLE_STATES):
        unfetched = pull_results(run_id, platform)
        cells = registry.cells_by_run().get(run_id, [])
    return _Gathered(run_id=run_id, heartbeat=heartbeat, platform=platform, cells=cells,
                     unfetched=unfetched)


def _describe(found: _Gathered, *, full: bool) -> dict:
    """One run as a plain dict -- the shape ``--json`` emits.

    Both renderings answer from :func:`_gather`, but only this one is a contract: a script reads
    ``state`` and ``results``, so the state word is the SAME one the terminal shows
    (``run_status.classify`` locally, ``PLATFORM_STATES`` for the org's record) rather than the
    raw store value, which differs between the two stores for the same condition.

    ``metrics`` keeps each value's KIND beside it, exactly as :func:`registry.cell_metrics`
    reports it -- a consumer that formats seconds as bytes is the bug the kind exists to prevent.
    """
    platform, heartbeat = found.platform, found.heartbeat
    if platform is not None:
        state = run_status.PLATFORM_STATES.get(platform.get("state", ""), run_status.UNKNOWN_STATE)
    elif heartbeat is not None:
        state = run_status.classify(heartbeat)
    else:
        state = run_status.UNKNOWN_STATE
    described = {
        "run_id": found.run_id,
        "state": state,
        "place": (platform or heartbeat or {}).get("place")
                 or (heartbeat or {}).get("placement") or "local",
        "org_record": platform,
        "heartbeat": heartbeat,
        "results": [_describe_cell(cell, full=full) for cell in found.cells],
        "results_unavailable": found.unfetched,
    }
    if runs_show_agent.is_agent(platform):
        described["agent_result"] = runs_show_agent.describe(found.run_id, platform, full=full)
    return described


def _describe_cell(cell, *, full: bool) -> dict:
    """One recorded cell: what it measured, and the conditions it measured under.

    ``environment`` is not behind ``--full``. It is what decides whether these numbers may be
    compared with another run's -- a consumer that has the latency and not the conditions has
    exactly the half that misleads.
    """
    described = {"target": cell.adapter, "eval": cell.benchmark, "composite": cell.composite,
                 "config_hash": cell.config_hash, "path": str(cell.path),
                 "environment": registry.cell_environment(cell.path)}
    if full:
        described["metrics"] = {
            group: {name: {"value": raw, "kind": kind} for name, (raw, kind) in fields.items()}
            for group, fields in registry.cell_metrics(cell.path).items()}
    return described


def show(run_id: str = typer.Argument(..., help="run-id (from `memrank runs ls`)"),
         full: bool = typer.Option(False, "--full",
                                   help="every metric each cell recorded, not just its score"),
         json_out: bool = typer.Option(False, "--json", help="machine-readable JSON output"),
         ) -> None:
    """Show one run in full: its state, and the results it recorded.

    ``--full`` expands each cell into what its artifact actually holds -- quality, latency
    percentiles, cost and the engine's provenance. Without it a run reports one number and a file
    path, which answers "did it finish" and nothing else. ``--json`` emits the same answer as a
    document: this is the record view, so it is the one a script or an agent reaches for first.
    """
    found = _gather(run_id)
    if json_out:
        style.out(json.dumps(_describe(found, full=full), indent=2, sort_keys=True, default=str))
        return
    _render(found, full=full)


def _render(found: _Gathered, *, full: bool) -> None:
    """The terminal form: the state block, then the cells the run recorded."""
    run_id, data, platform, cells = found.run_id, found.heartbeat, found.platform, found.cells
    if runs_show_agent.is_agent(platform):
        _render_platform(platform)
        runs_show_agent.render(run_id, platform, full=full)
        return
    if platform is not None:
        _render_platform(platform)
    elif data is not None:
        monitor_cli.render_status(run_id, data)
    else:
        detail.emit(detail.panel(run_id, [
            ("state", f"{run_status.UNKNOWN_STATE} "
                      f"{style.unit('(this run recorded no heartbeat)')}")]))
    if not cells:
        # "(none recorded yet)" is only true when nothing was produced. A run whose numbers are in
        # the org and not here has results; saying otherwise sends a reader looking for a failure
        # that did not happen.
        _field("results", style.caution(f"in the org, not on this machine -- {found.unfetched}")
               if found.unfetched else style.unit("(none recorded yet)"))
        return
    style.out(f"  {style.label('results')}")
    for cell in cells:
        # The headline, not the raw composite: on locomo, longmemeval and beam the composite is
        # withheld by the methodology and the judged score is the run's only quality number, so
        # reading `composite` here reported a judged run as having measured nothing.
        composite = f"{cell.headline:.4f}" if cell.headline is not None else "—"
        judged = " (judged)" if cell.headline_kind == headline.JUDGED else ""
        chash = (cell.config_hash or "")[:8] or "—"
        # `composite=` and `config=` keep their equals signs. The shape is a contract -- something
        # may be grepping it -- and this change is about colour, not about renaming fields.
        style.out(f"    {style.accent(f'{cell.adapter}×{cell.benchmark}')}  "
                  f"{style.label('composite=')}{composite}{style.unit(judged)}  "
                  f"{style.label('config=')}{style.unit(chash)}")
        _render_environment(registry.cell_environment(cell.path))
        style.out(f"      {style.unit(str(cell.path))}")
        if full:
            _render_metrics(registry.cell_metrics(cell.path))


def _render_environment(env: dict) -> None:
    """One line naming where the numbers above came from, in the DEFAULT view.

    Not behind ``--full``, unlike the metrics: place and machine are what decide whether a
    latency here may be set beside a latency there, so a reader who sees the number must see the
    condition. Silent for artifacts written before schema 2 -- inventing "local" for a run that
    recorded nothing would be the interface lying about exactly what this field exists to say.
    """
    if not env.get("place"):
        return
    machine = f"{env.get('arch', '?')}/{env.get('os', '?')}"
    style.out(f"      {style.label('env=')}{env['place']} {style.unit(f'({machine})')}")
