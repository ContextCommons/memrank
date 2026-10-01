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
"""Pushing a finished run's record to the org -- the half of syncing a LIBRARY run needs.

**WHY THIS IS ITS OWN MODULE.** It used to live inside :mod:`memrank.cli.sync`, which made a
finished sweep's automatic report to the org reachable only by importing the command line:
:mod:`memrank.orchestration.sweep` did `from memrank.cli import sync as sync_cli` for the sake
of three calls, and a library sweep therefore dragged the whole command-line surface --
argument parsing, terminal framework, every verb -- into a caller who asked for an evaluation.
The library cannot be the primary surface while it imports the surface being demoted.

:mod:`memrank.runs.record` records the same extraction one step earlier, for the same reason:
the RECORD was CLI-only and a cloud run's was therefore never built. This finishes that
extraction rather than choosing a new boundary -- the record and the push that carries it now
sit together, and ``memrank runs sync`` is a caller of both.

**The payload is the record.** Per-cell results with their receipts (config hash, environment
stamp, provenance pins) and the run summary: kilobytes. The per-query transcript and the
ingested corpus are artifacts -- tens of megabytes, and reproducible from the run dir that
holds them -- so they are stripped in :func:`memrank.runs.record.build` rather than shipped.

Notes go to stderr through :mod:`memrank.term.style`, which is presentation and not the
command line: a push that could not reach an org has to be able to say so on either surface.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from memrank import settings
from memrank.placement import hosted, run_api_client
from memrank.runs import record as run_record
from memrank.runs import registry
from memrank.runs import status as run_status
from memrank.term import style

#: The values of ``sync.auto`` that mean yes. Anything else -- unset, "0", "off", a typo --
#: leaves a finished run local, which is the safe direction for a setting about egress.
_AUTO_ON = ("1", "true", "yes")


def _dirty_source_run(run_dir: Path) -> bool:
    """Whether a receipt identifies mutable source that must remain on this machine."""
    for path in registry.cell_files(run_dir):
        cell = json.loads(path.read_text(encoding="utf-8"))
        provenance = cell.get("receipt", {}).get("engine_provenance", {})
        dirty = provenance.get("properties", {}).get("memrank:source_dirty")
        if str(dirty).lower() == "true":
            return True
    return False


def _tier(run_dir: Path) -> str | None:
    """The run's tier, which only the summary records."""
    summaries = sorted(run_dir.glob("summary__*.json"))
    if not summaries:
        return None
    return json.loads(summaries[0].read_text(encoding="utf-8")).get("tier")


def push_one(http, org: str, run_dir: Path, data: dict | None) -> None:
    """PUT one finished run and mark it synced. Raises; callers decide how loud that is."""
    if _dirty_source_run(run_dir):
        raise run_api_client.RunApiError(
            f"{run_dir.name} evaluated a dirty source tree and is local-only", code="unsyncable")
    if data is None or not (data.get("target") and data.get("benchmark")
                            and data.get("started_at")):
        raise run_api_client.RunApiError(
            f"{run_dir.name} has no complete heartbeat to sync (it predates the heartbeat, "
            f"or died before recording one)", code="unsyncable")
    record = run_record.build(run_dir)
    if data.get("experiment_id"):
        record["experiment"] = {
            key: data.get(key) for key in (
                "experiment_id", "experiment_label", "plan_hash", "experiment_spec")}
    payload = {"target_ref": data["target"], "benchmark": data["benchmark"],
               "slice": data.get("slice"), "tier": _tier(run_dir),
               "state": run_status.classify(data),
               "started_at": data["started_at"],
               # Read BEFORE the marker is written: annotate() restamps updated_at, and the
               # run finished when it finished, not when this reached the API.
               "finished_at": data.get("updated_at"), "error": data.get("error"),
               "record": record}
    run_api_client.sync_run(http, org, run_dir.name, payload)
    run_status.RunStatus(run_dir, data).annotate(
        synced_org=org, synced_at=datetime.now(timezone.utc).isoformat())


def auto_sync(run_dir: Path) -> None:
    """Push one just-finished run, best effort. NEVER raises into the run that called it.

    The run is already durably recorded before this runs, exactly as the MLflow mirror is --
    but where that mirror raises when it is enabled and broken, this degrades to a note. The
    difference is that a failed sync has a standing retry (`memrank runs sync`) and stays
    visible as pending in the listing, so nothing is lost quietly; failing a finished
    evaluation over a flaky network would destroy something that cannot be recovered.

    With no hosted side it is silent, by contract: local-first use accumulates records on the
    machine and nothing nags. Which reasons count as "no hosted side" is
    :mod:`memrank.placement.hosted`'s to decide, not this hook's.
    """
    if (settings.get("sync.auto") or "").strip().lower() not in _AUTO_ON:
        return
    org = None
    try:
        with run_api_client.authenticated_client() as http:
            org = settings.get("defaults.org")
            if org is None:
                _unsynced(hosted.no_org())
                return
            push_one(http, org, run_dir, run_status.read(run_dir))
    except run_api_client.RunApiError as exc:
        _unsynced(hosted.refused(exc))
    except Exception as exc:  # noqa: BLE001 - reported, retryable, and never fatal to a run
        _unsynced(hosted.unreachable(exc))
    else:
        style.note(f"synced -> org {org}")


def _unsynced(problem: hosted.Unavailable) -> None:
    """Say why a finished run did not reach an org -- when there is an org to have missed.

    A retry exists (`memrank runs sync`) and the run stays visible as pending in the listing, so
    the note is a courtesy rather than the only record; withholding it where nothing hosted was
    ever in play loses nothing and keeps a local evaluation's output about the evaluation.
    """
    if hosted.worth_saying(problem):
        style.note(f"not synced to the org: {problem} -- `memrank runs sync` retries")
