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
"""Cloud submission: describe the sweep for the platform and record what it answered.
"""
from __future__ import annotations

from typing import Any

import typer

from memrank.benchmarks.refs import canonical_ref
from memrank.orchestration.resolve import _experiment_metadata
from memrank.placement.remote_argv import submit_command
from memrank.runs import registry
from memrank.runs import status as run_status
from memrank.term import style


def _api_client():
    """An authenticated client bound to the memrank API, or a refusal to guess.

    The client itself lives in ``run_api_client`` (``runs_cli`` needs it and cannot import
    this module); what stays here is submission's own phrasing of "you are not signed in".
    """
    from memrank.placement.run_api_client import RunApiError, authenticated_client

    try:
        return authenticated_client()
    except RunApiError as exc:
        if exc.code == "credential_store":
            # Telling someone whose keychain refused to "log in again" sends them to redo work
            # that is already done; the session is fine, this machine could not read it.
            raise typer.BadParameter(str(exc)) from exc
        raise typer.BadParameter(
            "--on cloud submits through the memrank API; run `memrank auth login` first"
        ) from exc


def _submit_sweep(refs: list[str], params: dict[str, Any], *, org: str | None,
                  benchmark: str, slice_: str | None, tier: str | None) -> None:
    """Submit one run per target THROUGH THE API -- the laptop holds no AWS identity.

    A sweep fans out (one task cannot serve mixed shapes); the server launches each and records it
    against the org. What crosses the wire is the MEASUREMENT -- the resolved cell as a structured
    config -- and the server renders the command from it with the renderer its own platform image
    was built beside. So there is no contract to negotiate: this CLI's flag vocabulary is its own
    business, and a flag renamed on the platform moves the server's renderer alone.

    NO image is ever named either: the server runs its pinned platform harness.

    Naming a harness build used to be possible here (``--image-tag``), and it is not any more. It
    assumed a memrank checkout, a Docker daemon and ECR push rights -- none of which a user of the
    instrument has -- and it only ever swapped half the run: ``remote_command`` is rendered by the
    API from ITS build, so a branch-built harness executed inside the deployed command shape. The
    way to test unmerged changes against the cloud is `dev`, CI and staging.

    Raises:
        typer.Exit: 1 when any target was refused -- after every other target has been
            attempted. One refusal says nothing about the next target: a transient AWS
            capacity shortfall on the third of eight used to cost the remaining five their
            submission. Independent runs are submitted independently, and the refusals are
            named together at the end.
    """
    from memrank.application.submission import remote_config
    from memrank.placement import run_api_client
    from memrank.targets.portability import local_only, unportable_message

    if org is None:
        raise typer.BadParameter("--on cloud needs --org <slug>: runs are submitted through "
                                 "the memrank API and recorded against an org")

    # Before the first submission, not per target as the capacity refusal below is. That one is
    # transient and says nothing about the next target; this one is deterministic and already
    # decided, so launching the portable half would only leave a sweep half-measured for a
    # condition the command could have been told about while it was still free to fix.
    local = local_only(refs)
    if local:
        raise typer.BadParameter(unportable_message(local) + " Nothing was submitted.")

    refused: list[str] = []
    launched: list[str] = []
    with _api_client() as http:
        for ref in refs:
            experiment = (params.get("experiments_by_ref") or {})[ref]
            payload = {"config": remote_config(experiment, verbose=params["verbose"],
                                               judge_workers=params["judge_workers"],
                                               fail_fast=params["fail_fast"])}
            try:
                record = run_api_client.submit_run(http, org, payload)
            except run_api_client.RunApiError as exc:
                # Named now, tallied at the end, and the sweep goes on: whether AWS had
                # capacity for THIS target says nothing about the next one.
                style.error(f"could not submit {ref!r}: {exc}")
                refused.append(ref)
                continue
            launched.append(ref)
            _record_submission(http, org, ref, record, benchmark=benchmark, slice_=slice_,
                               tier=tier, experiment=experiment,
                               plan_hash=params.get("plan_hash"))
    if refused:
        _report_refusals(refused, launched, params=params, org=org)
        raise typer.Exit(1)


def _report_refusals(refused: list[str], launched: list[str], *, params: dict[str, Any],
                     org: str) -> None:
    """Close a partly-refused sweep by naming both halves and the command that finishes it.

    Each refusal already printed its own reason as it happened, so this counts rather than
    repeats -- and says what IS running, because these are independent runs: the launched
    ones keep going, and the refused ones need a resubmission, not a rollback.

    The retry line is RENDERED from the submission, never rebuilt from remembered fields. It used to
    be assembled here out of benchmark, slice and tier, which meant a sweep submitted with
    ``--judge --ack-egress`` was offered back without them: paste the suggestion after a capacity
    503 and the replacement runs are unjudged, scoring a different question with no error anywhere.
    """
    style.error(f"{len(refused)} of {len(refused) + len(launched)} targets were refused: "
                f"{', '.join(refused)}")
    style.say(f"  still running: {', '.join(launched) or 'none'}")
    retry = submit_command(params, refs=refused, on="cloud", org=org)
    style.say(f"  resubmit those: {' '.join(retry)}")


def _record_submission(http, org: str, ref: str, record: dict, *, benchmark: str,
                       slice_: str | None, tier: str | None, experiment=None,
                       plan_hash: str | None = None) -> None:
    """Write the local run record for one server-minted run and print its handle.

    A local ``runs/<id>/status.json`` is written at submit time so ``memrank ps`` sees the
    run immediately, carrying the task ARN and no pid -- the pid would belong to a process
    inside Fargate. The id is the SERVER's: local dir, S3 prefix, and the org's runs row
    all agree on one name. The bare id goes to stdout -- the machine-readable handle a
    script pipes into `memrank watch` -- and the human context to stderr.
    """
    run_dir = registry.new_run_dir(benchmark, slice_, tier, run_id=record["id"])
    status = run_status.RunStatus.create(
        run_dir, target=ref, benchmark=benchmark, slice_=slice_,
        eval_ref=canonical_ref(benchmark, {"tier": tier, "slice": slice_}))
    if experiment is not None:
        status.annotate(**_experiment_metadata(experiment, plan_hash))
    status.mark_remote(placement="cloud", task_arn=record["task_arn"],
                       region=record["region"], cluster=record["cluster"],
                       log_group=record["log_group"],
                       artifact_bucket=record["artifact"]["bucket"])
    status.update(state="running", message="submitted via API")

    task_id = record["task_arn"].rsplit("/", 1)[-1]
    style.out(run_dir.name)
    style.say(f"submitted {run_dir.name} (task {task_id})")
    style.say(f"  logs:  memrank logs {run_dir.name}")
    style.say(f"  track: memrank watch {run_dir.name}")
