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
"""Cloud submission for the MCP server's config path: an API client, and the local record of
what the platform answered. The CLI's own argv submission is gone
(ATO-2366).
"""
from __future__ import annotations

import typer

from memrank.benchmarks.refs import canonical_ref
from memrank.orchestration.resolve import _experiment_metadata
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
            "a cloud run is submitted through the memrank API; run `memrank auth login` first"
        ) from exc


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
