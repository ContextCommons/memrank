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
"""Launch one evaluation as a one-shot ECS Fargate task.

Cloud is not a :class:`~memrank.placement.base.Placement`. Locally memrank runs on your machine and
provisions an engine beside it -- provision, get an endpoint, run, tear down. Under ECS memrank runs
*inside* the task alongside the engine: there is no endpoint to hand back, nothing to tear down, and
the evaluation never executes on the submitting machine. This is a **submission**, so it gets its own
module rather than pretending to satisfy a protocol whose shape does not fit.

Originally ported from the retired ``scripts/cloud-run.sh``. The behaviour worth preserving is
recorded in the comments below; the reasons are load-bearing and were each learned from a failure.
Waiting/watching moved to the accounts API (GET /orgs/{org}/runs/{id}), which reads task state via
:func:`describe_many`.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from memrank.errors import MemrankError
from memrank.runs.checkpoint import CHECKPOINT_FILE

# Where the task writes results, and where upload_results.py reads them. A local --output-dir is
# meaningless inside the container, so it is always rewritten to this.
REMOTE_OUTPUT_DIR = "/work/results"
# Baked into the runner image at deploy/Dockerfile:48.
_UPLOAD_SCRIPT = "/opt/memrank/upload_results.py"
# S3 prefix for raw cloud-run artifacts. NEVER "runs/" -- that belongs to leaderboard-gc, which
# deletes any object there without a matching lb.run_artifacts row (deploy/upload_results.py:28-32).
_ARTIFACT_PREFIX = "cloud-runs"
# Run ids become both a shell token and an S3 key. run_registry.new_run_dir() only ever produces
# this alphabet; anything else is rejected rather than escaped, because the safe set is small.
_SAFE_RUN_ID = re.compile(r"\A[A-Za-z0-9._-]+\Z")

# Flags that describe the SUBMITTER's behaviour and mean nothing inside the task. `--on` is the
# dangerous one: left in place, the task would submit another task, and so on. (`--wait` and
# `--detach` used to be stripped here too; both flags no longer exist -- blocking is the `watch`
# verb, detached is the only lifecycle.)
_LOCAL_ONLY_FLAGS: tuple[str, ...] = ("--on", "--output-dir", "--run-id")


# The shape of the command sent to a task, as a version. BUMP THIS whenever `remote_command()`
# changes what it emits -- a renamed flag, a newly required option, positional arguments moving.
#
# It exists because the submitter renders a command against ITS OWN CLI and ships it to an image
# that may have been built from different source. On 2026-07-30 that skew cost a live run: the
# newest image in ECR was three days old, its `memrank run` took `--adapter` as an option and had
# no `--run-id`, and the task died parsing its arguments -- after the image pull and task start.
# Same lesson as CONFIG_HASH_VERSION in memrank/provenance/receipt.py (via Inspect AI's
# TASK_IDENTIFIER_VERSION): when two sides must agree on a format, version the format rather than
# inferring agreement from behaviour.
#: 2 -- the verb became ``submit`` (was ``run``), and the rendered command now pins ``--on none``.
#: 3 -- the evaluation is named solely by its eval ref (``beam:100k-smoke``); ``--tier`` and
#: ``--slice`` are retired and no longer rendered. An image older than the retirement cannot
#: parse ref-only argv, and a CLI older than it renders flags the current image refuses -- the
#: retirement itself shipped without a bump, and a run died parsing its own command on
#: 2026-08-13. Both skews are refused at submission now.
#: 4 -- ``--max-judge-calls`` is retired. The judge call cap is gone entirely: it bounded the
#: MEASUREMENT rather than the system under test, so which queries got graded would have depended
#: on where a counter ran out. An older CLI still renders the flag and the current image cannot
#: parse it -- the same skew as 3, bumped this time rather than discovered by a dead task.
#: 5 -- ``--ack-egress`` is retired with the egress acknowledgement itself. ``--judge`` already names
#: the provider it sends to, and the browser path never asked at all. Same skew, same reason.
#: 6 -- judging is DERIVED from the eval, so ``--judge`` became one half of ``--judge/--no-judge``
#: and the submitter now renders whichever half it resolved. An older image has no ``--no-judge``
#: to parse; a newer one would otherwise re-derive the default against its own build of the
#: benchmark, which is the same command meaning two things. Bumped so neither can happen.
REMOTE_CLI_CONTRACT = 6
# Where the constant lives, read out of the image's own commit. Kept next to the constant so the
# two move together.
_CONTRACT_SOURCE = "memrank/placement/cloud_submit.py"


class CloudLaunchError(MemrankError):
    """A task could not be registered or started."""


class ImageContractError(CloudLaunchError):
    """The runner image cannot parse the command this submission would send it."""


class ImageMissingError(CloudLaunchError):
    """The runner image is not in the registry, so the task would die pulling it."""


class CloudCapacityError(CloudLaunchError):
    """AWS had no Fargate capacity to place the task. Transient, and nothing is wrong here.

    Distinguished from every other launch failure because the answer is different: a
    misconfiguration needs fixing, this needs asking again. We launch into a single AZ on
    purpose (one subnet keeps latency comparable between runs of a cell), so a shortage in
    that zone refuses the whole submission rather than being routed around.
    """


#: What ECS says in a ``failures[].reason`` when the zone is full. Matched on the stable
#: prefix -- the sentence carries a trailing "Please try again later or in a different
#: availability zone" that has changed wording before.
_CAPACITY_REASON = "capacity is unavailable"


@dataclass(frozen=True)
class LaunchTarget:
    """Where to run a task. Distinct from :class:`~memrank.placement.cloud.AwsContext`, which
    describes the task *definition*; this is the cluster and network it runs on."""

    region: str
    cluster: str
    subnet: str
    security_group: str


@dataclass(frozen=True)
class TaskOutcome:
    """How a task ended. ``exit_code`` is ``None`` while it is still running.

    ``stopped_at`` is ECS's own answer to WHEN, and it is carried rather than derived because the
    caller learns of a stop only when someone happens to look. Deriving it from the observation
    would date every stop from the first read -- a run that ended at 14:00 and was first read at
    22:00 would be recorded as stopping at 22:00, which is plausible, wrong, and silent.
    """

    last_status: str
    exit_code: int | None = None
    stopped_reason: str = ""
    stopped_at: datetime | None = None

    @property
    def stopped(self) -> bool:
        return self.last_status == "STOPPED"

    @property
    def succeeded(self) -> bool:
        return self.stopped and self.exit_code == 0


def _strip_local_flags(argv: list[str]) -> list[str]:
    """Drop the flags that only mean something to the submitting process."""
    out: list[str] = []
    skip_next = False
    for arg in argv:
        if skip_next:
            skip_next = False
            continue
        # Both spellings: `--on cloud` and `--on=cloud`. Handling only the spaced form is how a
        # stripped-argv guard quietly stops working.
        if arg in _LOCAL_ONLY_FLAGS:
            skip_next = True
            continue
        if any(arg.startswith(f"{flag}=") for flag in _LOCAL_ONLY_FLAGS):
            continue
        out.append(arg)
    return out


def remote_command(argv: list[str], *, bucket: str, run_id: str) -> str:
    """The shell command the task runs: evaluate, then upload on success.

    Args:
        argv: The submitter's ``memrank`` arguments, starting at ``run``.
        bucket: Artifact bucket the results are uploaded to.
        run_id: The submitter's run id, reused as the S3 prefix so ``runs/<id>`` locally and
            ``cloud-runs/<id>`` in S3 name the same run.

    Returns:
        A single shell string. The image's ``ENTRYPOINT`` is ``memrank``; the task definition
        overrides ``entryPoint`` to a shell so the upload can be chained.

    Raises:
        ValueError: If ``run_id`` contains anything outside ``[A-Za-z0-9._-]``. It is interpolated
            into a shell string and an S3 key, so the safe alphabet is enforced rather than escaped.
    """
    if not _SAFE_RUN_ID.match(run_id):
        raise ValueError(
            f"unsafe run id {run_id!r}: it becomes both a shell token and an S3 key, so only "
            f"letters, digits, dot, dash and underscore are allowed.")

    args = " ".join(shlex.quote(a) for a in _strip_local_flags(argv))
    # `--on none` is appended by the RENDERER, after stripping, and never carried in argv: the
    # submitter's own `--on` is dropped above precisely so a task cannot inherit "cloud" and
    # submit another task. Stripping alone stops being enough the moment placement can come from
    # configuration instead of a flag -- there is no flag to strip then, and the shipped default
    # is cloud. Saying it here means the task's placement is decided by the thing rendering the
    # command, which is the only party that knows the task already has its engines beside it.
    # Where to publish live progress, as an environment prefix rather than a flag. Deliberately
    # NOT a REMOTE_CLI_CONTRACT bump: the contract guards against an older image failing to PARSE
    # what it is sent, and an unknown environment variable cannot do that. An image built before
    # this simply publishes nothing -- the evaluation itself is identical, so no run is degraded,
    # only unobserved.
    evaluate = (f"MEMRANK_PROGRESS_BUCKET={shlex.quote(bucket)} "
                f"memrank {args} --on none "
                f"--output-dir {REMOTE_OUTPUT_DIR} --run-id {run_id}")
    upload = (f"python {_UPLOAD_SCRIPT} --results-dir {REMOTE_OUTPUT_DIR} "
              f"--bucket {bucket} --prefix {_ARTIFACT_PREFIX}/{run_id}")
    # On failure, publish the retrieval checkpoint and NOTHING else, then still fail the task.
    # A run that dies in the judge stage has already paid for every ingest and retrieve -- 4h50m of
    # them on 2026-08-12 -- and without this the cloud path stays disposable while local runs become
    # resumable, which is backwards: the long runs are the cloud ones.
    rescue = (f"python {_UPLOAD_SCRIPT} --results-dir {REMOTE_OUTPUT_DIR} "
              f"--bucket {bucket} --prefix {_ARTIFACT_PREFIX}/{run_id} "
              f"--only {CHECKPOINT_FILE}")
    # `&&`, never `;`: a failed evaluation must not publish a partial artifact. The rescue names
    # ONE file for that reason -- the artifact beside it may be half-written -- and re-raises the
    # failure so the task is still marked failed.
    return f"{evaluate} && {upload} || ( {rescue}; exit 1 )"


def verify_image_exists(*, repository: str, tag: str, region: str) -> str:
    """Confirm the runner image is actually in the registry, and return its digest.

    The contract check above proves an image built from ``tag`` *would* understand the command;
    this proves one was ever pushed. They are different failures with the same cost if missed:
    ECS pulls, retries seven times, and reports ``CannotPullContainerError`` -- a task billed for
    discovering something a single registry read answers first.

    Args:
        repository: Full ECR repository URI (``<acct>.dkr.ecr.<region>.amazonaws.com/<name>``).
        tag: The image tag to look for.
        region: AWS region of the registry.

    Returns:
        The image's immutable manifest digest -- the real reproducibility pin, worth more in a
        receipt than a mutable tag.

    Raises:
        ImageMissingError: When no image carries that tag, listing what IS available so the fix is
            obvious rather than a guess.
    """
    import boto3
    from botocore.exceptions import ClientError

    name = repository.rsplit("/", 1)[-1]
    ecr = boto3.client("ecr", region_name=region)
    try:
        found = ecr.describe_images(repositoryName=name, imageIds=[{"imageTag": tag}])
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") != "ImageNotFoundException":
            raise
        raise ImageMissingError(
            f"image {name}:{tag} is not in the registry, so the task would fail pulling it "
            f"(CannotPullContainerError). {_available_tags_hint(ecr, name)} "
            f"Build the image and push it to {repository} under that tag, or submit one of the "
            f"tags that is already there.") from exc
    return found["imageDetails"][0].get("imageDigest", "")


def _available_tags_hint(ecr: Any, name: str) -> str:
    """The tags that DO exist, so a missing-image error names the alternatives."""
    try:
        images = ecr.describe_images(repositoryName=name)["imageDetails"]
    except Exception:                                   # noqa: BLE001 - a hint, never the failure
        return ""
    tags = [t for image in sorted(images, key=lambda i: i.get("imagePushedAt"), reverse=True)
            for t in image.get("imageTags", [])][:5]
    return f"Available tags: {', '.join(tags)}." if tags else "The repository has no tagged images."


def _client(launch: LaunchTarget) -> Any:
    """An ECS client. Lazy import and no explicit session, matching storage_client.py -- botocore
    resolves region and credentials from the standard chain and fails loudly if they are absent."""
    import boto3

    return boto3.client("ecs", region_name=launch.region)


def register(taskdef: dict[str, Any], launch: LaunchTarget) -> str:
    """Register a task definition and return its ARN.

    A fresh definition per run, because ``run-task`` overrides can change command, environment and
    sizing but NOT the image -- so a run pinned to a specific image tag needs its own.
    """
    try:
        response = _client(launch).register_task_definition(**taskdef)
    except Exception as exc:                                   # noqa: BLE001 - re-raised with context
        raise CloudLaunchError(f"could not register task definition: {exc}") from exc
    return response["taskDefinition"]["taskDefinitionArn"]


def submit(task_definition_arn: str, launch: LaunchTarget) -> str:
    """Start one task and return its ARN."""
    try:
        response = _client(launch).run_task(
            cluster=launch.cluster,
            taskDefinition=task_definition_arn,
            # FARGATE, never FARGATE_SPOT: a Spot reclamation mid-run discards the whole
            # evaluation, and runs are long.
            launchType="FARGATE",
            networkConfiguration={"awsvpcConfiguration": {
                "subnets": [launch.subnet],
                "securityGroups": [launch.security_group],
                # What lets the task reach ECR and the internet without a NAT gateway
                # (~$32/month idle). The security group has no ingress rules.
                "assignPublicIp": "ENABLED"}})
    except Exception as exc:                                   # noqa: BLE001 - re-raised with context
        raise CloudLaunchError(f"could not start task: {exc}") from exc

    failures = response.get("failures") or []
    if failures or not response.get("tasks"):
        reasons = " ".join(str(f.get("reason", "")) for f in failures)
        if _CAPACITY_REASON in reasons.lower():
            raise CloudCapacityError(
                f"AWS has no Fargate capacity in the benchmark's availability zone "
                f"(subnet {launch.subnet}) right now. This is transient and nothing is "
                f"misconfigured -- submit again, or submit fewer targets at once.")
        raise CloudLaunchError(f"run-task started nothing: {failures or response}")
    return response["tasks"][0]["taskArn"]


def _outcome_of(task: dict) -> TaskOutcome:
    """One task's reading from a describe-tasks entry -- the single parsing of that shape."""
    containers = [c for c in task.get("containers", []) if c.get("name") == "memrank"]
    return TaskOutcome(last_status=task.get("lastStatus", "UNKNOWN"),
                       exit_code=containers[0].get("exitCode") if containers else None,
                       stopped_reason=task.get("stoppedReason", ""),
                       # Absent while the task runs. boto3 has already parsed it to an aware
                       # datetime; there is nothing to convert and nothing to default.
                       stopped_at=task.get("stoppedAt"))


def describe_many(task_arns: list[str], launch: LaunchTarget) -> dict[str, TaskOutcome]:
    """Point-in-time readings for many tasks, batched at the API's 100-ARN limit.

    A task ECS no longer knows (stopped >~1h ago) is simply absent from the result -- the
    caller decides what absence means (the accounts API reports it as ``unknown``).
    """
    client = _client(launch)
    readings: dict[str, TaskOutcome] = {}
    for start in range(0, len(task_arns), 100):
        chunk = task_arns[start:start + 100]
        response = client.describe_tasks(cluster=launch.cluster, tasks=chunk)
        for task in response.get("tasks") or []:
            readings[task["taskArn"]] = _outcome_of(task)
    return readings


def stop_task(task_arn: str, launch: LaunchTarget, *, reason: str) -> None:
    """Ask ECS to stop one task.

    Asynchronous by design: ECS SIGTERMs the containers and SIGKILLs after the stop grace
    period, so the task WILL end -- but not before this returns. The caller must not record
    a stop it has only requested; the terminal observation arrives via :func:`describe_many`.
    """
    try:
        _client(launch).stop_task(cluster=launch.cluster, task=task_arn, reason=reason)
    except Exception as exc:                                   # noqa: BLE001 - re-raised with context
        raise CloudLaunchError(f"could not stop task {task_arn}: {exc}") from exc
