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
"""One evaluation launch, from target ref to a running ECS task.

Extracted from ``runner.py`` (the seam open-issues #19 earmarked) so that BOTH launch
surfaces -- the CLI's ``--on cloud`` path and the accounts API's ``POST /orgs/{org}/runs``
-- go through one implementation: resolve the target, render the task definition from its
manifest, register it, and run it. Two launchers is how a forwarded flag or a sizing rule
gets fixed in one and silently not the other.

This module renders and launches; it does NOT verify images (the caller's preflight owns
that -- the CLI checks the contract against its git checkout, the API checks existence
against ECR), does NOT create local run directories, and does NOT write status files.
Those are surface concerns; this is the shared core.

The ``ctx`` dict everywhere here is the AWS-context shape validated by
``memrank.config._AWS_CONTEXT_FIELDS`` -- from ``.aws-context.json`` on an operator
machine, or from ``MEMRANK_BENCH_CONTEXT`` in the API's task environment.
"""
from __future__ import annotations

from dataclasses import dataclass


def resolve_target_or_inprocess(ref: str):
    """The manifest a placement provisions, or a stand-in for a legacy adapter name.

    Resolve-only -- never constructs an adapter. The legacy form takes bare adapter names,
    which mostly have manifests; anything without one cannot be provisioned, so it gets an
    in-process stand-in and ``--on none`` behaviour.
    """
    from memrank.targets import TargetNotFound, resolve_target
    from memrank.targets.manifest import Manifest

    try:
        return resolve_target(ref)
    except TargetNotFound:
        return Manifest(name=ref, kind="in-process", adapter=ref)


def render_for_launch(ref: str, ctx: dict, *, command: str, image_tag: str,
                      secret_arns: dict[str, str] | None = None,
                      harness_env_extra: dict[str, str] | None = None) -> tuple[dict, str, str]:
    """Render the ECS task definition for one target, from its manifest.

    A thin wrapper over :func:`memrank.placement.cloud.render_from_context`, which this,
    the CLI, and ``memrank targets render`` all use -- so what you inspect is what launches.

    Returns:
        ``(taskdef, engine_tag, engine_digest)`` -- the tag/digest ride along because the
        caller records them (receipt CLI-side, runs row API-side).

    Raises:
        RegistryError: If an image cannot be resolved to a digest. Deliberately fatal, and before
            any task is registered: a run that cannot name what it will execute must not produce a
            receipt claiming it did. This replaced a best-effort lookup that returned "" on any
            failure, which meant an unreproducible run left only an empty field nobody reads.
    """
    from memrank.placement.cloud import render_from_context

    target = resolve_target_or_inprocess(ref)
    if target.kind == "in-process":
        # A control arm pulls no engine image, so there is nothing to resolve and nothing to pin.
        return (render_from_context(target, ctx, command=command, image_tag=image_tag,
                                    secret_arns=secret_arns,
                                    harness_env_extra=harness_env_extra), "", "")
    resolutions = _resolved_graph(target)
    engine = resolutions[target.service]
    taskdef = render_from_context(target, ctx, command=command, image_tag=image_tag,
                                  resolutions=resolutions, secret_arns=secret_arns,
                                  harness_env_extra=harness_env_extra)
    return taskdef, engine.tag, engine.platform_digest


def _resolved_graph(target) -> dict:
    """Resolve every image the target runs, once, for the platform the TASK will run as.

    The platform matters and is not the resolver's guess: ``ecs_compile.runtime_platform`` is
    already the authority on what an ECS task runs as, derived from the graph's own ``platform:``
    declarations. Selecting a manifest for any other platform would pin a digest the task cannot
    execute -- and Fargate does not emulate.
    """
    from memrank.placement.ecs_compile import runtime_platform
    from memrank.placement.graph import load_graph, pin

    document = load_graph(target)
    architecture = runtime_platform(document)["cpuArchitecture"]
    platform = "linux/arm64" if architecture == "ARM64" else "linux/amd64"
    return pin(document, default_platform=platform)[1]


@dataclass(frozen=True)
class LaunchResult:
    """What one launch produced -- everything a record of the run needs to name."""

    run_id: str
    taskdef_arn: str
    task_arn: str
    engine_tag: str
    engine_digest: str
    artifact_prefix: str


def launch(ctx: dict, ref: str, argv: list[str], *, run_id: str, image_tag: str,
           secret_arns: dict[str, str] | None = None,
           harness_env_extra: dict[str, str] | None = None) -> LaunchResult:
    """Launch one evaluation as an ECS task: command -> render -> register -> run.

    A fresh task definition per run, because ``run-task`` overrides cannot change the
    image. Raises :class:`memrank.placement.cloud_submit.CloudLaunchError` on AWS refusal.

    Args:
        secret_arns: Which credentials the task spends, as ``{canonical name: ARN}`` -- the
            submitting org's own keys. Falls back to the deployment-wide set in the context
            when absent, which is the operator path and the CLI-side render.
        harness_env_extra: Environment for the harness container alone -- the progress endpoint
            and this run's token. Absent on the operator path, which has no API to report to;
            the task then falls back to writing only ``progress.json`` to S3, exactly as before
            this channel existed.
    """
    from memrank.placement import cloud_submit

    launch_target = cloud_submit.LaunchTarget(
        region=ctx["region"], cluster=ctx["cluster"], subnet=ctx["subnet"],
        security_group=ctx["security_group"])
    command = cloud_submit.remote_command(argv, bucket=ctx["artifact_bucket"],
                                          run_id=run_id)
    taskdef, engine_tag, engine_digest = render_for_launch(
        ref, ctx, command=command, image_tag=image_tag, secret_arns=secret_arns,
        harness_env_extra=harness_env_extra)
    taskdef_arn = cloud_submit.register(taskdef, launch_target)
    task_arn = cloud_submit.submit(taskdef_arn, launch_target)
    return LaunchResult(run_id=run_id, taskdef_arn=taskdef_arn, task_arn=task_arn,
                        engine_tag=engine_tag, engine_digest=engine_digest,
                        artifact_prefix=f"cloud-runs/{run_id}")
