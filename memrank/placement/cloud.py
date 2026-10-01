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
"""Render an ECS task definition for one target, from that target's manifest.

This closes the defect the whole design exists to fix. ``deploy/ecs/taskdef.*.json.tpl`` were
hand-authored, one per **adapter**, with the components welded in as literals: the mem0 template
said ``BAAI/bge-small-en-v1.5`` / ``384`` whatever ref you asked for. So ``mem0:voyage`` could not be
expressed in the cloud at all -- it ran bge-small and the leaderboard row still said voyage. Two
placements, one name, two different systems measured.

Here the containers come from the target's own declared Compose graph, compiled by
:mod:`memrank.placement.ecs_compile`, and the components from
:func:`memrank.targets.engine_env.component_env` -- the same file and the same function the local
renderer reads. What this module still owns is everything ECS-specific that no graph states: the
harness container, Fargate sizing, log configuration, and secret ARNs.
``tests/placement/test_no_drift.py`` enumerates the catalog and proves both placements agree, for
every container of every target.

Cloud is not a :class:`~memrank.placement.base.Placement`. Locally memrank runs on your machine and
provisions an engine beside it, so there is an endpoint to hand back. Under ECS memrank runs *inside*
the task alongside the engine: there is nothing to connect to and nothing to tear down. This module
therefore renders only -- launching stays in ``scripts/cloud-run.sh``, which already owns the parts
proven in production (terraform outputs, ECR digest pinning, register, run-task, the poll loop).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Imported rather than restated: judge_client owns which credential the judge uses.
from memrank.judging.client import JUDGE_SECRET as _JUDGE_SECRET
from memrank.placement.base import CloudRenderError
from memrank.placement.ecs_compile import LOCALHOST, compile_service, runtime_platform
from memrank.placement.graph import (
    engine_image,
    load_graph,
    reference_parts,
    service_name,
)
from memrank.targets.engine_env import (
    ENGINE_SETTINGS,
    HEALTHCHECK_INTERVAL_S,
    HEALTHCHECK_RETRIES,
    component_env,
    declared_env,
    engine_command,
    engine_secret_vars,
    engine_token_env,
    harness_env,
    readiness_probe,
)
from memrank.targets.manifest import Manifest


@dataclass(frozen=True)
class AwsContext:
    """The infrastructure coordinates a task definition needs, all resolved by terraform.

    Grouped rather than passed one by one: ``scripts/cloud-run.sh`` reads every one of these from a
    ``terraform output``, so they travel together and mean nothing apart.
    """

    family: str
    region: str
    log_group: str
    execution_role_arn: str
    task_role_arn: str
    memrank_image: str
    command: str
    cpu: str = "4096"
    memory: str = "8192"
    # {canonical credential name: SSM/Secrets Manager ARN}. ARNs, never values -- the rendered
    # document has to stay inert enough to log, diff and attach to a review.
    secret_arns: dict[str, str] = field(default_factory=dict)
    # The engine image's digest, which only the launcher can resolve. Its repository and tag are
    # NOT here: those are read from the reference the target's graph declares, so nothing can
    # supply a provenance record for an image other than the one that will be pulled.
    engine_digest: str = ""
    # Extra env for the ENGINE container, for settings that are neither components nor topology
    # (atomicmemory's CORE_API_KEY). Deliberately narrow: anything a receipt should describe
    # belongs in the manifest, where the drift gate can see it.
    engine_env_extra: dict[str, str] = field(default_factory=dict)
    # Extra env for the HARNESS container: where to report progress, and the run token that says
    # whose progress it is (`memrank/api/run_progress.py`). Deliberately NOT part of `command`,
    # which is a shell string readable through `DescribeTaskDefinition` and `/proc/<pid>/cmdline`
    # -- the same exposure Buildkite re-execs over a pipe to avoid. Only this container gets it:
    # sidecars run vendor images and share the task's ENI, but not its environment.
    harness_env_extra: dict[str, str] = field(default_factory=dict)


def _log_configuration(aws: AwsContext, stream_prefix: str) -> dict[str, Any]:
    return {"logDriver": "awslogs",
            "options": {"awslogs-group": aws.log_group,
                        "awslogs-region": aws.region,
                        "awslogs-stream-prefix": stream_prefix}}


def _as_pairs(env: dict[str, str]) -> list[dict[str, str]]:
    """ECS wants ``[{"name": ..., "value": ...}]``. Sorted so a re-render diffs cleanly."""
    return [{"name": k, "value": v} for k, v in sorted(env.items())]


def _engine_secrets(target: Manifest, aws: AwsContext) -> list[dict[str, str]]:
    """The credentials the engine container needs, as ARNs under the names its process reads.

    Derived AND declared, from the shared table in :func:`memrank.targets.engine_env.
    engine_secret_vars`. This used to build its own list from the provider table, which silently
    dropped every credential a target declared for itself -- the exact case the declaration exists
    for.
    """
    out: list[dict[str, str]] = []
    for canonical, variables in sorted(engine_secret_vars(target).items()):
        arn = aws.secret_arns.get(canonical)
        if not arn:
            raise CloudRenderError(
                f"{target.name!r} needs {canonical} but no ARN was supplied for it. ECS resolves "
                f"secrets by ARN at container start; rendering without one would launch a task "
                f"that dies on a missing credential after paying for the pull.")
        out.extend({"name": variable, "valueFrom": arn} for variable in variables)
    return out


def _engine_health_check(target: Manifest) -> dict[str, Any]:
    """Probe the engine. Per adapter, because there is no generalisable answer.

    mem0 answers ``/configure``, hindsight and atomicmemory ``/health``, supermemory ``/``, and two
    of the images have no ``curl``. Assuming mem0's endpoint for all four left hindsight's task
    stuck in PENDING -- the engine ran fine, its health check probed a path it does not serve, and
    the harness waited on ``dependsOn: HEALTHY`` until the task was killed.

    The embedder used to be probed from here too, spliced into this command, because the sidecar
    could not state its own readiness. It states it now (``healthcheck`` in the graph) and the
    engine waits through a real ``dependsOn``, so a slow model load delays the engine rather than
    failing the engine's own probe.
    """
    probe, start_period = readiness_probe(target)
    return {"command": ["CMD-SHELL", probe], "timeout": 10,
            "interval": HEALTHCHECK_INTERVAL_S, "retries": HEALTHCHECK_RETRIES,
            "startPeriod": start_period}


def _engine_extras(target: Manifest, aws: AwsContext, compiled: dict[str, Any]) -> dict[str, Any]:
    """What the engine container needs beyond what its graph states.

    The graph says which image runs, what it waits for and how it addresses its neighbours. It
    deliberately does NOT say which model is under test -- that comes from the manifest's
    ``components:`` block, the one place a receipt and a leaderboard row both read.

    Args:
        target: The resolved manifest.
        aws: Infrastructure coordinates.
        compiled: The engine's container definition as translated from the graph.

    Returns:
        The same definition, with environment, command and secrets merged in.
    """
    # component_env last so a launcher extra cannot override a declared component; PORT is the
    # image's own listening port, which is a manifest fact and appears in no graph.
    compiled["environment"] = _as_pairs(
        ENGINE_SETTINGS.get(target.adapter, {})
        | {"PORT": str(target.engine.port)}
        | compiled["environment"]
        | declared_env(target)
        | engine_token_env(target)
        | aws.engine_env_extra
        | component_env(target))
    compiled["healthCheck"] = _engine_health_check(target)
    # From engine_env.ENGINE_COMMAND, so local and cloud cannot disagree about how an engine starts.
    command = engine_command(target)
    if command is not None:
        compiled["command"] = command
    secrets = _engine_secrets(target, aws)
    if secrets:
        compiled["secrets"] = secrets
    return compiled


def _memrank_container(target: Manifest, aws: AwsContext,
                       resolutions: dict[str, Any] | None = None) -> dict[str, Any]:
    """The harness itself. Unlike local placement it runs INSIDE the task, beside the engine.

    In-process targets have no engine container, so no base URL and nothing to wait for -- this is
    the whole of their task definition.
    """
    container: dict[str, Any] = {
        "name": "memrank",
        "image": aws.memrank_image,
        "essential": True,
        "entryPoint": ["/bin/sh", "-c"],
        "command": [aws.command],
        # In-process arms carry only the harness extras; a stack target's topology is merged in
        # below. Never empty now: the progress channel applies to every placement.
        "environment": _as_pairs(aws.harness_env_extra),
        "logConfiguration": _log_configuration(aws, "memrank"),
    }
    # The judge is the HARNESS calling Anthropic, not the engine, so its key belongs on this
    # container. It was missing entirely until now: `--on cloud --judge` would have died inside
    # judge_client, and nothing showed it locally because a `.env` file answered there. Attached
    # unconditionally rather than only when --judge is set, because the flag lives in the command
    # string and the task definition is rendered before it is parsed; an unused secret costs one
    # SSM read at container start.
    judge_arn = aws.secret_arns.get(_JUDGE_SECRET)
    if judge_arn:
        container["secrets"] = [{"name": _JUDGE_SECRET, "valueFrom": judge_arn}]
    if target.kind != "in-process":
        url = f"http://{LOCALHOST}:{target.engine.port}"
        # Provenance from the RESOLUTION when there is one -- both digests and the platform the
        # task will run as -- and from the declared reference otherwise, which is the inspection
        # path. Identical to what LocalPlacement.provision records, which is what makes two rows
        # comparable.
        service = service_name(target)
        engine = (resolutions or {}).get(service)
        repository, tag = reference_parts(engine_image(target))
        container["environment"] = _as_pairs({
            **harness_env(
                target, url=url,
                engines_repo=engine.repository if engine else repository,
                tag=engine.tag if engine else tag,
                digest=engine.platform_digest if engine else aws.engine_digest,
                index_digest=engine.index_digest if engine else "",
                platform=engine.platform if engine else ""),
            # Last so the launch path's own settings cannot be shadowed by a manifest that
            # happens to declare the same name -- the topology is the target's to describe, the
            # progress channel is not.
            **aws.harness_env_extra})
        container["dependsOn"] = [{"containerName": service, "condition": "HEALTHY"}]
    return container


def _task(aws: AwsContext, containers: list[dict[str, Any]],
          platform: dict[str, str]) -> dict[str, Any]:
    """The Fargate task wrapper every rendered definition shares."""
    return {
        "family": aws.family,
        "networkMode": "awsvpc",
        "requiresCompatibilities": ["FARGATE"],
        "cpu": aws.cpu,
        "memory": aws.memory,
        "runtimePlatform": platform,
        "executionRoleArn": aws.execution_role_arn,
        "taskRoleArn": aws.task_role_arn,
        "containerDefinitions": containers,
    }


_IN_PROCESS_PLATFORM = {"cpuArchitecture": "X86_64", "operatingSystemFamily": "LINUX"}


def render_taskdef(target: Manifest, *, aws: AwsContext,
                   resolutions: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build the ECS task definition for one run of ``target``.

    Every container comes from the target's declared Compose graph, compiled by
    :mod:`memrank.placement.ecs_compile`, plus the harness container this module adds -- the harness
    is not part of the system under test and so is not part of its graph.

    Args:
        target: The resolved manifest -- the sole source of every component value.
        aws: Infrastructure coordinates, all from terraform outputs.
        resolutions: ``{service: Resolution}`` when the caller has already resolved every tag to a
            digest, so the task runs identities rather than pointers. Absent for inspection
            (`targets render`), which shows the graph as declared and reaches no registry.

    Returns:
        A task definition ready for ``aws ecs register-task-definition --cli-input-json``.

    Raises:
        CloudRenderError: If the graph declares something ECS cannot express, or the target needs a
            credential with no ARN supplied.
        ValueError: If the target declares no graph, or names a service the graph does not define.
    """
    if target.kind == "in-process":
        # The control arms and baseline run inside the harness: one container, no sidecars. This
        # is what deploy/ecs/taskdef.json.tpl encoded before the renderer replaced it.
        return _task(aws, [_memrank_container(target, aws)], _IN_PROCESS_PLATFORM)

    document = load_graph(target)
    if resolutions:
        document = {**document, "services": {
            name: {**service, "image": resolutions[name].pinned}
            for name, service in document["services"].items()}}
    services = list(document["services"])
    containers = [_memrank_container(target, aws, resolutions)]
    for name, declared in document["services"].items():
        compiled = compile_service(name, declared, services=services,
                                   log_configuration=_log_configuration(aws, name))
        containers.append(_engine_extras(target, aws, compiled) if name == target.service
                          else {**compiled, "environment": _as_pairs(compiled["environment"])})
    return _task(aws, containers, runtime_platform(document))


# Fargate sizing. A stack target runs engine + datastore + embedder sidecars in one task; an
# in-process arm is only the harness.
SIZE_STACK = ("4096", "8192")
SIZE_IN_PROCESS = ("1024", "2048")


def render_from_context(target: Manifest, ctx: dict[str, Any], *, command: str, image_tag: str,
                        engine_digest: str = "", resolutions: dict[str, Any] | None = None,
                        secret_arns: dict[str, str] | None = None,
                        harness_env_extra: dict[str, str] | None = None) -> dict[str, Any]:
    """Render a task definition from the AWS context file that ``scripts/internal/aws-context.sh`` writes.

    The single mapping from that file's shape to :class:`AwsContext`. It exists because there were
    two: ``memrank submit --on cloud`` built one and ``memrank targets render`` built another from the
    raw JSON keys -- so `--aws .aws-context.json` never actually worked, and anything inspected with
    it was not necessarily what would launch. Rendering and launching must go through one function
    or inspection means nothing.

    Args:
        target: The resolved manifest.
        ctx: Parsed ``.aws-context.json``.
        command: The shell command the harness container runs.
        image_tag: Runner image tag.
        engine_digest: Kept for the inspection path, which resolves nothing.
        resolutions: Every image resolved to a digest, when a launcher did the resolving.
        secret_arns: The credentials this run should spend, as ``{canonical name: ARN}``.
            An ORG's own keys when the caller resolved them; the context's deployment-wide set
            otherwise, which is what a CLI-side render and the operator path still use. ARNs
            either way -- no secret value passes through here.
        harness_env_extra: Environment for the harness container only -- the progress endpoint and
            this run's token. Absent on the inspection path (``memrank targets render``), which
            mints nothing and must not print a credential.
    """
    in_process = target.kind == "in-process"
    cpu, memory = SIZE_IN_PROCESS if in_process else SIZE_STACK
    aws = AwsContext(
        family="memrank-bench" if in_process else f"memrank-bench-{target.adapter}",
        region=ctx["region"], log_group=ctx["log_group"],
        execution_role_arn=ctx["execution_role_arn"], task_role_arn=ctx["task_role_arn"],
        memrank_image=f"{ctx['runner_repository']}:{image_tag}",
        command=command, cpu=cpu, memory=memory,
        secret_arns=ctx["secret_arns"] if secret_arns is None else secret_arns,
        engine_digest=engine_digest, harness_env_extra=harness_env_extra or {})
    return render_taskdef(target, aws=aws, resolutions=resolutions)
