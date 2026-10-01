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
"""Compile a target's declared Compose graph into ECS container definitions.

A translation, not a second description. Everything here has a published meaning on both sides --
Compose's service key becomes a container name, its `healthcheck` becomes `healthCheck`, its
`depends_on` condition becomes a `dependsOn` condition -- so the cloud runs the graph the author
wrote rather than one the renderer inferred from a `depends` list. That inference is what let the
two placements drift: cloud pulled a mirrored embedder image at an ambient tag while local pulled
the vendor's, under one target name, with no receipt field recording which.

Two things do NOT translate, and are handled rather than ignored:

* **Topology.** Compose gives each service its own DNS name; under `awsvpc` every container in a
  task shares one network namespace, so a neighbour is reached on `localhost`. This is the only
  axis on which a rendered cloud environment may legitimately differ from the local one.
* **Anything ECS cannot express faithfully** -- mounted volumes, `build`, published host ports,
  services that disagree about their platform. Those are REFUSED by name. A renderer that quietly
  drops what it does not understand produces a task definition that launches and measures
  something other than what was declared.
"""

from __future__ import annotations

import re
from typing import Any

from memrank.placement.base import CloudRenderError

#: Compose's `depends_on` conditions, and the ECS conditions that mean the same thing. Compose's
#: `service_completed_successfully` has an ECS counterpart (`SUCCESS`) but no graph uses it, so it
#: is refused rather than translated untested.
CONDITIONS: dict[str, str] = {"service_healthy": "HEALTHY", "service_started": "START"}

#: Compose keys with no faithful ECS equivalent under Fargate. Each is refused by name.
#:
#: `volumes` and `build` are the substantive ones -- a bind mount has no meaning in a task that
#: pulls its filesystem from a registry, and Fargate builds nothing. They are also exactly what a
#: developer reaches for first when adapting a dev compose file, which is why the refusal names the
#: key instead of producing a task that starts and then behaves differently.
UNSUPPORTED: tuple[str, ...] = ("volumes", "build", "network_mode", "extends")

#: Compose keys a graph declares for the local placement that Fargate has no way to honour, so
#: the compiler leaves them out rather than refusing a graph every cloud run uses.
#:
#: `shm_size` sizes /dev/shm for the engines that run Postgres (hindsight.compose.yaml says why).
#: ECS's counterpart, `linuxParameters.sharedMemorySize`, is not supported on Fargate, so a cloud
#: task keeps Fargate's own /dev/shm. Refusing the key would stop every hindsight, atomicmemory and
#: mem0 cloud run for a limit that bites only at scale; when it does, the engine's own error names
#: it.
LOCAL_ONLY: tuple[str, ...] = ("shm_size",)

#: Under awsvpc a task is one network namespace, so every container addresses its neighbours here.
LOCALHOST = "localhost"

#: ECS's documented health-check ranges, as ``{ecs field: (minimum, maximum)}``.
#:
#: Compose bounds none of these, so a graph that runs locally can still be rejected by
#: RegisterTaskDefinition -- which is what happened: `retries: 20` on a datastore came back as a 502
#: naming a container the author never wrote and no file it appears in. They are checked HERE
#: because botocore's own model carries types and no ranges, so validating the rendered document
#: against it cannot see them; only AWS can, and only after a round trip.
HEALTH_CHECK_LIMITS: dict[str, tuple[int, int]] = {
    "interval": (5, 300), "timeout": (2, 60), "retries": (1, 10), "startPeriod": (0, 300)}

_DURATION = re.compile(r"^(\d+)(ms|s|m|h)?$")
_UNIT_SECONDS = {"ms": 0.001, "s": 1, "m": 60, "h": 3600, None: 1}


def duration_seconds(value: str | int, *, service: str, key: str) -> int:
    """Compose durations (``5s``, ``30s``) as the whole seconds ECS wants.

    Args:
        value: The declared duration.
        service: The service it came from, for the refusal message.
        key: The field it came from, for the refusal message.

    Returns:
        Seconds, rounded down.

    Raises:
        CloudRenderError: If the duration is not one this compiler can convert. Sub-second values
            round to zero, which ECS rejects with a message naming neither the file nor the field.
    """
    match = _DURATION.match(str(value).strip())
    if not match:
        raise CloudRenderError(
            f"service {service!r} declares {key}={value!r}, which is not a duration this compiler "
            f"can convert to ECS seconds (expected e.g. '5s', '30s', '1m')")
    seconds = int(match.group(1)) * _UNIT_SECONDS[match.group(2)]
    if seconds < 1:
        raise CloudRenderError(
            f"service {service!r} declares {key}={value!r}, which is under one second; ECS "
            f"expresses health-check timings in whole seconds")
    return int(seconds)


def localhost_environment(environment: dict[str, str], services: list[str],
                          own: str = "") -> dict[str, str]:
    """Rewrite neighbour service names to ``localhost``, where they are actually hosts.

    `POSTGRES_HOST: datastore` becomes `localhost`, and `http://embedder/v1` becomes
    `http://localhost/v1`. Driven by the graph's own service names, so a target that adds a
    container is translated without anyone remembering to extend a table -- which is precisely how
    the retired ``_topology_environment`` went stale, keyed as it was on a hardcoded ``depends``
    list rather than on the services that actually exist.

    Only two positions count as a host: the whole value (optionally with a port), and the authority
    of a URL. A match anywhere else is left alone. Rewriting every occurrence turned ``MODE: engine``
    into ``MODE: localhost`` -- an engine whose configuration happened to contain a word matching a
    service name would have been silently misconfigured, with no error and nothing to fail.

    A bare whole-value match is only rewritten for a NEIGHBOUR. A service naming itself is far more
    likely to be saying what it is than where to reach itself, and the two cannot be told apart from
    the value. The residual ambiguity is real and deliberate: a neighbour's name as a whole value is
    treated as a host whatever the variable means, because a compiler with no schema cannot know
    better -- and the alternative, requiring a URL, would not rewrite ``POSTGRES_HOST: datastore``,
    which is the commonest form there is. A self-reference written as a URL is still rewritten,
    since that one is unambiguous.

    Args:
        environment: The service's declared environment.
        services: Every service name in the graph.
        own: This service's own name, excluded from bare whole-value matching.

    Returns:
        The environment with neighbour hosts replaced.
    """
    # Longest first, so a service named `db` cannot rewrite half of a name like `db-replica`.
    names = "|".join(re.escape(n) for n in sorted(services, key=len, reverse=True))
    neighbours = "|".join(re.escape(n) for n in sorted((s for s in services if s != own),
                                                       key=len, reverse=True))
    whole_value = re.compile(rf"^({neighbours})(:\d+)?$") if neighbours else None
    url_authority = re.compile(rf"(?<=//)(?:([^/@\s]*)@)?({names})(?=[:/?#]|$)")

    def rewrite(value: str) -> str:
        if whole_value is not None and whole_value.match(value):
            return whole_value.sub(rf"{LOCALHOST}\2", value)
        return url_authority.sub(
            lambda m: f"{m.group(1)}@{LOCALHOST}" if m.group(1) else LOCALHOST, value)

    return {key: rewrite(str(value)) for key, value in environment.items()}


def _within_limits(service: str, field: str, value: int) -> int:
    """``value``, or a refusal naming what ECS will accept.

    A refusal and not a clamp. Quietly reducing `retries: 60` to 10 would shorten how long a run
    waits for an embedder to load its model -- a semantic edit nobody asked for, made invisible.
    """
    low, high = HEALTH_CHECK_LIMITS[field]
    if not low <= value <= high:
        raise CloudRenderError(
            f"service {service!r} declares a health check {field} of {value}, outside the {low}-{high} "
            f"ECS accepts. Compose has no such limit, so this runs locally and is refused at "
            f"registration; keep the same wait by trading interval against retries.")
    return value


def _health_check(service: str, declared: dict[str, Any]) -> dict[str, Any]:
    """Compose's `healthcheck` as ECS's `healthCheck`, with durations converted and bounds checked."""
    check: dict[str, Any] = {"command": list(declared["test"])}
    for compose_key, ecs_key in (("interval", "interval"), ("timeout", "timeout"),
                                 ("start_period", "startPeriod")):
        if compose_key in declared:
            seconds = duration_seconds(declared[compose_key], service=service, key=compose_key)
            check[ecs_key] = _within_limits(service, ecs_key, seconds)
    if "retries" in declared:
        check["retries"] = _within_limits(service, "retries", int(declared["retries"]))
    return check


def _depends_on(service: str, declared: dict[str, Any] | list[str]) -> list[dict[str, str]]:
    """Compose's `depends_on` as ECS's `dependsOn`, refusing any condition not proven here.

    Both declared forms are accepted. The short form -- a list of names -- is what Compose calls
    `service_started`, and it used to reach ``.items()`` and raise an ``AttributeError`` from
    inside the compiler: neither a translation nor a refusal.
    """
    entries = declared if isinstance(declared, dict) else dict.fromkeys(declared, "service_started")
    out = []
    for name, spec in entries.items():
        condition = spec.get("condition") if isinstance(spec, dict) else spec
        if condition not in CONDITIONS:
            raise CloudRenderError(
                f"service {service!r} waits on {name!r} with condition {condition!r}, which this "
                f"compiler does not translate (known: {', '.join(sorted(CONDITIONS))})")
        out.append({"containerName": name, "condition": CONDITIONS[condition]})
    return out


def _refuse_unsupported(service: str, declared: dict[str, Any]) -> None:
    """Name what cannot be expressed, rather than dropping it and rendering something else.

    `ports` is deliberately absent from this check. Every graph publishes one so a developer can
    reach the engine from the host, and under awsvpc that is not wrong -- it is simply nothing,
    since the containers already share a namespace with the harness that calls them. Dropping it
    changes no behaviour, which is the test for what may be dropped silently.
    """
    for key in UNSUPPORTED:
        if declared.get(key):
            raise CloudRenderError(
                f"service {service!r} declares {key!r}, which an ECS task cannot express. Remove it "
                f"from the graph or give the target a cloud-specific one.")
    if not declared.get("image"):
        raise CloudRenderError(
            f"service {service!r} names no image, so there is nothing for ECS to pull. Every "
            f"service in a graph must name a complete reference; Fargate builds nothing.")


def compile_service(name: str, declared: dict[str, Any], *, services: list[str],
                    log_configuration: dict[str, Any]) -> dict[str, Any]:
    """One Compose service as one ECS container definition.

    Args:
        name: The service key, which becomes the container name.
        declared: That service's Compose block.
        services: Every service name in the graph, for the localhost rewrite.
        log_configuration: The awslogs block, whose stream prefix is the container's own name.

    Returns:
        A container definition. `essential` is False: only the harness container may end the task.

    Raises:
        CloudRenderError: If the service declares something ECS cannot express faithfully.
    """
    _refuse_unsupported(name, declared)
    container: dict[str, Any] = {
        "name": name,
        "image": declared["image"],
        "essential": False,
        "environment": localhost_environment(declared.get("environment") or {}, services, name),
        "logConfiguration": log_configuration,
    }
    if declared.get("command"):
        container["command"] = list(declared["command"])
    if declared.get("healthcheck"):
        container["healthCheck"] = _health_check(name, declared["healthcheck"])
    if declared.get("depends_on"):
        container["dependsOn"] = _depends_on(name, declared["depends_on"])
    return container


def runtime_platform(document: dict[str, Any]) -> dict[str, str]:
    """The one platform a task runs on, taken from the graph rather than assumed.

    An ECS task has a single ``runtimePlatform`` and Fargate does not emulate, so a graph whose
    services disagree cannot run as one task -- and must say so here rather than at launch, where
    the symptom is a container that never starts. Services that declare nothing inherit the task's
    platform, which is what Compose means by leaving it out.

    Args:
        document: The loaded Compose document.

    Returns:
        The ECS ``runtimePlatform`` block.

    Raises:
        CloudRenderError: If two services name different platforms.
    """
    declared = {name: service["platform"] for name, service in document["services"].items()
                if service.get("platform")}
    distinct = set(declared.values())
    if len(distinct) > 1:
        detail = ", ".join(f"{name}={value}" for name, value in sorted(declared.items()))
        raise CloudRenderError(
            f"this target's containers do not agree on a platform ({detail}). An ECS task has one "
            f"runtimePlatform and Fargate does not emulate, so they cannot run together.")
    platform = distinct.pop() if distinct else "linux/amd64"
    architectures = {"linux/amd64": "X86_64", "linux/arm64": "ARM64"}
    if platform not in architectures:
        raise CloudRenderError(
            f"platform {platform!r} has no Fargate equivalent "
            f"(known: {', '.join(sorted(architectures))})")
    return {"cpuArchitecture": architectures[platform], "operatingSystemFamily": "LINUX"}
