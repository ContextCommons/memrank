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
"""Provision a target as a disposable compose project on this machine.

Renders the compose from the manifest rather than reusing the engine repo's own file. That file
builds from source and bind-mounts the developer's working tree, which is right for a dev loop and
wrong for a reproducible run: it would evaluate uncommitted engine code under a target name that
means something else. ``--on local`` therefore runs the *published* artifact, the same one the cloud
runs, so a local row and a cloud row can be compared on more than a name.

Host ports are never chosen. Publishing target-only (``127.0.0.1::8000``) makes the daemon bind port
0, which is atomic, and the assigned port is read back with ``docker compose port``. Picking a port --
fixed or base-offset -- is the documented anti-pattern and is racy between the check and the bind;
``scripts/local-eval.sh`` did exactly that and every disposable stack collided. Because each run gets
its own project name and its own kernel-assigned ports, K runs coexist with no coordination.
"""

from __future__ import annotations

import json
import subprocess
import time
from types import TracebackType
from typing import Any

from memrank.placement.base import Endpoint, PlacementError, Requirement, require
from memrank.placement.graph import load_graph, pin, service_name
from memrank.targets.engine_env import (
    ENGINE_SETTINGS,
    component_env,
    declared_env,
    engine_command,
    engine_secret_vars,
    engine_token_env,
    harness_env,
    readiness_budget_s,
    readiness_path,
)
from memrank.targets.manifest import Manifest

# Readiness paths are per engine (memrank/targets/engine_env.py READINESS): mem0 answers
# /configure, hindsight and atomicmemory /health, supermemory /. Assuming mem0's for all four is
# what left a hindsight cloud task stuck waiting on a health check it could never pass.
_READY_INTERVAL_S = 2.0


def _engine_environment(target: Manifest) -> dict[str, str]:
    """Topology for this placement, plus the components every placement must state identically.

    Components come from :mod:`memrank.targets.engine_env`, never restated here: this function used
    to hardcode ``MEM0_*`` names while reading as though it were generic, so rendering a compose
    project for hindsight would have configured it with mem0's variables and silently left the
    declared LLM unset.
    """
    return (ENGINE_SETTINGS.get(target.adapter, {}) | {"PORT": str(target.engine.port)}
            | declared_env(target)
            | engine_token_env(target) | component_env(target))


def engine_secrets(target: Manifest) -> dict[str, str]:
    """The credentials the engine process needs, resolved through memrank's one chokepoint.

    Which credentials those are comes from :func:`memrank.targets.engine_env.engine_secret_vars`,
    shared with the cloud renderer; this function only resolves the values, which never leave the
    local machine.

    Values are injected even when the same name is already in memrank's own environment. That is
    true of a process which inherits its environment and false of a **container**, which inherits
    nothing.

    mem0's TEI path also needs an ``OPENAI_API_KEY``, because it reaches the sidecar through the
    OpenAI SDK and that client refuses to construct without one. It is NOT resolved here: TEI never
    checks the value, so it is a constant the image needs rather than a credential anyone owns, and
    it is declared in the graph beside the container that needs it, not resolved as a secret.
    No target shipped here runs a TEI sidecar since the matched mem0 variants moved to the research
    lane (2026-08-19); `sidecar_stack` in tests/placement/conftest.py is what keeps this covered.
    """
    from memrank import config
    from memrank.secrets import requirements

    providers = requirements.keyed_providers(target.components)
    # Preflight the DERIVED set only, which is what it has always covered: it raises on a missing
    # credential rather than launching an engine that dies on one. A declared credential is checked
    # by `catalog.secret_status` at submit time, where its name is known.
    config.preflight(target.adapter, embedder=providers.get("embedder"), llm=providers.get("llm"))
    # Names come from the one table both placements read, so a credential the cloud taskdef carries
    # is a credential the compose project carries.
    secrets: dict[str, str] = {}
    for name, variables in engine_secret_vars(target).items():
        value = config.secret(name)
        if value is None:
            continue
        for variable in variables:
            secrets[variable] = value
    return secrets


def render_compose(target: Manifest, *, project: str) -> dict[str, Any]:
    """The compose project for one disposable run of ``target``.

    Loaded from the target's declared graph rather than assembled here. What this function still
    supplies is everything that cannot be written down ahead of time: the per-run project name,
    the environment derived from the manifest's components, and the start command the image needs.

    Args:
        target: The resolved manifest.
        project: A per-run compose project name; this is what isolates one run from another.

    Returns:
        A compose document, ready to serialise.

    Raises:
        ValueError: If the target is in-process, or its graph cannot be resolved.
    """
    if target.kind == "in-process":
        raise ValueError(f"{target.name!r} is in-process and needs no compose project")
    document = load_graph(target)
    engine = document["services"][service_name(target)]
    # Merged rather than declared: these come from the manifest, which is the one place that says
    # what a target is made of. See graph_variables.
    engine["environment"] = {**(engine.get("environment") or {}), **_engine_environment(target)}
    # From engine_env.ENGINE_COMMAND -- a property of the IMAGE (mem0's published build starts
    # uvicorn without migrating), so it belongs with the engine table rather than in each graph.
    command = engine_command(target)
    if command is not None:
        engine["command"] = command
    return {"name": project, **document}


class LocalPlacement:
    """A disposable compose project per run, on this machine.

    Always used as a context manager, so teardown survives an exception. The shell version relied on
    an ``EXIT`` trap that read a ``local`` from ``main()`` and, under ``set -u``, died with
    "project: unbound variable" -- losing the containers *and* the real error.
    """

    def __init__(self, *, run_id: str) -> None:
        # Compose project names must be lowercase; the run id already is.
        self.project = f"memrank-{run_id}".lower()
        self._document: str | None = None
        self._up = False

    def _run(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        assert self._document is not None, "provision() must render the document first"
        return subprocess.run(
            ["docker", "compose", "-p", self.project, "-f", "-", *args],
            input=self._document, text=True, check=check, capture_output=True)

    def _discover_url(self, service: str, port: int) -> str:
        """Read back the port the KERNEL assigned. Never assume one."""
        result = self._run("port", service, str(port))
        published = result.stdout.strip().splitlines()
        if not published:
            raise PlacementError(
                f"compose published no host port for {service}:{port} in {self.project}")
        host, _, assigned = published[0].rpartition(":")
        # Use the host compose reports, not "localhost": with a remote or rootless daemon they differ.
        return f"http://{host or '127.0.0.1'}:{assigned}"

    def _wait_ready(self, target: Manifest, base_url: str) -> None:
        """Poll the engine's own readiness path until it answers.

        How long to allow is per engine (``READINESS[...].start_period``) for the same reason the
        path is: how long a build takes to become ready is a property of the build. A flat constant
        happened to be safe only because it was longer than all four.
        """
        import httpx

        path = readiness_path(target)
        budget = float(readiness_budget_s(target))
        deadline = time.monotonic() + budget
        last: Exception | None = None
        # Probe first, then decide whether to wait again: a `while deadline` loop skips the engine
        # entirely when the budget is small, and sleeps once more after the attempt that ran out --
        # so the error it raises names no cause at all.
        while True:
            try:
                if httpx.get(f"{base_url}{path}", timeout=5.0).status_code < 500:
                    return
            except Exception as exc:                      # noqa: BLE001 - retried below
                last = exc
            if time.monotonic() + _READY_INTERVAL_S >= deadline:
                break
            time.sleep(_READY_INTERVAL_S)
        raise PlacementError(
            f"{self.project} did not become ready at {base_url}{path} within "
            f"{budget:.0f}s; last error: {last}")

    def daemon_platform(self) -> str:
        """The ``os/arch`` this machine's Docker daemon runs containers as.

        Asked of the daemon rather than of Python: on an arm64 Mac with Rosetta the interpreter and
        the daemon can disagree, and what a receipt must record is the platform the CONTAINER ran
        as. Docker reports `aarch64`, which is the same thing an image index calls `arm64`.
        """
        result = subprocess.run(
            ["docker", "system", "info", "--format", "{{.OSType}}/{{.Architecture}}"],
            text=True, check=False, capture_output=True)
        reported = result.stdout.strip()
        if not reported:
            raise PlacementError(
                f"could not ask Docker which platform it runs containers as: "
                f"{result.stderr.strip() or 'no output'}")
        return {"aarch64": "linux/arm64", "x86_64": "linux/amd64"}.get(
            reported.rpartition("/")[2], reported)

    def check(self) -> list[Requirement]:
        """What running here needs: a Docker CLI, and a daemon that answers. Never raises.

        Asked of the daemon rather than of the filesystem, because a Docker Desktop that is
        installed and not started is the common case and looks identical to a working one from
        ``which docker``.
        """
        try:
            result = subprocess.run(["docker", "version", "--format", "{{.Server.Version}}"],
                                    text=True, check=False, capture_output=True)
        except (OSError, ValueError) as exc:
            return [Requirement(name="docker", ok=False, detail=str(exc),
                                fix="install Docker (https://docs.docker.com/get-docker/), "
                                    "or run elsewhere with `--on cloud`")]
        if result.returncode != 0 or not result.stdout.strip():
            detail = (result.stderr or result.stdout).strip().splitlines()
            return [Requirement(name="docker", ok=False,
                                detail=detail[0] if detail else "the daemon did not answer",
                                fix="start Docker and re-run, or run elsewhere with `--on cloud`")]
        return [Requirement(name="docker", ok=True, detail=result.stdout.strip())]

    def provision(self, target: Manifest) -> Endpoint:
        """Bring the target up and return where it is actually reachable."""
        # The same rows the caller was shown before any run was minted: enforcement and diagnosis
        # are one code path, so they cannot disagree about whether this machine can run here.
        require(self.check())
        document = render_compose(target, project=self.project)
        # Resolved ONCE, here, and what runs is the digest. Compose would otherwise ask the
        # registry itself (`--pull always` below) and ECS would ask again in the cloud -- two
        # answers to "what does :latest mean", which is how one sweep can straddle a tag that moved
        # between its cells with nothing in either receipt to say so.
        document, resolutions = pin(document, default_platform=self.daemon_platform())
        # Merged HERE, not in render_compose: the rendered document is inert and safe to log or
        # diff, and credentials must never be part of a comparable artifact.
        service = service_name(target)
        document["services"][service]["environment"] |= engine_secrets(target)
        self._document = json.dumps(document)          # compose accepts JSON as YAML
        try:
            # `--pull always` is now belt AND braces: the references are already digests, so
            # there is nothing left to resolve, but a stale local layer for that digest is still
            # worth re-checking against the registry.
            self._run("up", "-d", "--wait", "--pull", "always")
            self._up = True
        except subprocess.CalledProcessError as exc:
            self.teardown()
            raise PlacementError(
                f"could not start {self.project}: {exc.stderr or exc.stdout}") from exc
        base_url = self._discover_url(service, target.engine.port)
        # Per engine, not mem0's assumed for all -- see engine_env.READINESS.
        self._wait_ready(target, base_url)
        # The FULL harness env, the same function the cloud renderer uses. Returning only the base
        # URL is what left a local run on its adapter's 60s default timeout and, worse, with no
        # image digest -- so engine_provenance could build no component purl and a local row was not
        # comparable with a cloud one, which is the property this whole design exists to guarantee.
        # From the RESOLUTION, not read back from the daemon: what was resolved is what was
        # pulled, and asking Docker afterwards was only ever an approximation of that. The tag
        # travels here too -- a pinned reference no longer contains it, and a receipt still wants
        # to say which pointer was followed.
        engine = resolutions[service]
        return Endpoint(base_url=base_url,
                        adapter_env=harness_env(target, url=base_url,
                                                engines_repo=engine.repository,
                                                tag=engine.tag,
                                                digest=engine.platform_digest,
                                                index_digest=engine.index_digest,
                                                platform=engine.platform),
                        image_digests={name: r.platform_digest
                                       for name, r in resolutions.items()},
                        details={"project": self.project})

    def teardown(self) -> None:
        """Remove containers, networks and volumes. Safe to call twice."""
        if self._document is None:
            return
        self._run("down", "-v", "--remove-orphans", check=False)
        self._up = False

    def __enter__(self) -> LocalPlacement:
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None,
                 tb: TracebackType | None) -> None:
        self.teardown()
