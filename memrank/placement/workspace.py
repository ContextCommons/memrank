# Copyright 2026 AtomicStrata
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Run an engine directly from a developer's source checkout."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shlex
import shutil
import signal
import socket
import subprocess
import time
from pathlib import Path
from types import TracebackType
from typing import TextIO

from memrank.placement.base import Endpoint, PlacementError, Requirement, require
from memrank.placement.local import engine_secrets
from memrank.targets.engine_env import (
    ENGINE_SETTINGS,
    component_env,
    declared_env,
    engine_token_env,
    harness_env,
    readiness_budget_s,
)
from memrank.targets.manifest import Manifest

_READY_INTERVAL_S = 2.0
_STOP_TIMEOUT_S = 10.0


def _git(source: Path, *args: str, text: bool = True) -> subprocess.CompletedProcess:
    """Run a bounded, read-only git query in the source checkout."""
    return subprocess.run(
        ["git", *args], cwd=source, capture_output=True, check=False,
        text=text, timeout=10)


def _source_facts(source: Path) -> tuple[str | None, bool, str]:
    """Return HEAD, dirty state, and an opaque hash of the working-tree delta.

    HEAD is ``None`` when ``source`` is not in a git repository. Version control is a FACT about
    the directory, not a precondition for running from it: requiring `git init` on a scratch folder
    buys nothing, since a single-commit throwaway SHA identifies no more than a content hash would,
    while blocking the case this placement exists to serve -- evaluating something handed to you
    five minutes ago. What the absence costs is recorded (see engine_provenance), and the run is
    already `development_observation` / `publishable: false` either way.

    Note git answers from a SUBDIRECTORY too, reporting the containing repository. A launcher in
    ``myengine/bench`` therefore records the engine repo's HEAD, which is the right answer for that
    layout and needs no special case here.
    """
    head = _git(source, "rev-parse", "HEAD")
    if head.returncode != 0 or not head.stdout.strip():
        return None, False, ""
    status = _git(source, "status", "--porcelain=v1", "--untracked-files=all")
    if status.returncode != 0:
        raise PlacementError(f"could not inspect source checkout {source}: {status.stderr.strip()}")
    digest = hashlib.sha256()
    diff = _git(source, "diff", "--binary", "HEAD", "--", text=False)
    if diff.returncode != 0:
        raise PlacementError(f"could not hash source changes in {source}")
    digest.update(diff.stdout)
    untracked = _git(source, "ls-files", "--others", "--exclude-standard", "-z", text=False)
    if untracked.returncode != 0:
        raise PlacementError(f"could not list untracked source files in {source}")
    for raw in sorted(path for path in untracked.stdout.split(b"\0") if path):
        digest.update(raw + b"\0")
        digest.update((source / os.fsdecode(raw)).read_bytes())
    return head.stdout.strip(), bool(status.stdout), digest.hexdigest()


def _command(target: Manifest) -> list[str]:
    """Expand the reviewed launcher into direct-exec argv; no implicit shell."""
    assert target.launch is not None
    try:
        argv = shlex.split(target.launch.command.format(port=target.engine.port))
    except (KeyError, ValueError) as exc:
        raise PlacementError(f"invalid workspace command for {target.name!r}: {exc}") from exc
    if not argv:
        raise PlacementError(f"workspace command for {target.name!r} produced no arguments")
    return argv


def _launcher_hashes(target: Manifest) -> tuple[str, str]:
    """Hash the reviewed launcher profile and the exact expanded argv separately."""
    assert target.launch is not None
    profile = {"command": target.launch.command,
               "requires": list(target.launch.requires), "port": target.engine.port}
    profile_blob = json.dumps(profile, sort_keys=True, separators=(",", ":")).encode()
    argv_blob = json.dumps(_command(target), separators=(",", ":")).encode()
    return hashlib.sha256(profile_blob).hexdigest(), hashlib.sha256(argv_blob).hexdigest()


class WorkspacePlacement:
    """A fresh native engine process tied to one Memrank run."""

    def __init__(self, *, target: Manifest, log_path: Path | None = None) -> None:
        self.target = target
        if target.binding is None:
            raise PlacementError(f"target {target.name!r} has no source binding")
        # None only for an unlinked `rootFrom: link` target. Not raised here: construction happens
        # before `check()`, and a missing link is a thing to REPORT with its remedy, not a crash.
        self.source = Path(target.binding.root).resolve() if target.binding.root else None
        self.log_path = log_path
        self._process: subprocess.Popen[str] | None = None
        self._log: TextIO | None = None

    def check(self) -> list[Requirement]:
        """Report source, manifest markers, git, and launcher availability."""
        if self.source is None:
            # The descriptor deliberately does not carry a path, so there is nothing to "fix" in it.
            # Name the command instead: this is the one failure whose remedy is a single line, and
            # the version that pointed at the author's home directory was unactionable for everyone
            # else who ever ran it.
            ref = self.target.binding.link if self.target.binding else self.target.name
            return [Requirement("checkout", False, f"{ref} is not linked on this machine",
                                f"memrank targets link {ref} <path to your checkout>")]
        rows = [Requirement("source", self.source.is_dir(), str(self.source),
                            f"fix binding.root in target {self.target.name!r}")]
        if not self.source.is_dir() or self.target.launch is None:
            return rows
        for marker in self.target.launch.requires:
            found = (self.source / marker).exists()
            rows.append(Requirement(marker, found, "present" if found else "missing",
                                    f"binding.root must point at the {self.target.name} checkout"))
        argv = _command(self.target)
        executable = argv[0]
        available = ((self.source / executable).is_file() if "/" in executable
                     else shutil.which(executable) is not None)
        rows.append(Requirement(executable, available, "available" if available else "not found",
                                f"install {executable} or fix launch.command"))
        # Reported, never required. git identifies what ran when it is there; when it is not, the
        # run still happens and the receipt says the directory was unversioned. A missing `git`
        # BINARY is the same situation as a directory with no repository -- nothing to read.
        if shutil.which("git") is not None:
            versioned = _git(self.source, "rev-parse", "HEAD").returncode == 0
            rows.append(Requirement("git", True,
                                    "versioned" if versioned else "unversioned directory", ""))
        else:
            rows.append(Requirement("git", True, "not installed; source will be unrecorded", ""))
        return rows

    def _ensure_port_free(self) -> None:
        """Refuse an occupied declared port instead of attaching to the wrong engine."""
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            probe.bind(("127.0.0.1", self.target.engine.port))
        except OSError as exc:
            raise PlacementError(
                f"port {self.target.engine.port} is already in use; stop the existing "
                f"{self.target.name} process and retry") from exc
        finally:
            probe.close()

    def _environment(self) -> dict[str, str]:
        """The same declared engine settings used by container placements.

        `declared_env` sits after the ambient environment so a target's own declaration wins over
        whatever happens to be exported -- that is the point of declaring it -- and before components
        and secrets, which it is forbidden to collide with anyway.
        """
        return (dict(os.environ) | ENGINE_SETTINGS.get(self.target.adapter, {})
                | declared_env(self.target)
                | engine_token_env(self.target) | component_env(self.target)
                | engine_secrets(self.target))

    def _start(self) -> None:
        """Start the reviewed command as its own process group."""
        destination = self.log_path or Path(os.devnull)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._log = destination.open("a", encoding="utf-8")
        try:
            self._process = subprocess.Popen(
                _command(self.target), cwd=self.source, env=self._environment(),
                stdout=self._log, stderr=subprocess.STDOUT, text=True,
                start_new_session=True)
        except (OSError, ValueError) as exc:
            assert self._log is not None
            self._log.close()
            self._log = None
            raise PlacementError(f"could not start {self.target.name} from {self.source}: {exc}") from exc

    def _wait_ready(self, base_url: str) -> None:
        """Wait for readiness while also detecting an early process exit."""
        import httpx

        deadline = time.monotonic() + float(readiness_budget_s(self.target))
        assert self.target.network is not None
        url = f"{base_url}{self.target.network.readiness.path}"
        last = "no response"
        while time.monotonic() < deadline:
            assert self._process is not None
            if self._process.poll() is not None:
                raise PlacementError(
                    f"{self.target.name} exited with code {self._process.returncode}; "
                    f"see {self.log_path or os.devnull}")
            try:
                if httpx.get(url, timeout=5.0).status_code < 500:
                    return
            except Exception as exc:  # noqa: BLE001 - readiness retries report the last cause
                last = str(exc)
            time.sleep(_READY_INTERVAL_S)
        raise PlacementError(f"{self.target.name} did not become ready at {url}: {last}")

    def provision(self, target: Manifest) -> Endpoint:
        """Launch the source checkout and return its local endpoint and provenance."""
        if target != self.target or target.launch is None or target.binding is None:
            raise PlacementError(f"target {target.name!r} has no matching source launcher")
        require(self.check())
        # `require` raised above if the checkout was unlinked, so the path is known from here on.
        assert self.source is not None
        self._ensure_port_free()
        sha, dirty, delta = _source_facts(self.source)
        profile_hash, command_hash = _launcher_hashes(target)
        self._start()
        base_url = f"http://127.0.0.1:{target.engine.port}"
        self._wait_ready(base_url)
        prefix = target.adapter.upper()
        env = harness_env(target, url=base_url, tag=sha or "",
                          platform=f"{platform.system().lower()}/{platform.machine().lower()}")
        env |= {f"{prefix}_ENGINE_EXECUTION_BINDING": "workspace",
                f"{prefix}_ENGINE_LAUNCHER_PROFILE_SHA256": profile_hash,
                f"{prefix}_ENGINE_COMMAND_SHA256": command_hash,
                "MEMRANK_EVIDENCE_CLASS": "development_observation"}
        # A commit, a dirty flag and a delta hash are one coherent statement about a versioned
        # directory. With no repository there is nothing to say, so nothing is said -- an empty SHA
        # would read as a value, and "clean" would be a claim about a tree nobody tracked.
        if sha is None:
            env[f"{prefix}_ENGINE_WORKSPACE_UNVERSIONED"] = "true"
        else:
            env |= {f"{prefix}_ENGINE_SOURCE_SHA": sha,
                    f"{prefix}_ENGINE_SOURCE_DIRTY": str(dirty).lower(),
                    f"{prefix}_ENGINE_SOURCE_DELTA_SHA256": delta}
        return Endpoint(base_url=base_url, adapter_env=env,
                        details={"binding": "workspace", "source_dirty": dirty,
                                 "versioned": sha is not None})

    def teardown(self) -> None:
        """Terminate the whole launcher process group; safe to call twice."""
        process = self._process
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                process.wait()
            else:
                try:
                    process.wait(timeout=_STOP_TIMEOUT_S)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
        if self._log is not None and not self._log.closed:
            self._log.close()

    def __enter__(self) -> WorkspacePlacement:
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None,
                 tb: TracebackType | None) -> None:
        self.teardown()
