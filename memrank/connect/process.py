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
"""Starting an HTTP agent for the length of a run, when its spec says how.

An agent spec may carry ``start: {argv: [...], ready: <url>}``. The runner launches that
command before the leak probe, waits until ``ready`` answers 2xx, and stops the process when
the run ends -- however it ends. Waiting is the one place the runner watches a clock, because a
server that is still booting has no other way to say so; the wait is bounded, and a process
that exits, or never becomes ready, fails the run with the tail of its own log. An agent spec
without ``start`` is taken to be running already and is simply called.

``argv`` may use ``{python}`` (this interpreter) and ``{evaluation}`` (the run's eval ref, or its
evaluation file's path), so one spec can start an agent that answers the way each evaluation
asks -- the full-context reference agent reads it to pick the evaluation's reader prompt. Any
other ``{name}`` is refused before anything starts.
"""

from __future__ import annotations

import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import httpx
from pydantic import BaseModel, ConfigDict, Field

from memrank.connect import template
from memrank.errors import ActionRequired
from memrank.outcome import Step

_LOG_TAIL_CHARS = 2000
_POLL_S = 0.5
_STOP_GRACE_S = 10.0


class StartSpec(BaseModel):
    """How to launch an HTTP agent, and the URL that answers once it is ready."""

    model_config = ConfigDict(extra="forbid")

    argv: list[str] = Field(min_length=1)
    ready: str
    ready_timeout_s: float = Field(default=120.0, gt=0)


class AgentStartFailed(ActionRequired):
    """The agent's start command exited, or never became ready.

    A step for the user rather than a memrank failure: the agent is theirs, and what it said is
    in its log, whose path is the line to copy. ``output`` is the end of that log.
    """

    def __init__(self, statement: str, log: Path) -> None:
        super().__init__(statement, steps=(Step("See why in its full output:", (str(log),)),))
        self.output = _tail(log)

    def __str__(self) -> str:
        return f"{super().__str__()}\nThe end of its output:\n{self.output}"


def _ready(url: str) -> bool:
    try:
        return httpx.get(url, timeout=2.0).is_success
    except httpx.TransportError:
        return False


def _tail(log: Path) -> str:
    text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
    return text[-_LOG_TAIL_CHARS:].strip() or "(no output)"


def _argv(spec: StartSpec, evaluation: str) -> list[str]:
    variables = {"python": sys.executable, "evaluation": evaluation}
    return [str(template.render(arg, variables)) for arg in spec.argv]


def wait_ready(process: subprocess.Popen[bytes], spec: StartSpec, log: Path, *,
               ready: Callable[[str], bool] = _ready,
               clock: Callable[[], float] = time.monotonic,
               sleep: Callable[[float], None] = time.sleep) -> None:
    """Return once ``spec.ready`` answers; raise if the process exits or the wait runs out."""
    deadline = clock() + spec.ready_timeout_s
    while not ready(spec.ready):
        if process.poll() is not None:
            raise AgentStartFailed(f"Your agent's start command exited with "
                                   f"{process.returncode} before {spec.ready} answered.", log)
        if clock() >= deadline:
            raise AgentStartFailed(f"Your agent never became ready: {spec.ready} did not answer "
                                   f"within {spec.ready_timeout_s}s of starting it.", log)
        sleep(_POLL_S)


def _stop(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=_STOP_GRACE_S)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


@contextmanager
def started(spec: StartSpec | None, log: Path, evaluation: str) -> Iterator[None]:
    """Run the agent's start command for the duration of the block, or do nothing without one.

    ``evaluation`` is what ``{evaluation}`` in the command becomes.
    """
    if spec is None:
        yield
        return
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("wb") as sink:
        process = subprocess.Popen(_argv(spec, evaluation), stdout=sink, stderr=subprocess.STDOUT)
        try:
            wait_ready(process, spec, log)
            yield
        finally:
            _stop(process)
