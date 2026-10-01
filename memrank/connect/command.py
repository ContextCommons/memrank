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
"""The command connector: an agent that is a program, driven by its argv.

Each operation is an argv template, run once per step without a shell, so an argument is
never re-parsed. Feed writes the case's transcript to the program's stdin; ask reads the
answer from its stdout. A non-zero exit is a failure, with the tail of stderr as its reason.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from memrank.connect import template
from memrank.connect.base import AgentError, ConnectorConfigError
from memrank.service.protocol import Step

_STDERR_TAIL_CHARS = 500


class CommandSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reset: list[str] | None = None
    feed: list[str] = Field(min_length=1)
    ask: list[str] = Field(min_length=1)
    timeout_s: float = Field(default=600.0, gt=0)
    vars: dict[str, Any] = Field(default_factory=dict)


class CommandConnector:
    """Carries steps to an agent by running its commands."""

    def __init__(self, spec: CommandSpec) -> None:
        for argv in (spec.reset, spec.feed, spec.ask):
            program = str(template.render(argv[0], {"python": sys.executable})) if argv else ""
            if argv is not None and shutil.which(program) is None:
                raise ConnectorConfigError(f"agent command {program!r} is not on PATH")
        self.spec = spec

    def close(self) -> None:
        """Nothing is held between steps."""

    def _run(self, argv: list[str], variables: dict[str, Any], stdin: str = "") -> str:
        rendered = [str(template.render(arg, variables) or "") for arg in argv]
        try:
            done = subprocess.run(rendered, input=stdin, capture_output=True, text=True,
                                  timeout=self.spec.timeout_s, check=False)
        except subprocess.TimeoutExpired as exc:
            raise AgentError(f"{rendered[0]} did not finish in {self.spec.timeout_s}s") from exc
        if done.returncode != 0:
            raise AgentError(f"{rendered[0]} exited {done.returncode}: "
                             f"{done.stderr[-_STDERR_TAIL_CHARS:].strip()}")
        return done.stdout

    def reset(self, step: Step) -> None:
        if self.spec.reset is not None:
            self._run(self.spec.reset, template.base_vars(step, self.spec.vars))

    def feed(self, step: Step) -> None:
        variables = template.case_vars(step, self.spec.vars)
        self._run(self.spec.feed, variables, stdin=variables["transcript"])

    def ask(self, step: Step) -> str:
        return self._run(self.spec.ask, template.ask_vars(step, self.spec.vars)).strip()
