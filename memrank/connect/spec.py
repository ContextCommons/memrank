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
"""An agent spec: one YAML file naming the agent and how to reach it.

Three connector kinds, all declarative::

    # a preset -- an OpenAI-compatible chat endpoint, started by the runner
    name: my-agent
    connector: openai-chat
    base_url: http://127.0.0.1:8100
    vars: {model: gpt-4.1-mini}
    auth: {env: OPENAI_API_KEY}
    start: {argv: ["{python}", my_agent.py, --port, "8100"], ready: "http://127.0.0.1:8100/health"}

    # an HTTP mapping -- see memrank.connect.http
    connector: http
    base_url: ...
    reset: {path: "/sessions/{session_id}", method: DELETE}
    feed: {per: turn, path: /memory, body: {session: "{session_id}", text: "{text}"}}
    ask: {path: /ask, body: {session: "{session_id}", q: "{question}"}, answer: answer}

    # a program -- see memrank.connect.command
    connector: command
    feed: ["{python}", agent.py, feed, "{session_id}"]
    ask: ["{python}", agent.py, ask, "{session_id}", "{question}"]

``start`` (see :mod:`memrank.connect.process`) is for HTTP agents; without it the agent is
assumed to be running. ``name``, ``version`` and ``description`` are the agent's own words.
Unknown keys are refused, so a misspelled field fails here rather than being ignored.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from memrank.connect.base import Connector, ConnectorConfigError
from memrank.connect.command import CommandConnector, CommandSpec
from memrank.connect.http import HttpConnector, HttpSpec
from memrank.connect.presets import PRESETS, preset_mapping
from memrank.connect.process import StartSpec
from memrank.service.protocol import AgentRef


@dataclass
class AgentSpec:
    """An agent file, loaded: who the agent says it is, how to reach it, how to start it."""

    ref: AgentRef
    connector: Connector
    start: StartSpec | None
    description: str | None


def _read(path: Path) -> tuple[dict[str, Any], str]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ConnectorConfigError(f"cannot read agent spec {path}: {exc}") from exc
    data = yaml.safe_load(raw)
    if not isinstance(data, dict):
        raise ConnectorConfigError(f"agent spec {path} must be a mapping")
    return data, hashlib.sha256(raw).hexdigest()


def _connector(kind: str, fields: dict[str, Any], start: Any) -> Connector:
    if kind == "command":
        if start is not None:
            raise ConnectorConfigError("`start` is for HTTP agents; a command agent is "
                                       "already started once per step")
        return CommandConnector(CommandSpec.model_validate(fields))
    if kind in PRESETS:
        fields = {**preset_mapping(kind, fields.get("vars") or {}), **fields}
    elif kind != "http":
        raise ConnectorConfigError(
            f"unknown connector {kind!r}; use http, command, or a preset "
            f"({', '.join(sorted(PRESETS))})")
    return HttpConnector(HttpSpec.model_validate(fields))


def load_agent(path: Path) -> AgentSpec:
    """The agent a spec file names, ready to start (when it says how) and to carry steps."""
    data, sha256 = _read(path)
    try:
        ref = AgentRef(name=data.pop("name"), version=data.pop("version", None),
                       spec_path=str(path), spec_sha256=sha256)
        description = data.pop("description", None)
        start = data.pop("start", None)
        connector = _connector(str(data.pop("connector")), data, start)
        return AgentSpec(ref=ref, connector=connector, description=description,
                         start=None if start is None else StartSpec.model_validate(start))
    except KeyError as exc:
        raise ConnectorConfigError(f"agent spec {path} is missing {exc.args[0]!r}") from exc
    except ValidationError as exc:
        raise ConnectorConfigError(f"agent spec {path} is invalid:\n{exc}") from exc
