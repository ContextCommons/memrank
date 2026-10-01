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
"""The agents memrank ships, as agent files inside the package.

A name given to ``memrank run --agent`` resolves only here -- to a file shipped in this
directory -- and anything else is a path to an agent file. There is no search path, no
configuration folder and no plugin: what a name means is fixed by the installed version.
"""

from __future__ import annotations

from pathlib import Path

from memrank.connect.base import ConnectorConfigError
from memrank.errors import ActionRequired
from memrank.outcome import Step

SHIPPED_DIR = Path(__file__).parent


class AgentNotFound(ConnectorConfigError, ActionRequired):
    """``--agent`` names neither a shipped agent nor a file: the user has one to point at."""


def shipped() -> dict[str, Path]:
    """Every shipped agent file, by name (the file's stem)."""
    return {path.stem: path for path in sorted(SHIPPED_DIR.glob("*.yaml"))}


def resolve(ref: str) -> Path:
    """The agent file ``ref`` names: a shipped agent by name, else a path that must exist."""
    named = shipped()
    if ref in named:
        return named[ref]
    path = Path(ref)
    if path.is_file():
        return path
    raise AgentNotFound(
        f"{ref!r} is neither a shipped agent ({', '.join(named) or 'none'}) nor an agent file.",
        steps=(Step("Pass the path to your agent's YAML file, or the name of a shipped agent. "
                    "To list the shipped agents:", ("memrank agents ls",)),))
