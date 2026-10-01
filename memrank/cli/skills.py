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
"""``memrank skills install``: the instructions a coding agent reads to drive memrank.

    memrank skills install --agent claude     # ~/.claude/skills/memrank
    memrank skills install --agent codex      # ~/.agents/skills/memrank
    memrank skills install --agent cursor     # ~/.cursor/skills/memrank

The skill is the files in ``memrank/skill/``, copied as they are into the folder each host
reads personal skills from. With ``--agent`` it never asks; without it, it asks only at a
terminal and otherwise says which flag to pass. A folder memrank did not write, or one whose
files were edited since, is never overwritten: a marker beside the files records what memrank
wrote, and anything else is the person's.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import typer

from memrank.errors import ActionRequired
from memrank.outcome import Step
from memrank.term import style
from memrank.term.outcome import CHECK

#: The skill's source: one copy in the repository, installed and published from here.
SKILL_SOURCE = Path(__file__).resolve().parent.parent / "skill"
SKILL_FILES = ("SKILL.md", "agent-files.md")
SKILL_NAME = "memrank"

#: Where each host reads a person's own skills, under their home directory, and its name.
HOSTS: dict[str, tuple[str, str]] = {
    "claude": (".claude/skills", "Claude Code"),
    "codex": (".agents/skills", "Codex"),
    "cursor": (".cursor/skills", "Cursor"),
}

MARKER = ".memrank-owned.json"
MARKER_OWNER = "memrank"

skills_app = typer.Typer(help="The memrank skill, for coding agents to evaluate your agent.",
                         no_args_is_help=True)


def skill_folder(home: Path, host: str) -> Path:
    """The folder ``host`` reads the memrank skill from."""
    return home / HOSTS[host][0] / SKILL_NAME


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _not_ours(folder: Path, host: str, why: str) -> ActionRequired:
    return ActionRequired(
        f"{folder} {why}, so memrank will not overwrite it.",
        steps=(Step("Keep your changes elsewhere, remove the folder, then install again:",
                    (f"rm -r {folder}", f"memrank skills install --agent {host}")),))


def _check_ours(folder: Path, host: str) -> None:
    """Refuse unless ``folder`` holds exactly what memrank last wrote there."""
    if folder.is_symlink() or not folder.is_dir():
        raise _not_ours(folder, host, "is not a folder memrank made")
    try:
        marker = json.loads((folder / MARKER).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise _not_ours(folder, host, "was not written by memrank") from exc
    if not isinstance(marker, dict) or marker.get("owner") != MARKER_OWNER \
            or not isinstance(marker.get("files"), dict):
        raise _not_ours(folder, host, "was not written by memrank")
    for name, sha256 in marker["files"].items():
        path = folder / name
        if path.is_symlink() or not path.is_file() or _sha256(path.read_bytes()) != sha256:
            raise _not_ours(folder, host, f"has an edited {name}")


def _write(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` by renaming a finished file over it."""
    staged = path.with_name(f".{path.name}.staged")
    staged.write_bytes(data)
    os.replace(staged, path)


def install(home: Path, host: str, source: Path = SKILL_SOURCE) -> bool:
    """Install the skill for ``host``; ``True`` when a file changed, ``False`` when current."""
    folder = skill_folder(home, host)
    wanted = {name: (source / name).read_bytes() for name in SKILL_FILES}
    if folder.exists() or folder.is_symlink():
        _check_ours(folder, host)
        if all((folder / name).read_bytes() == data for name, data in wanted.items()):
            return False
    folder.mkdir(parents=True, exist_ok=True)
    for name, data in wanted.items():
        _write(folder / name, data)
    marker = {"owner": MARKER_OWNER, "files": {n: _sha256(d) for n, d in wanted.items()}}
    _write(folder / MARKER, json.dumps(marker, indent=2, sort_keys=True).encode())
    return True


def _interactive() -> bool:
    """Whether someone at a terminal can answer. The seam tests patch."""
    return sys.stdin.isatty() and sys.stderr.isatty()


def _chosen_host(agent: str | None) -> str:
    """The host named by ``--agent``, asked for at a terminal, else refused with the flags."""
    if agent is None and not _interactive():
        raise ActionRequired(
            "Which coding agent should get the skill? Nobody is at a terminal to ask.",
            steps=(Step("Name it:", tuple(f"memrank skills install --agent {host}"
                                          for host in HOSTS)),))
    if agent is None:
        agent = typer.prompt(f"Install the skill for which agent ({', '.join(HOSTS)})",
                             err=True).strip()
    if agent not in HOSTS:
        raise typer.BadParameter(f"{agent!r} is not one of {', '.join(HOSTS)}",
                                 param_hint="--agent")
    return agent


@skills_app.command("install")
def cli_install(
    agent: str | None = typer.Option(None, "--agent",
                                     help="Which coding agent: claude, codex or cursor. Asked "
                                          "for at a terminal when not given."),
) -> None:
    """Install the memrank skill where a coding agent reads it. Never overwrites your edits."""
    host = _chosen_host(agent)
    folder = skill_folder(Path.home(), host)
    changed = install(Path.home(), host)
    name = HOSTS[host][1]
    done = "Installed" if changed else "Already installed:"
    style.say(f"{style.good(CHECK)} {done} the memrank skill for {name}, in {folder}")
    style.step(f"{name} finds it in new sessions. A session already running can read it now:",
               (str(folder / "SKILL.md"),))
