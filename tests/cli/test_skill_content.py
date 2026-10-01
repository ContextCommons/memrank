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
"""The skill a coding agent follows names only commands and flags memrank has.

A skill that tells an agent to run a flag memrank dropped sends it into a refusal it was never
told how to read, so every ``memrank ...`` line in the skill is checked against the CLI itself.
"""

from __future__ import annotations

import re
import shlex

import pytest
import typer
import yaml

from memrank.cli import skills
from memrank.runner import app

#: The Agent Skills format's limits, which Codex and Cursor enforce (name <= 64, description
#: <= 1024) and Claude Code truncates at (1,536 characters).
NAME = re.compile(r"^[a-z0-9-]{1,64}$")
DESCRIPTION_CHARS = 1024

#: A command is a line of a fenced block, or an inline code span, that starts with ``memrank``.
FENCE = re.compile(r"^```[a-z]*\n(.*?)^```", re.M | re.S)
SPAN = re.compile(r"`(memrank [^`]+)`")


def _frontmatter() -> dict:
    text = (skills.SKILL_SOURCE / "SKILL.md").read_text(encoding="utf-8")
    head, _, _ = text.removeprefix("---\n").partition("\n---\n")
    return yaml.safe_load(head)


def test_the_frontmatter_is_what_every_host_accepts():
    front = _frontmatter()

    assert front["name"] == skills.SKILL_NAME and NAME.match(front["name"])
    assert 0 < len(front["description"]) <= DESCRIPTION_CHARS


def test_every_file_the_skill_links_is_installed_with_it():
    text = (skills.SKILL_SOURCE / "SKILL.md").read_text(encoding="utf-8")
    linked = set(re.findall(r"\]\(([\w.-]+\.md)\)", text))

    assert linked and linked <= set(skills.SKILL_FILES)


def _command_lines() -> list[str]:
    lines = []
    for name in skills.SKILL_FILES:
        text = (skills.SKILL_SOURCE / name).read_text(encoding="utf-8")
        fenced = [row for block in FENCE.findall(text) for row in block.splitlines()]
        lines += [row.split("#")[0].strip() for row in fenced
                  if row.strip().startswith("memrank ")]
        lines += SPAN.findall(text)
    return sorted(set(lines))


@pytest.mark.parametrize("line", _command_lines())
def test_every_command_the_skill_gives_exists_with_its_flags(line):
    root = typer.main.get_command(app)
    command = root
    words = shlex.split(line.replace("...", ""))[1:]
    while words and hasattr(command, "commands") and words[0] in command.commands:
        command = command.commands[words.pop(0)]
    flags = {opt for param in command.params for opt in (*param.opts, *param.secondary_opts)}
    used = {word for word in words if word.startswith("--")}

    assert command is not root or words == ["--version"], f"{line!r} names no command"
    assert not hasattr(command, "commands") or command is root, f"{line!r} names a group"
    assert used <= flags, f"{line!r} uses {sorted(used - flags)}"
