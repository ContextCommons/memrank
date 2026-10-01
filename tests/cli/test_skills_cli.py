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
"""``memrank skills install``: where each host's skill lands, and what it never overwrites."""

from __future__ import annotations

import io
import json
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from memrank.cli import skills
from memrank.runner import main

Invoke = Callable[..., tuple[int, str]]


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


@pytest.fixture
def invoke(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> Invoke:
    """``memrank skills install ...`` through the real boundary: its exit code and output."""
    def run(*args: str, interactive: bool = False, answer: str = "") -> tuple[int, str]:
        monkeypatch.setattr(skills, "_interactive", lambda: interactive)
        monkeypatch.setattr(sys, "stdin", io.StringIO(answer))
        monkeypatch.setattr(sys, "argv", ["memrank", "skills", "install", *args])
        with pytest.raises(SystemExit) as exited:
            main()
        shown = capsys.readouterr()
        return int(exited.value.code or 0), shown.out + shown.err
    return run


@pytest.mark.parametrize(("host", "folder"), [
    ("claude", ".claude/skills/memrank"),
    ("codex", ".agents/skills/memrank"),
    ("cursor", ".cursor/skills/memrank"),
])
def test_each_host_gets_the_skill_where_it_reads_personal_skills(home, invoke, host, folder):
    code, output = invoke("--agent", host)

    assert code == 0, output
    for name in skills.SKILL_FILES:
        installed = (home / folder / name).read_bytes()
        assert installed == (skills.SKILL_SOURCE / name).read_bytes()
    assert str(home / folder / "SKILL.md") in output, "a running session is told the path"


def test_without_a_terminal_it_names_the_flag_and_installs_nothing(home, invoke):
    code, output = invoke()

    assert code == 1
    for host in skills.HOSTS:
        assert f"memrank skills install --agent {host}" in output
    assert not any(home.iterdir())


def test_at_a_terminal_it_asks_which_agent(home, invoke):
    code, output = invoke(interactive=True, answer="codex\n")

    assert code == 0, output
    assert (home / ".agents/skills/memrank/SKILL.md").is_file()


@pytest.mark.parametrize("args", [("--agent", "vim"), ()])
def test_an_unknown_agent_is_refused(home, invoke, args):
    code, output = invoke(*args, interactive=True, answer="vim\n")

    assert code == 2
    assert "not one of claude, codex, cursor" in output


def test_a_rerun_changes_nothing_and_a_new_version_replaces_its_own_files(home, tmp_path):
    assert skills.install(home, "claude") is True
    assert skills.install(home, "claude") is False
    newer = tmp_path / "newer"
    newer.mkdir()
    for name in skills.SKILL_FILES:
        (newer / name).write_text(f"new {name}\n", encoding="utf-8")

    assert skills.install(home, "claude", source=newer) is True
    assert (skills.skill_folder(home, "claude") / "SKILL.md").read_text() == "new SKILL.md\n"


def _edited(folder: Path) -> None:
    with (folder / "SKILL.md").open("a", encoding="utf-8") as skill:
        skill.write("my own note\n")


def _unmanaged(folder: Path) -> None:
    (folder / skills.MARKER).unlink()


def _foreign_marker(folder: Path) -> None:
    (folder / skills.MARKER).write_text(json.dumps({"owner": "someone", "files": {}}))


def _garbled_marker(folder: Path) -> None:
    (folder / skills.MARKER).write_text("[1, 2")


def _symlinked(folder: Path) -> None:
    real = folder.with_name("elsewhere")
    folder.rename(real)
    folder.symlink_to(real, target_is_directory=True)


@pytest.mark.parametrize("damage", [_edited, _unmanaged, _foreign_marker, _garbled_marker,
                                    _symlinked])
def test_a_folder_memrank_did_not_write_as_it_is_is_never_overwritten(home, invoke, damage):
    folder = skills.skill_folder(home, "claude")
    skills.install(home, "claude")
    damage(folder)
    before = {p.name: p.read_bytes() for p in folder.iterdir()}

    code, output = invoke("--agent", "claude")

    assert code == 1
    assert "memrank will not overwrite it" in output
    assert f"rm -r {folder}" in output
    assert {p.name: p.read_bytes() for p in folder.iterdir()} == before
