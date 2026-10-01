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
"""Running an evaluation file's own programs: a case source, a grader.

A command is an argv list, run without a shell from the evaluation file's folder, so a script
named beside the file is found and an argument is never re-parsed. ``{python}`` is this
interpreter, as in an agent file. The files a command names are part of the evaluation, so
:func:`command_files` finds them for its fingerprint.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from memrank.errors import ActionRequired

#: How much of a failed command's stderr its refusal shows.
STDERR_TAIL_CHARS = 800


class CommandFailed(ActionRequired):
    """An evaluation file's program failed: it exited non-zero, timed out, or is missing."""


def argv_of(command: list[str]) -> list[str]:
    """``command`` with ``{python}`` replaced by this interpreter."""
    return [sys.executable if arg == "{python}" else arg for arg in command]


def command_files(command: list[str], folder: Path) -> list[Path]:
    """The existing files ``command`` names, relative to ``folder`` unless absolute."""
    found = []
    for arg in command:
        if arg == "{python}" or arg.startswith("-"):
            continue
        candidate = Path(arg) if Path(arg).is_absolute() else folder / arg
        if candidate.is_file():
            found.append(candidate)
    return found


def _program(argv: list[str], folder: Path) -> str:
    """The program to execute: a file beside the evaluation, or one on ``PATH``."""
    local = folder / argv[0]
    if not Path(argv[0]).is_absolute() and "/" in argv[0] and local.is_file():
        return str(local)
    return argv[0]


def run(command: list[str], folder: Path, *, stdin: str, timeout_s: float, what: str) -> str:
    """Run ``command`` from ``folder`` and return its stdout; refuse loudly when it fails.

    Args:
        command: The argv as written in the evaluation file.
        folder: The evaluation file's folder, where it runs.
        stdin: What the program reads.
        timeout_s: How long it may take.
        what: What the program is, for the refusal (``the case command``).
    """
    argv = argv_of(command)
    argv[0] = _program(argv, folder)
    if shutil.which(argv[0]) is None and not Path(argv[0]).is_file():
        raise CommandFailed(f"{what} {command[0]!r} is not a file beside the evaluation and "
                            "not a program on PATH.")
    try:
        done = subprocess.run(argv, input=stdin, capture_output=True, text=True, cwd=folder,
                              timeout=timeout_s, check=False)
    except subprocess.TimeoutExpired as exc:
        raise CommandFailed(f"{what} did not finish within {timeout_s}s: "
                            f"{' '.join(command)}") from exc
    if done.returncode != 0:
        tail = done.stderr[-STDERR_TAIL_CHARS:].strip() or "(no output on stderr)"
        raise CommandFailed(f"{what} exited {done.returncode}: {' '.join(command)}\n{tail}")
    return done.stdout
