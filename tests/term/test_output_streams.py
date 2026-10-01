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
"""The stream chokepoint: stdout carries the answer, stderr carries everything else.

stdout is what a caller redirects -- run ids, ``--json``, log content, the rows a listing was
asked for. Narration about producing that answer (progress, confirmations, hints, warnings,
errors) is stderr, so ``memrank runs ls --json > runs.json`` yields JSON and nothing else.

Enforced by ENUMERATION rather than per-command review, because per-command discipline is how
the leak happened: the run path was exemplary while `secrets`, `auth` and `config` narrated onto
stdout for months without a test noticing. Every module is walked, so a new command physically
cannot pick a stream by hand -- the only emitters are :func:`memrank.term.style.out` and
:func:`memrank.term.style.say`, and the choice is made by naming what the line IS.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest
from typer.testing import CliRunner

from memrank.runner import app

runner = CliRunner()

PACKAGE = Path(__file__).resolve().parents[2] / "memrank"

#: The one module allowed to touch a stream directly -- it IS the chokepoint. Package-relative,
#: not a bare filename: matching on name alone would exempt any future module called `style.py`.
CHOKEPOINT = Path("term/style.py")

#: The one deliberate bypass, named so it cannot grow silently: `watch`'s redraw writes ANSI
#: cursor movement straight to stderr because `typer.echo` strips escape codes when it does not
#: believe the destination is a terminal, which would erase the very codes that path emits.
#: It is stderr -- the rule it bypasses is the chokepoint's, not the stream's.
DIRECT_STDERR_WRITER = Path("cli/watch.py")

#: Names that pick a stream by hand. `print` is here too: `memrank/config.py` used one, which put
#: a confirmation on stdout AND outside the NO_COLOR gate at the same time.
FORBIDDEN_CALLS = ("typer.echo", "typer.secho", "print", "sys.stdout.write", "sys.stderr.write")


def _dotted(node: ast.AST) -> str:
    """``sys.stdout.write`` for an attribute chain, ``print`` for a bare name, else ``''``."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return ""
    parts.append(node.id)
    return ".".join(reversed(parts))


def raw_emissions(source: str) -> list[tuple[int, str]]:
    """Every call in ``source`` that writes to a stream without going through `style`."""
    return [(node.lineno, name)
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Call) and (name := _dotted(node.func)) in FORBIDDEN_CALLS]


def _modules() -> list[Path]:
    return sorted(p for p in PACKAGE.rglob("*.py")
                  if p.relative_to(PACKAGE) != CHOKEPOINT)


@pytest.mark.parametrize("module", _modules(), ids=lambda p: str(p.relative_to(PACKAGE)))
def test_no_module_picks_a_stream_by_hand(module):
    """Emission goes through `style.out`/`style.say`, so the stream is chosen by naming the line."""
    found = raw_emissions(module.read_text(encoding="utf-8"))
    if module.relative_to(PACKAGE) == DIRECT_STDERR_WRITER:
        found = [(line, name) for line, name in found if name != "sys.stderr.write"]
    assert found == [], f"{module.relative_to(PACKAGE)} writes to a stream directly: {found}"


#: The only modules allowed to import `rich`. They render into a buffer and hand the lines to
#: `style.out`, which is what lets a table exist without a second emitter existing beside it.
RENDERERS = (Path("term/table.py"), Path("term/detail.py"))


def rich_imports(source: str) -> list[int]:
    """Every line in ``source`` that pulls `rich` in, by either import form."""
    lines = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            lines += [node.lineno for alias in node.names
                      if alias.name == "rich" or alias.name.startswith("rich.")]
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == "rich" or module.startswith("rich."):
                lines.append(node.lineno)
    return sorted(lines)


@pytest.mark.parametrize("module", _modules(), ids=lambda p: str(p.relative_to(PACKAGE)))
def test_only_the_renderers_import_rich(module):
    """`rich` is a RENDERER, and a renderer that reaches a stream is a second chokepoint.

    Enumerated for the same reason the stream rule is: `rich.print` writes to stdout directly and
    would sail past `raw_emissions`, which only knows the names it was told about. Confining the
    import is the check that does not depend on guessing every way a library can emit -- and it
    keeps command modules rendering through `term`, where the two layouts and the NO_COLOR gate
    live, rather than each growing a Console of its own.
    """
    if module.relative_to(PACKAGE) in RENDERERS:
        return
    found = rich_imports(module.read_text(encoding="utf-8"))
    assert found == [], (f"{module.relative_to(PACKAGE)} imports rich at line(s) {found}; "
                         f"render through memrank.term.table / memrank.term.detail instead")


def test_the_rich_scan_would_catch_either_import_form():
    """Guards the guard: `import rich.table` and `from rich import box` are the same leak."""
    planted = ("import os\n"
               "import rich.table\n"
               "from rich import box\n"
               "from rich.console import Console\n"
               "from richness import nothing\n")

    assert rich_imports(planted) == [2, 3, 4]


def test_the_scan_would_catch_a_reintroduced_raw_echo():
    """Guards the guard: a laxer scan passes everything, which is worse than no test."""
    planted = "import typer\ndef f():\n    typer.echo('narration')\n    print('worse')\n"

    assert raw_emissions(planted) == [(3, "typer.echo"), (4, "print")]


def test_the_scan_does_not_flag_the_sanctioned_emitters():
    """`style.out`/`style.say` are the point of the rule, not violations of it."""
    assert raw_emissions("from memrank import style\nstyle.out('a')\nstyle.say('b')\n") == []


# --- and the rule holds at runtime, not only in the source -------------------------------------- #

@pytest.fixture
def wallet(monkeypatch, tmp_path):
    """This test's own config dir and run store -- `secrets set` and `config set` really write."""
    monkeypatch.setenv("MEMRANK_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("MEMRANK_RUNS_DIR", str(tmp_path / "runs"))
    return tmp_path


def test_storing_a_secret_says_nothing_on_stdout(wallet):
    """`secrets set` produces no answer -- only a confirmation, which a pipe must not receive."""
    result = runner.invoke(app, ["secrets", "set", "OPENAI_API_KEY", "--value", "sk-test"])

    assert result.exit_code == 0, result.output
    assert result.stdout == ""
    assert "stored OPENAI_API_KEY" in result.stderr


def test_setting_a_default_says_nothing_on_stdout(wallet):
    result = runner.invoke(app, ["config", "set", "defaults.on", "local"])

    assert result.exit_code == 0, result.output
    assert result.stdout == ""
    assert "defaults.on = local" in result.stderr


def test_a_listing_puts_its_rows_on_stdout_and_its_hints_on_stderr(wallet):
    """The other half of the same rule: an answer must NOT be diverted to stderr."""
    result = runner.invoke(app, ["config", "ls"])

    assert result.exit_code == 0, result.output
    assert "defaults.on" in result.stdout


def test_json_output_is_alone_on_stdout(wallet):
    """`--json` is the shape a script parses: notices still narrate, but never into the parse."""
    import json

    result = runner.invoke(app, ["runs", "ls", "--json"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == []
