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
"""The library does not import the command line -- with one module excepted, by name.

The command line is being demoted to a wrapper over the Python interface, and the library
cannot be the primary surface while it imports the surface being demoted.
:mod:`memrank.orchestration.sweep` did exactly that: `from memrank.cli import sync as sync_cli`,
for the sake of three `auto_sync` calls, so a library sweep pulled every verb and the argument
parser into a caller who asked for an evaluation. The push it needed now lives in
:mod:`memrank.runs.push` (ATO-2142).

The exception is :mod:`memrank.runner`, which IS the command line's assembly point -- it exists
to collect the verbs into one app, and a rule that forbade it importing them would forbid the
command line existing.

Static rather than a subprocess import: the question is which modules the source declares a
dependency on, and a dependency inside a function body is exactly the escape hatch this should
still see. ``tests/repo/test_import_weight.py`` asks the runtime question -- what a completed
run actually loads -- and the two are deliberately different checks.
"""
from __future__ import annotations

import ast
from pathlib import Path

#: The package the library may not depend on.
_CLI_PACKAGE = "memrank.cli"

#: The command line's own assembly point, and the only module excepted. Spelled as a path so
#: the failure message and this list read the same way.
_ASSEMBLY_POINT = "memrank/runner.py"


def _package_root() -> Path:
    return Path(__file__).resolve().parents[2] / "memrank"


def _imports_the_cli(path: Path) -> list[tuple[int, str]]:
    """Every declared import of `memrank.cli` in one module, as ``(line, what)``.

    Both statement forms, and at any depth -- ``ast.walk`` reaches a deferred import inside a
    function, which is the shape someone would reach for to get around this.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [(node.lineno, alias.name) for alias in node.names
                      if _is_cli(alias.name)]
        elif isinstance(node, ast.ImportFrom) and node.module and _is_cli(node.module):
            found += [(node.lineno, f"{node.module}.{alias.name}") for alias in node.names]
    return found


def _is_cli(module: str) -> bool:
    return module == _CLI_PACKAGE or module.startswith(f"{_CLI_PACKAGE}.")


def test_no_library_module_imports_the_command_line():
    root = _package_root().parent
    offenders = []
    for path in sorted(_package_root().rglob("*.py")):
        module = path.relative_to(root).as_posix()
        if module.startswith("memrank/cli/") or module == _ASSEMBLY_POINT:
            continue
        offenders += [f"{module}:{line} imports {what}" for line, what in _imports_the_cli(path)]
    assert offenders == [], (
        "these import the command-line package from outside it:\n  " + "\n  ".join(offenders)
        + "\n\nThe command line is a wrapper over the library, so the dependency runs one way. "
        "Move the code the library needs into a library module -- `memrank.runs.push` is the "
        "precedent, and `memrank.runs.record` the one before it -- and let the command line "
        f"call it. Only {_ASSEMBLY_POINT}, which assembles the verbs into the app, may import "
        "them.")


def test_the_assembly_point_still_is_one():
    """The exception is load-bearing, so it must not quietly stop applying.

    If `memrank/runner.py` ever stops importing the verbs, the exception above is excusing
    nothing and the next module to need excusing would be added beside it as though there were
    a precedent for two.
    """
    imports = _imports_the_cli(_package_root().parent / _ASSEMBLY_POINT)

    assert imports, (
        f"{_ASSEMBLY_POINT} no longer imports {_CLI_PACKAGE}, so it is not the command line's "
        f"assembly point any more. Remove its exception from this file rather than leaving an "
        f"allowance nothing needs.")
