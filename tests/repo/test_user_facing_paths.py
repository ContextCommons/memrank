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
"""No message a user can see names a path they do not have.

`pip install memrank` installs the `memrank` package and nothing else: `scripts/`, `docs/`,
`tests/`, `examples/`, `tools/`, `deploy/` and everything else in this repository stay here. A
message that says "start the engines with `scripts/internal/backends.sh up`" therefore sends the
reader to a file that does not exist on their machine, at the moment they are already stuck --
which is what the 2026-09-16 sessions hit, and what ATO-2125 removed.

WHY ONE TEST AND NOT ONE PER SITE. The defect is a category. There were nine of these strings, in
six modules that have nothing to do with each other, and a test per site checks the nine that were
already found while saying nothing about the tenth. This enumerates instead: it walks every module
that ships, and a NEW reference fails here rather than in somebody's terminal.

WHAT COUNTS AS USER-VISIBLE. Every string literal except the docstrings -- error messages, `help=`
text, printed lines -- plus the docstrings of Typer commands, which `--help` prints verbatim. A
module or helper docstring is developer-facing and may say `docs-internal/...` freely; a command's
docstring is output.

WHAT COUNTS AS A BAD REFERENCE. A repository directory absent from an installed copy: every
`internal` prefix in `publish.toml`, plus the four public directories `pyproject.toml` excludes
from the wheel. Derived from the manifest rather than listed here, so the check follows the
boundary when it moves. A URL is exempt -- `https://github.com/ContextCommons/memrank/blob/main/docs`
contains `docs/` and is the correct thing to print, because the reader can open it from anywhere;
`memrank.docs.doc_url` builds those.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from tools import manifest as mf

ROOT = Path(__file__).resolve().parents[2]

#: Public directories that ship in the repository but not in the wheel. `pyproject.toml` excludes
#: `tests*`, `examples*` and `docs*` from `packages.find`, and `tools/` is not a `memrank*` package
#: so it is never collected either.
NOT_IN_THE_WHEEL = ("tests/", "examples/", "docs/", "tools/")


def _forbidden_prefixes(manifest: dict) -> tuple[str, ...]:
    """Every directory prefix an installed copy does not have, longest first.

    Longest first so a match reports the most specific rule that caught it -- `scripts/internal/`
    rather than `scripts/` -- when both would fit.
    """
    internal = tuple(str(r) for r in manifest[mf.INTERNAL] if str(r).endswith("/"))
    return tuple(sorted(set(internal) | set(NOT_IN_THE_WHEEL), key=len, reverse=True))


def _is_command(node: ast.AST) -> bool:
    """Whether ``node`` is a function registered as a Typer command, whose docstring is output."""
    return any(".command(" in ast.unparse(d) or ast.unparse(d).endswith(".command")
               for d in getattr(node, "decorator_list", []))


def _docstring_ids(tree: ast.Module) -> set[int]:
    """The literals that are docstrings and not output: everything but a command's own."""
    ids = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        body = getattr(node, "body", None)
        if not body or not isinstance(body[0], ast.Expr):
            continue
        first = body[0].value
        if isinstance(first, ast.Constant) and isinstance(first.value, str) and not _is_command(node):
            ids.add(id(first))
    return ids


def _inside_a_url(text: str, start: int) -> bool:
    """Whether the match at ``start`` sits inside a URL, which a reader can open from anywhere."""
    token_start = max((text.rfind(c, 0, start) for c in " \t\n(\"'`<"), default=-1)
    return "://" in text[token_start + 1:start]


def _offences(path: Path, prefixes: tuple[str, ...]) -> list[str]:
    """Every user-visible literal in ``path`` naming a directory an installed copy does not have."""
    tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
    skip = _docstring_ids(tree)
    pattern = re.compile("|".join(re.escape(p) for p in prefixes))
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        if id(node) in skip:
            continue
        for match in pattern.finditer(node.value):
            if _inside_a_url(node.value, match.start()):
                continue
            excerpt = node.value[max(0, match.start() - 50):match.end() + 50]
            found.append(f"{path}:{node.lineno}: names {match.group(0)!r} -- ...{excerpt}...")
    return found


@pytest.fixture(scope="module")
def manifest() -> dict:
    return mf.load(ROOT)


@pytest.fixture(scope="module")
def shipped_modules(manifest: dict) -> tuple[Path, ...]:
    """Every `memrank` module that ships. Internal subpackages are nobody's user-facing surface."""
    return tuple(p for p in sorted((ROOT / "memrank").rglob("*.py"))
                 if mf.classify(p.relative_to(ROOT).as_posix(), manifest)[0] == mf.PUBLIC)


def test_shipped_modules_were_actually_found(shipped_modules: tuple[Path, ...]) -> None:
    """The sweep below is vacuous if the enumeration silently finds nothing.

    A classifier change, a moved package or a bad root makes `shipped_modules` empty, and an
    empty-input sweep passes while checking nothing. `preflight.py` is named because it held the
    original offending string, so this also fails if the module the ticket was about goes missing.
    """
    assert len(shipped_modules) > 100, "the shipped-module enumeration collapsed"
    assert any(p.name == "preflight.py" for p in shipped_modules)


def test_no_user_facing_message_names_a_path_absent_from_an_install(
        shipped_modules: tuple[Path, ...], manifest: dict) -> None:
    """The sweep itself. One assertion over every shipped module; ATO-2125."""
    prefixes = _forbidden_prefixes(manifest)
    offences = [o for module in shipped_modules for o in _offences(module, prefixes)]
    assert not offences, (
        "user-facing text names paths an installed copy does not have. State what to set or "
        "what to do instead, or print a URL (`memrank.docs.doc_url`):\n  "
        + "\n  ".join(offences))
