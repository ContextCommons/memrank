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
"""The three first surfaces say the same thing about what memrank is.

There are three surfaces a person meets first and no two of them met the same way: `README.md`
called memrank *"an instrument for measuring AI memory engines"*, the package docstring said the
same in different words, and `memrank --help` called it a *"benchmark suite"* -- a different
word for the thing, on the surface with the least room to explain itself. Two recorded feedback
sessions in September 2026 failed at exactly this point: a person who had read the entry page
could not say what memrank measures.

Three surfaces drift because nothing compares them. This compares them: each must carry the
one-line definition, and none may keep a retired one -- because a surface that gains the
sentence and keeps an older description of memrank has half-landed and reads as landed.

`pyproject.toml` is the fourth surface, and it is what PyPI renders, so it carries the sentence
and the word too. `docs/SPEC.md`'s "One line" is the fifth: the specification calls itself the
document to cite in a dispute about method, so the line a citer quotes says the same thing.

The sentence pinned here is the maintainer's own: *"Evaluate your agent on your tasks."* It
replaced decision 0011's *"Memrank is a tool for reproducible, auditable evaluation of memory
systems."* (ATO-2249) once memrank evaluated agents rather than memory systems, and traded some
reproducibility for flexibility and ease of use. With that sentence went the assertion that every
surface calls memrank a "tool": the new sentence is an imperative and names no noun for memrank.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

# `tomllib` is stdlib only from 3.11, and `pyproject.toml` declares `requires-python = ">=3.10"`.
try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover -- 3.10 only
    import tomli as tomllib

from memrank.runner import _APP_HELP

ROOT = Path(__file__).resolve().parents[2]

#: The one line, lowercased: some surfaces open with it, and the package docstring follows
#: "Memrank --" with it.
MEASURES = "evaluate your agent on your tasks"

#: Earlier descriptions of memrank that no entry surface may keep: decision 0011's predicate,
#: and "benchmark suite" -- a fixed set of tests with a scoreboard, which memrank is not.
RETIRED = ("reproducible, auditable evaluation of memory systems", "benchmark suite")


def _normalised(text: str) -> str:
    """Lowercased, with markdown quoting and line wrapping flattened.

    The sentence is wrapped across lines on every surface that has a line length, so a literal
    substring search would only ever match by accident.
    """
    return re.sub(r"\s+", " ", re.sub(r"^[>#\s]+", "", text, flags=re.MULTILINE)).lower()


def _readme_entry() -> str:
    """The README down to its first heading after the title -- what a reader meets first."""
    body = (ROOT / "README.md").read_text(encoding="utf-8").split("\n## ")[0]
    return _normalised(body)


def _package_docstring() -> str:
    """Read as text rather than imported, so this states what the file says."""
    import memrank

    assert memrank.__doc__ is not None, "the package docstring is the entry surface"
    return _normalised(memrank.__doc__)


def _spec_one_line() -> str:
    """`docs/SPEC.md`'s "One line:" paragraph -- the sentence a reader citing the spec quotes.

    SPEC calls itself the document to cite in a disagreement about method, and it went on
    calling memrank "an instrument for evaluating AI-agent memory engines" after the three
    surfaces above had moved to decision 0011's sentence, because it was not one of them.
    """
    text = (ROOT / "docs" / "SPEC.md").read_text(encoding="utf-8")
    paragraph = text.split("**One line:**", 1)[1].split("\n\n", 1)[0]
    return _normalised(paragraph)


SURFACES = {
    "README.md": _readme_entry,
    "docs/SPEC.md": _spec_one_line,
    "memrank/__init__.py": _package_docstring,
    "memrank --help": lambda: _normalised(_APP_HELP),
}


@pytest.mark.parametrize("surface", sorted(SURFACES))
def test_every_entry_surface_says_what_memrank_measures(surface: str):
    """A reader of any one of the three restates the same sentence."""
    assert MEASURES in SURFACES[surface](), f"{surface} does not carry the entry sentence"


@pytest.mark.parametrize("surface", sorted(SURFACES))
def test_no_entry_surface_keeps_a_retired_description(surface: str):
    """Absent as well as present, so a half-done rewording fails rather than passing quietly."""
    text = SURFACES[surface]()
    for retired in RETIRED:
        assert retired not in text, f"{surface} still describes memrank as {retired!r}"


def test_the_distribution_description_agrees_with_them():
    """What PyPI renders is the surface a stranger meets before the repository."""
    with (ROOT / "pyproject.toml").open("rb") as handle:
        description = _normalised(tomllib.load(handle)["project"]["description"])
    assert MEASURES in description
    for retired in RETIRED:
        assert retired not in description
