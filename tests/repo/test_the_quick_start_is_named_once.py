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
"""The quick-start pair is declared once, and every page and docstring that names it agrees.

0.4.4 made TFIDF on SQuAD the quick start. The README and install.md blocks moved; the prose on
the pages a reader reaches next did not. `docs/install.md` went on saying "where `WordOverlap()`
goes" through two commits that edited that very file with the documentation guards green, and
`reference/catalog.md` and `systems/word-overlap.md` went on giving the first result to the
demoted pair. The retired-name guard cannot see it by design: `WordOverlap` was DEMOTED, not
retired, and a demotion leaves the name valid everywhere.

What made it unguardable was that the pair was declared nowhere. It now is --
`memrank.instrument.catalog.QUICK_START` -- and this file holds three things to it:

1. the README's first Python block and install.md's quick-start block import exactly the pair;
2. the docstrings that teach a first import (`help(memrank)`, `memrank.systems`,
   `memrank.evaluations`, `Evaluation.run`) and examples/01 name exactly the pair;
3. no published sentence gives a first-result role -- "first result", "first number", "quick
   start", "smoke run", "where `X()` goes" -- to any other shipped class. On a page about one
   shipped class, a sentence naming no class is about that page's class ("It is also the system
   a first number is taken on").

The honest limit of (3): a phrase list misses a paraphrase. The declaration is what makes the
next demotion one edit, and this list is what finds the sentences the edit left behind.
"""
from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path

import pytest

import memrank
from memrank import evaluations, systems
from memrank.instrument import catalog as catalog_module
from memrank.instrument.catalog import QUICK_START, ShippedEvaluation, ShippedSystem
from memrank.instrument.evaluation import Evaluation
from tests.repo.prose import ROOT, Sentence, published_sentences, sentences_of
from tests.repo.test_runnable_blocks import BLOCKS

PAIR = {QUICK_START.system, QUICK_START.evaluation}

#: Every shipped class, by its Python name and by its catalog string, mapped to the Python name.
ENTRIES: list[ShippedSystem | ShippedEvaluation] = [*systems.SHIPPED, *evaluations.SHIPPED]
SHIPPED = {**{e.python_name: e.python_name for e in ENTRIES},
           **{e.name: e.python_name for e in ENTRIES}}

#: The phrases that give a class the quick-start role. "first run" is deliberately absent: "cached
#: after the first run" is about a download, not about which pair a reader starts with.
ROLE = re.compile(r"first result|first number|quick[- ]start|smoke run", re.I)

#: "where `X()` goes" gives the role to X alone -- README's "`AtomicMemory` ... take a `base_url=`
#: where `NoteBook()` goes" names four shipped classes and gives the role to none of them.
GOES = re.compile(r"`?(\w+)\(\)`? goes")

#: A shipped Python name as a word, or a catalog string in backticks.
MENTION = re.compile(r"`([\w-]+)(?:\(\))?`|\b([A-Z][A-Za-z0-9]+)\b")

#: Sentences allowed to pair a role phrase with another shipped class, each with its reason.
#: Empty: a demoted class has no first-result role anywhere.
ALLOWED: dict[str, str] = {}

#: Where the pair must be exactly what is imported. The README is not here: its quick start is the
#: one-command install and `memrank run`, and it carries no Python block.
QUICK_START_BLOCKS = ("docs/install.md",)


def _page_subject(path: str) -> str | None:
    """The shipped class a per-object page is about -- `docs/systems/word-overlap.md`."""
    parts = Path(path).parts
    if len(parts) != 3 or parts[0] != "docs" or parts[1] not in ("systems", "evaluations"):
        return None
    slug = Path(path).stem
    return next((python for name, python in SHIPPED.items()
                 if name.replace("_", "-").lower() == slug), None)


def _named(text: str) -> set[str]:
    """The shipped classes a piece of text names."""
    found = set()
    for quoted, bare in MENTION.findall(text):
        name = quoted or bare
        if name in SHIPPED:
            found.add(SHIPPED[name])
    return found


def offences(sentences: list[Sentence]) -> list[str]:
    """Every sentence giving the quick-start role to a class outside the declared pair."""
    found = []
    for sentence in sentences:
        if sentence.where in ALLOWED:
            continue
        named = {SHIPPED[x] for x in GOES.findall(sentence.text) if x in SHIPPED}
        if ROLE.search(sentence.text):
            subject = _page_subject(sentence.path)
            named |= _named(sentence.text) or ({subject} if subject else set())
        if named - PAIR:
            found.append(f"{sentence.where} gives the quick-start role to "
                         f"{sorted(named - PAIR)}, but QUICK_START is {sorted(PAIR)}: "
                         f"{sentence.text!r}")
    return found


def _imports(source: str) -> set[str]:
    """What a block imports from `memrank.systems` and `memrank.evaluations`."""
    return {alias.name
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.ImportFrom)
            and node.module in ("memrank.systems", "memrank.evaluations")
            for alias in node.names}


@pytest.mark.parametrize("path", QUICK_START_BLOCKS)
def test_the_quick_start_block_imports_exactly_the_declared_pair(path: str):
    """The block a reader runs first is the declaration, not a copy of it."""
    first = next((b for b in BLOCKS if b.path == path), None)
    assert first is not None, f"{path} carries no Python block"
    assert _imports(first.source) == PAIR, (
        f"{first.id} imports {sorted(_imports(first.source))}; QUICK_START is {sorted(PAIR)}")


#: The docstrings a reader meets through `help()` and an editor's hover, which teach an import.
DOCSTRINGS = {
    "memrank": memrank.__doc__,
    "memrank.systems": systems.__doc__,
    "memrank.evaluations": evaluations.__doc__,
    "memrank.instrument.catalog": catalog_module.__doc__,
    "memrank.Evaluation.run": inspect.getdoc(Evaluation.run),
}

#: How a docstring teaches a shipped class as THE example: an import, a dotted construction, or
#: a construction by catalog string.
TAUGHT = re.compile(r"from memrank\.(?:systems|evaluations) import (\w+)"
                    r"|memrank\.(?:systems|evaluations)\.(\w+)\(\)"
                    r"|memrank\.(?:system|evaluation)\(\"([\w-]+)\"\)")


@pytest.mark.parametrize("name", sorted(DOCSTRINGS))
def test_every_docstring_example_is_the_declared_pair(name: str):
    """`help(memrank)` taught `Demo().run(system=WordOverlap())` a release after the demotion."""
    taught = {SHIPPED.get(next(filter(None, m)), next(filter(None, m)))
              for m in TAUGHT.findall(DOCSTRINGS[name] or "")}
    assert taught, f"{name}'s docstring teaches no shipped class -- the scan is reading nothing"
    assert taught <= PAIR, f"{name}'s docstring teaches {sorted(taught - PAIR)} as the example"


def test_the_first_example_runs_the_declared_pair():
    """examples/01 is "the shortest working run" that reference/run.md links."""
    source = (ROOT / "examples/01-first-result/run.py").read_text(encoding="utf-8")
    assert _imports(source) == PAIR


def test_no_published_sentence_gives_the_role_to_another_class():
    """The sentence-level half: the pages a reader reaches after the quick start."""
    found = offences(published_sentences())
    assert not found, "\n".join(found)


def test_every_allowance_still_matches_a_sentence():
    """An allowance for a sentence that is gone is an exemption waiting for its next tenant."""
    present = {s.where for s in published_sentences()}
    assert set(ALLOWED) <= present, f"stale allowances: {sorted(set(ALLOWED) - present)}"


def test_the_scan_catches_the_sentences_that_shipped_wrong():
    """Guards the guard, on the three sentences 0.4.4 left behind, verbatim."""
    caught = offences([
        *sentences_of("docs/install.md", "Its clients take a `base_url=` where `WordOverlap()`"
                      " goes in the block above."),
        *sentences_of("docs/reference/catalog.md", "`WordOverlap` and `Demo` need nothing, "
                      "which is why they are what a first result is made of."),
        *sentences_of("docs/systems/word-overlap.md", "It is also the system a first number is"
                      " taken on, because it needs nothing."),
    ])
    assert len(caught) == 3, "\n".join(caught)
    assert not offences(sentences_of("docs/systems/tfidf.md", "It is also the system a first "
                                     "number is taken on, because it needs nothing."))
