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
"""Published prose as sentences, for the guards that check what a page CLAIMS.

The older documentation guards read syntactic units -- a link target, a fenced block, a command
invocation. The claim guards (`test_the_quick_start_is_named_once.py`,
`test_prose_matches_the_catalog.py`, `test_prose_commands_exist.py`,
`test_documented_defaults_match_the_code.py`) read sentences, because a claim is a sentence: a
count and the noun it counts, a role and the class it is given to, can sit on two lines of one
wrapped paragraph. This module is the one place that decides what a sentence is.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from tests.repo.test_doc_links import _strip_fenced_blocks
from tests.repo.test_runnable_blocks import DOCUMENTS

ROOT = Path(__file__).resolve().parents[2]

#: A sentence ends at terminal punctuation followed by whitespace. Abbreviations in the corpus
#: ("e.g.") split a sentence early, which can only make a claim guard miss, never invent a hit.
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

#: Lines that are their own unit whatever surrounds them: a table row, a heading, a list item.
STANDALONE = re.compile(r"^\s*(\||#|[-*] |\d+\. )")


@dataclass(frozen=True)
class Sentence:
    """One sentence of one published page, with the line its paragraph starts on."""

    path: str
    line: int
    text: str

    @property
    def where(self) -> str:
        return f"{self.path}:{self.line}"


def _units(text: str) -> list[tuple[int, str]]:
    """Paragraphs, list items and table rows, each joined onto one line, with its first line."""
    units: list[tuple[int, list[str]]] = []
    for number, line in enumerate(_strip_fenced_blocks(text).splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("<!--"):
            units.append((number, []))
            continue
        if STANDALONE.match(line) or not units or not units[-1][1]:
            units.append((number, [line.strip()]))
        else:
            units[-1][1].append(line.strip())
    return [(first, " ".join(lines)) for first, lines in units if lines]


def sentences_of(path: str, text: str) -> list[Sentence]:
    """Every sentence of one document, outside fenced blocks."""
    return [Sentence(path, first, piece)
            for first, unit in _units(text)
            for piece in SENTENCE_END.split(unit) if piece]


def published_sentences() -> list[Sentence]:
    """Every sentence of every public markdown document the published tree carries."""
    return [sentence
            for path in DOCUMENTS
            for sentence in sentences_of(path, (ROOT / path).read_text(encoding="utf-8"))]
