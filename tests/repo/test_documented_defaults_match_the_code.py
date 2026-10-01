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
"""A documented service address is the address the shipped system actually defaults to.

Three published pages gave Hindsight's default as `http://localhost:7000` -- the command line's
defaults table, its `export` example, and local-development's -- while
`memrank/adapters/hindsight.py` defaulted to 8888, as did the Hindsight page itself and the
target manifest. A reader who trusted the table pointed memrank at a port nothing listened on.

The truth is read from the systems, not from a second table: each shipped system that takes its
address from an environment variable is constructed with that variable unset, and the
`base_url` it settles on is the default. Then every published line or sentence that pairs one of
those variables with a `http://localhost:N` address, and every `System(base_url="...")` example,
must agree with it.
"""
from __future__ import annotations

import os
import re

from memrank import systems
from tests.repo.prose import ROOT, Sentence, published_sentences, sentences_of
from tests.repo.test_runnable_blocks import DOCUMENTS

ADDRESS = re.compile(r"http://localhost:\d+")
CONSTRUCTED = re.compile(r"\b(\w+)\(base_url=\"(http://localhost:\d+)\"")

#: The variable a catalog entry says it reads its address from -- "a running engine, at
#: base_url= or HINDSIGHT_API_URL".
NEEDS_VARIABLE = re.compile(r"\b[A-Z][A-Z0-9_]*_URL\b")


def _defaults() -> tuple[dict[str, str], dict[str, str]]:
    """(variable -> default, class name -> default), with every variable unset while read."""
    by_variable, by_class = {}, {}
    for entry in systems.SHIPPED:
        declared = NEEDS_VARIABLE.search(entry.needs)
        if declared is None:
            continue
        variable = declared.group(0)
        saved = os.environ.pop(variable, None)
        try:
            built = getattr(systems, entry.python_name)()
        finally:
            if saved is not None:
                os.environ[variable] = saved
        assert built.base_url_env == variable, f"{entry.python_name} reads {built.base_url_env}"
        by_variable[variable] = built.base_url
        by_class[entry.python_name] = built.base_url
    return by_variable, by_class


BY_VARIABLE, BY_CLASS = _defaults()
VARIABLE = re.compile(r"\b(" + "|".join(map(re.escape, BY_VARIABLE)) + r")\b")


def offences(sentence: Sentence) -> list[str]:
    """A documented default that is not the one the code settles on."""
    found = []
    variables, addresses = VARIABLE.findall(sentence.text), ADDRESS.findall(sentence.text)
    if len(set(variables)) == 1 and len(set(addresses)) == 1:
        variable, address = variables[0], addresses[0]
        if address != BY_VARIABLE[variable]:
            found.append(f"{sentence.where} gives {variable} as {address}; "
                         f"the code defaults to {BY_VARIABLE[variable]}")
    for name, address in CONSTRUCTED.findall(sentence.text):
        if name in BY_CLASS and address != BY_CLASS[name]:
            found.append(f"{sentence.where} constructs {name} at {address}; "
                         f"it defaults to {BY_CLASS[name]}")
    return found


def _lines() -> list[Sentence]:
    """Every raw line, fenced blocks included -- an `export VAR=...` lives in a shell block."""
    return [Sentence(path, number, line)
            for path in DOCUMENTS
            for number, line in enumerate(
                (ROOT / path).read_text(encoding="utf-8").splitlines(), start=1)]


def test_every_documented_default_is_the_codes():
    found = sorted({o for unit in (*_lines(), *published_sentences()) for o in offences(unit)})
    assert not found, "\n".join(found)


def test_every_system_that_reads_an_address_is_known():
    """Guards the guard: the five service-backed systems all contributed a default."""
    assert set(BY_CLASS) == {"AtomicMemory", "Hindsight", "Supermemory", "Mem0", "Native"}


def test_the_scan_catches_the_default_that_shipped_wrong():
    caught = [o for s in sentences_of("d.md", "| Hindsight | `HINDSIGHT_API_URL` | "
                                      "`http://localhost:7000` |") for o in offences(s)]
    assert len(caught) == 1
    assert not [o for s in sentences_of("d.md", "export HINDSIGHT_API_URL=http://localhost:8888")
                for o in offences(s)]
