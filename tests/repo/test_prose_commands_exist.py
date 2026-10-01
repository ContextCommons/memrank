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
"""A `memrank <verb>` named in published prose is a command the CLI has.

`test_docs_teach_the_current_cli.py` catches a RETIRED form inside an invocation of `memrank
submit` or `memrank run`. It cannot see a command that never existed, and it does not read a
retired flag outside such an invocation: `docs/methodology.md` said "`memrank list-adapters` does
not offer it" -- a command the CLI answers with "No such command" -- and `docs/systems.md` said a
target "only supports `--adapter myengine`", a flag that exits 2 naming its replacement.

So every inline-code span in a published sentence is read two ways:

- `memrank <verb> [<sub>]` must resolve in the Typer app -- the verb, and the subcommand where the
  verb is a group and the next word is a plain word rather than a flag or a placeholder;
- a flag from `memrank.cli.retired.RETIRED_FLAGS` fails unless its sentence is naming it as
  retired ("retired", "deprecated", "no longer", "refuse(d)", "exits 2").
"""
from __future__ import annotations

import re

import click
import typer.main

from memrank.cli.retired import RETIRED_FLAGS
from memrank.runner import app
from tests.repo.prose import Sentence, published_sentences, sentences_of

CLI = typer.main.get_command(app)
SPAN = re.compile(r"`([^`]+)`")
VERB = re.compile(r"(?:^|\s)memrank\s+([a-z][\w-]*)(?:\s+([a-z][\w-]*))?")
NAMED_AS_RETIRED = re.compile(r"retired|deprecated|no longer|refus|exits 2", re.I)


def _resolves(verb: str, sub: str | None) -> bool:
    command = CLI.get_command(click.Context(CLI), verb)  # type: ignore[attr-defined]
    if command is None:
        return False
    if sub is None or not isinstance(command, click.Group):
        return True
    return command.get_command(click.Context(command), sub) is not None


def offences(sentence: Sentence) -> list[str]:
    found = []
    for span in SPAN.findall(sentence.text):
        for verb, sub in VERB.findall(span):
            if not _resolves(verb, sub or None):
                found.append(f"{sentence.where} names `{span}`, which the CLI does not have")
        for flag, instead in RETIRED_FLAGS.items():
            if re.search(rf"(?<![\w-]){re.escape(flag)}(?=[\s=]|$)", span) \
                    and not NAMED_AS_RETIRED.search(sentence.text):
                found.append(f"{sentence.where} teaches `{flag}` in `{span}` -- {instead}")
    return found


def test_every_command_named_in_prose_exists():
    found = [o for s in published_sentences() for o in offences(s)]
    assert not found, "\n".join(found)


def test_the_scan_reads_commands_in_the_corpus():
    """Guards the guard: a span pattern that matched nothing would make the test above pass."""
    named = [s for s in published_sentences()
             if any(VERB.search(span) for span in SPAN.findall(s.text))]
    assert len(named) >= 10, f"only {len(named)} sentence(s) name a command"


def test_the_scan_catches_the_commands_that_shipped_wrong():
    def caught(text: str) -> int:
        return sum(len(offences(s)) for s in sentences_of("d.md", text))

    assert caught("`memrank list-adapters` does not offer it.") == 1
    assert caught("A target only supports `--adapter myengine`.") == 1
    assert caught("`--adapter` is retired; use a target.") == 0
    assert caught("`memrank runs ls` lists them, and `memrank submit tfidf squad` runs.") == 0
    assert caught("`memrank runs lss` lists them.") == 1
