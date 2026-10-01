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
"""The hard cutover, enumerated.

Every assertion walks ``RETIRED`` / ``RETIRED_FLAGS`` rather than naming commands, so a
retirement added to a table is covered the moment it lands -- and one that ships without a
working replacement fails here instead of in a user's terminal.
"""
from __future__ import annotations

import pytest
from typer.testing import CliRunner

from memrank.cli.retired import RETIRED, RETIRED_EXIT_CODE, RETIRED_FLAGS
from memrank.runner import app

runner = CliRunner()


@pytest.mark.parametrize("old", sorted(RETIRED))
def test_retired_name_names_its_replacement(old):
    """An old name must fail loudly and say what to type instead."""
    result = runner.invoke(app, [*old.split()])
    assert result.exit_code == RETIRED_EXIT_CODE
    assert RETIRED[old] in result.output


@pytest.mark.parametrize("old", sorted(RETIRED))
def test_retired_name_survives_its_old_arguments(old):
    """A whole old command line reaches the message, not a parser error about its flags."""
    result = runner.invoke(app, [*old.split(), "some-target", "--slice", "smoke"])
    assert result.exit_code == RETIRED_EXIT_CODE
    assert RETIRED[old] in result.output


def _command_names(help_text: str) -> set[str]:
    """The command NAMES in a help listing, not the prose describing them.

    Matching the whole output cannot tell a command called `run` from the word "run" in a
    description -- and it did exactly that once `status` began showing `runs show`'s docstring,
    "Show one run in full". A retired name is a name; look only where names live.
    """
    names: set[str] = set()
    for line in help_text.splitlines():
        stripped = line.strip().strip("│").strip()
        first = stripped.split(" ", 1)[0]
        if first and first.replace("-", "").isalnum():
            names.add(first)
    return names


def test_retired_names_are_absent_from_help():
    """`memrank --help` shows the surface as it is, not a museum of what it was."""
    listed = _command_names(runner.invoke(app, ["--help"]).output)
    leaked = sorted(old for old in RETIRED if old in listed)
    assert leaked == [], f"retired names visible in help: {leaked}"


def test_the_help_scan_would_catch_a_name_that_really_leaked():
    """Guards the guard: a laxer scan passes everything, which is worse than no test."""
    assert "agents" in _command_names(runner.invoke(app, ["--help"]).output)


@pytest.mark.parametrize("old,new", sorted(RETIRED.items()))
def test_replacement_actually_exists(old, new):
    """The pointer must point somewhere: every replacement resolves to a real command."""
    result = runner.invoke(app, [*new.split(), "--help"])
    assert result.exit_code == 0, f"`memrank {new}` (replacing {old}) does not resolve"


# --- and the same rule for flags ----------------------------------------------------------------- #

#: A value each retired flag accepts. Switches take none; the rest take any string, since the
#: refusal happens before anything is resolved -- except a flag Click TYPES, which must parse
#: before `submit` gets to refuse it at all.
_FLAG_ARGS = {"--all-adapters": [], "--max-judge-calls": ["760"]}


@pytest.mark.parametrize("flag", sorted(RETIRED_FLAGS))
def test_retired_flag_names_its_replacement(flag):
    """A retired flag says what to type instead, rather than dying on Click's "No such option"."""
    result = runner.invoke(app, ["submit", flag, *_FLAG_ARGS.get(flag, ["word-overlap"])])

    assert result.exit_code == RETIRED_EXIT_CODE
    assert flag in result.output and "retired" in result.output


@pytest.mark.parametrize("flag", sorted(RETIRED_FLAGS))
def test_a_retired_flag_is_hidden_from_help(flag):
    """`submit --help` shows the surface as it is, not a museum of what it was."""
    assert flag not in runner.invoke(app, ["submit", "--help"]).output


def test_the_help_scan_would_catch_a_flag_that_really_leaked():
    """Guards the guard, as above: the surviving flags ARE listed.

    Was `--slice` until it was retired in favour of the eval ref -- which is exactly the drift
    this control exists to notice, so it now watches a flag with no plans to leave."""
    assert "--workers" in runner.invoke(app, ["submit", "--help"]).output
