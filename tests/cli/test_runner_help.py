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
"""Smoke tests for the CLI surface (no live backends).

The two assertions about the OPERATOR entrypoint -- that `memrank --help` leaks none of
`memrank.ops.OPS_COMMANDS`, and that every one of them still has a home on `memrank-ops` -- moved
to `tests/internal/test_runner_help_ops.py`. Both are about a public guarantee, but neither can be
stated without the internal command table, and `memrank.ops` is not in a public tree.
"""

from __future__ import annotations

import re

import pytest
import typer
from typer.testing import CliRunner

from memrank.runner import app

runner = CliRunner()

#: The noun-spaces the interface model says the CLI is made of.
DIMENSIONS = ("agents", "evals", "runs", "auth", "secrets")

#: The engine-hosting path, hidden from help and still runnable (ATO-2343): (group, command).
#: `task-exec` is the internal executor `submit` left behind (ATO-2366).
HIDDEN = (((), "task-exec"), ((), "targets"), ((), "ps"), ((), "logs"), ((), "watch"),
          ((), "kill"), (("runs",), "logs"), (("runs",), "watch"), (("runs",), "kill"),
          (("runs",), "sync"))


def test_help_exits_zero():
    """``memrank --help`` should exit cleanly and mention the subcommands."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "memrank" in result.stdout.lower()
    assert "run" in result.stdout


def test_help_shows_every_dimension():
    """The model's noun-spaces are the surface; each must be reachable from the root."""
    listed = runner.invoke(app, ["--help"]).stdout
    missing = [name for name in DIMENSIONS if name not in listed]
    assert missing == [], f"dimensions absent from the CLI: {missing}"


def _listed(group: tuple[str, ...]) -> set[str]:
    """The command names a group's help lists, one per row of its Commands box."""
    rows = runner.invoke(app, [*group, "--help"]).stdout.split("Commands")[-1].splitlines()
    border = "\u2502 "  # the Commands box's left edge, escaped for the ASCII gate
    return {row.strip(border).split(" ")[0] for row in rows if row.startswith(border)}


@pytest.mark.parametrize("group,command", HIDDEN)
def test_the_engine_hosting_path_is_hidden_but_still_runs(group, command):
    """A new user meets `memrank run --agent` only; the old path answers when typed."""
    assert command not in _listed(group)
    assert runner.invoke(app, [*group, command, "--help"]).exit_code == 0


def test_the_hidden_scan_sees_a_listed_command():
    """Guards the guard: a scan that finds nothing would pass every hidden command."""
    assert {"run", "agents", "runs"} <= _listed(())
    assert {"ls", "show"} <= _listed(("runs",))


def test_version_prints_semver():
    """``memrank version`` should print a non-empty version string."""
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip()
    assert "." in result.stdout


def test_the_version_flag_and_the_version_command_cannot_disagree():
    """Both forms exist because both get typed; one implementation, so they say one thing."""
    for form in (["--version"], ["-v"], ["-V"]):
        result = runner.invoke(app, form)
        assert result.exit_code == 0, form
        assert result.stdout == runner.invoke(app, ["version"]).stdout, form


def test_help_lists_the_short_version_flag_beside_the_long_one():
    """`-v` is discoverable where `--version` is (ATO-2696)."""
    result = runner.invoke(app, ["--help"], terminal_width=200)
    assert result.exit_code == 0
    version_rows = [line for line in result.output.splitlines() if "--version" in line]
    assert len(version_rows) == 1 and "-v," in version_rows[0], result.output


def test_a_subcommand_keeps_its_own_short_v_as_verbose():
    """The root `-v` is parsed before the command only; after it, `-v` is still --verbose."""
    root = typer.main.get_command(app)
    task_exec = root.commands["task-exec"]  # type: ignore[attr-defined]
    owners = [param.name for param in task_exec.params if "-v" in param.opts]
    assert owners == ["verbose"]


def test_the_root_callback_leaves_no_args_showing_help():
    """A callback added for `--version` must not turn a bare `memrank` into a silent exit."""
    result = runner.invoke(app, [])

    assert "Usage:" in result.output
    assert "Commands" in result.output


def test_run_names_its_judge_as_provider_model_and_the_old_flag_is_gone():
    """`--judge-model` broke with no alias (ATO-2503): the judge is one provider:model spec."""
    result = runner.invoke(app, ["run", "--help"], terminal_width=200)
    assert result.exit_code == 0
    assert "--judge" in result.output and "SPEC" in result.output
    assert "anthropic:claude-haiku-4-5" in result.output  # the default, shown
    assert "--judge-model" not in result.output
    refused = runner.invoke(app, ["run", "locomo", "--agent", "full-context",
                                  "--judge-model", "claude-haiku-4-5"])
    assert refused.exit_code != 0


def _help_words() -> str:
    result = runner.invoke(app, ["run", "--help"], terminal_width=200)
    assert result.exit_code == 0
    # Rich wraps the help in a box; read it as one line of words.
    return re.sub(r"[\u2502\s]+", " ", result.output)


@pytest.mark.parametrize("option", ["--concurrency", "--ingest-concurrency",
                                    "--answer-concurrency", "--judge-concurrency",
                                    "--questions-in-order"])
def test_every_concurrency_option_says_at_most_what(option):
    words = _help_words()
    assert option in words
    # the option, its short form and metavar if any, its range, then its help
    assert re.search(re.escape(option) + r"(?: -j)?(?: N)?(?: \[x>=1\])? At most", words), option


def test_each_stage_ceiling_shows_its_default_and_the_shorthand():
    words = _help_words()
    for option in ("--ingest-concurrency", "--answer-concurrency", "--judge-concurrency"):
        assert "[default: (8, or --concurrency)]" in words.split(option, 1)[1], option


def test_judge_concurrency_with_no_judge_is_refused_before_anything_runs():
    refused = runner.invoke(app, ["run", "locomo", "--agent", "full-context", "--no-judge",
                                  "--judge-concurrency", "5"])
    assert refused.exit_code == 2
    assert "--no-judge makes none" in refused.output


def test_an_answer_concurrency_below_one_is_refused():
    refused = runner.invoke(app, ["run", "locomo", "--agent", "full-context",
                                  "--answer-concurrency", "0"])
    assert refused.exit_code == 2
