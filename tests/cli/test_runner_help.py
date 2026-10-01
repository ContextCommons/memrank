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

import pytest
from typer.testing import CliRunner

from memrank.runner import app

runner = CliRunner()

#: The noun-spaces the interface model says the CLI is made of.
DIMENSIONS = ("agents", "evals", "runs", "auth", "secrets")

#: The engine-hosting path, hidden from help and still runnable (ATO-2343): (group, command).
HIDDEN = (((), "submit"), ((), "targets"), ((), "ps"), ((), "logs"), ((), "watch"),
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
    for form in (["--version"], ["-V"]):
        result = runner.invoke(app, form)
        assert result.exit_code == 0, form
        assert result.stdout == runner.invoke(app, ["version"]).stdout, form


def test_the_root_callback_leaves_no_args_showing_help():
    """A callback added for `--version` must not turn a bare `memrank` into a silent exit."""
    result = runner.invoke(app, [])

    assert "Usage:" in result.output
    assert "Commands" in result.output


def test_run_judge_needs_no_second_flag_for_real_data(monkeypatch):
    """`--judge` on a non-synthetic benchmark used to be refused without `--ack-egress`.

    The gate is retired -- `--judge` already names the provider it sends to, and the browser path
    never asked -- so the command proceeds past the point that used to refuse it. It still fails
    here, on the engine that is not running, which is what proves the egress check is gone rather
    than merely quiet.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    result = runner.invoke(app, ["submit", "word-overlap", "locomo:smoke", "--judge", "--on", "none"])
    assert "egress" not in result.output.lower() or "sends retrieved content" in result.output
