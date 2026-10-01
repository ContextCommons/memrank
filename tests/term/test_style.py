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
"""The style chokepoint: color enters padded, leaves stripped, and never touches data.

Under pytest/CliRunner the streams are not TTYs, so Click strips escape codes at the echo
-- the same mechanism that keeps piped output clean in the shell. Tests that pass
``CliRunner.invoke(color=True)`` simulate a TTY, proving styling actually reaches a
terminal -- and that ``NO_COLOR`` still wins there.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import click
import pytest
from typer.testing import CliRunner

from memrank.runner import app
from memrank.runs import registry
from memrank.term import style

runner = CliRunner()


@pytest.fixture
def runs_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMRANK_RUNS_DIR", str(tmp_path))
    return tmp_path


def _heartbeat(run_dir, *, state="done"):
    """A minimal terminal-state heartbeat, enough for `runs ls` to list one row."""
    payload = {"run_id": run_dir.name, "pid": None, "target": "word-overlap", "benchmark": "demo",
               "slice": None, "state": state, "progress": {}, "message": "",
               "started_at": "2026-01-01T00:00:00",
               "updated_at": datetime.now(timezone.utc).isoformat(), "error": None}
    (run_dir / "status.json").write_text(json.dumps(payload))


def test_state_pads_before_styling():
    """The ANSI codes wrap the padded cell, so `len()`-sized columns stay aligned."""
    styled = style.state("failed", width=9)
    assert "\x1b[31m" in styled
    assert click.unstyle(styled) == "failed   "


def test_unknown_state_passes_through_unstyled():
    """An unmapped state carries no color claim -- but still gets its padding."""
    assert style.state("unknown", width=9) == "unknown  "
    assert "\x1b" not in style.state("—")


def test_every_state_color_is_a_valid_click_color():
    for value, color in style._STATE_COLORS.items():
        assert "\x1b[" in click.style(value, fg=color)


def test_error_keeps_prefix_and_always_uses_stderr(capsys):
    """`error:` stays greppable, and no caller can route a failure onto stdout.

    The escape hatch that allowed it is gone: a surface that "owns stdout errors" is a
    surface whose failures land in the file a caller was capturing an answer into.
    """
    style.error("boom")
    captured = capsys.readouterr()
    assert captured.err == "error: boom\n"
    assert captured.out == ""


def test_warn_and_note_narrate_on_stderr(capsys):
    style.warn("careful")
    style.note("fyi")
    captured = capsys.readouterr()
    assert captured.err == "warning: careful\nnote: fyi\n"
    assert captured.out == ""


def test_listing_is_plain_when_not_a_tty(runs_dir):
    """Piped output is byte-clean: no escape code survives a non-TTY stream."""
    _heartbeat(registry.new_run_dir("demo"))
    result = runner.invoke(app, ["runs", "ls"])
    assert result.exit_code == 0
    assert "\x1b" not in result.output


def test_forced_color_reaches_the_table(runs_dir):
    """On a terminal the state cell really is colored -- green for a done run."""
    _heartbeat(registry.new_run_dir("demo"))
    result = runner.invoke(app, ["runs", "ls"], color=True)
    assert result.exit_code == 0
    assert "\x1b[32m" in result.output


def test_no_color_disables_styling_even_on_a_tty(runs_dir, monkeypatch):
    """The NO_COLOR convention closes the chokepoint before Click ever sees a code."""
    monkeypatch.setenv("NO_COLOR", "1")
    _heartbeat(registry.new_run_dir("demo"))
    result = runner.invoke(app, ["runs", "ls"], color=True)
    assert result.exit_code == 0
    assert "\x1b" not in result.output
    assert style.state("failed", width=9) == "failed   "


def test_the_gate_is_public_because_more_than_colour_asks_it(monkeypatch):
    """`watch`'s redraw emits cursor control, which NO_COLOR silences by the same argument --
    so `term.progress` asks the gate rather than re-reading the environment beside it."""
    from memrank.term import progress

    class _Tty:
        def isatty(self):
            return True

    assert progress.supports_redraw(_Tty()) is True
    monkeypatch.setenv("NO_COLOR", "1")
    assert progress.supports_redraw(_Tty()) is False
    assert style.wants_color() is False


# --- the vocabulary, and the padding rule it depends on ------------------------------------------ #

#: Every helper that emits colour. ENUMERATED so a new one cannot quietly skip the NO_COLOR gate --
#: the failure would be invisible until someone with NO_COLOR set found escapes in their log.
COLOURING = ("good", "bad", "caution", "dim", "bold", "heading", "label", "accent", "unit")


@pytest.mark.parametrize("name", COLOURING)
def test_no_color_silences_every_helper(name, monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")

    assert getattr(style, name)("text") == "text"


@pytest.mark.parametrize("name", COLOURING)
def test_styling_never_changes_what_is_visible(name):
    """Colour is presentation. The characters a reader sees are the characters passed in."""
    assert click.unstyle(getattr(style, name)("text")) == "text"


def test_value_is_deliberately_unstyled():
    """Everything AROUND a measured value is dimmed or coloured, so plain is the emphasis -- and it
    stays readable on light and dark themes, which a fixed bright colour would not."""
    assert style.value("0.269737") == "0.269737"


def test_pad_pads_before_styling():
    """The bug this function makes impossible. `f"{style.dim(x):<9}"` pads a string whose length
    already counts its escape codes, so it adds nothing and the column collapses -- while every
    substring assertion still passes, because all the characters are present."""
    padded = style.pad("ab", 9, style.dim)

    assert click.unstyle(padded) == "ab       "


def test_pad_right_aligns_when_asked():
    """Right-aligned columns are the ones most likely to be re-padded by hand afterwards -- the
    same bug wearing a different hat, and it was made in this function's first caller."""
    assert click.unstyle(style.pad("7", 5, style.dim, align=">")) == "    7"


def test_the_listing_columns_line_up_whatever_the_state(runs_dir, monkeypatch):
    """The whole table, measured by VISIBLE width. Styling a padded cell shifts every column to
    its right, and no grep-style assertion in this suite would notice."""
    for state in ("done", "failed", "running"):
        _heartbeat(registry.new_run_dir("demo"), state=state)

    result = runner.invoke(app, ["runs", "ls"], color=True)

    rows = [click.unstyle(line) for line in result.output.splitlines() if line.strip()]
    # The header is FOUND, not assumed to be first: a `note:` advisory can precede it, and an
    # off-by-one here would silently measure the note against the rows.
    head = next(i for i, row in enumerate(rows) if row.startswith("ID "))
    header, data = rows[head], rows[head + 1:]
    # Anchored on the HEADER's offsets rather than on a substring: run ids contain the benchmark
    # name too, so searching for it finds the wrong column and passes for the wrong reason.
    for column, expected in (("PLACE", ("local",)),
                             ("STATE", ("done", "failed", "running"))):
        start = header.index(column)
        # Checked at the OFFSET rather than by slicing a fixed width: a column is wider than its
        # heading (STATE is 9 for `running`), and slicing len("STATE") would compare "runni".
        for row in data:
            assert row[start:].startswith(expected), (
                f"{column} does not start at column {start}: {row[start:start + 12]!r}\n{row}")
