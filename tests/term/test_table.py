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
"""One listing, two renderings -- and the rule that decides which.

A border is a human affordance: 12-Factor #8 bans it because a piped listing must stay one
plain row per entry, and that reason is about the destination, not about tables. So the tests
here come in pairs -- what a terminal gets, and what a pipe gets -- and the pipe's half is the
one that protects every other assertion in this suite.
"""
from __future__ import annotations

import click
import pytest

from memrank.term import style, table

COLUMNS = (
    table.Column("ID", styler=style.unit),
    table.Column("TARGET", styler=style.accent),
    table.Column("STATE"),
    table.Column("AGE", align=">"),
)

#: The same listing with a truncatable handle -- `runs ls`'s real shape, and the one that
#: exercises the width solver.
ELLIPSIS_COLUMNS = (table.Column("ID", styler=style.unit, ellipsis=True), *COLUMNS[1:])

ROWS = [
    ["20260819-aaa__locomo__x", "mem0:voyage",
     table.Cell("done", style.state_styler("done")), "12m"],
    ["20260818-b__beam__y", "hindsight",
     table.Cell("running", style.state_styler("running")), "2h"],
]


@pytest.fixture
def boxed(monkeypatch):
    """A terminal: stdout is a TTY, so the listing is a table."""
    monkeypatch.setattr(style, "supports_boxes", lambda: True)
    monkeypatch.setattr(style, "terminal_width", lambda default=100: 80)


@pytest.fixture
def piped(monkeypatch):
    """A pipe: no box, no title, no truncation."""
    monkeypatch.setattr(style, "supports_boxes", lambda: False)


def visible(lines: list[str]) -> list[str]:
    return [click.unstyle(line) for line in lines]


# --- the piped rendering: the contract every other test in the suite rests on ------------------- #

def test_a_pipe_gets_plain_rows_with_no_border(piped):
    lines = visible(table.render(COLUMNS, ROWS, title="Runs"))

    assert not any(char in "".join(lines) for char in "┏┃│└─╇"), lines
    assert lines[0].startswith("ID ")
    assert len(lines) == 1 + len(ROWS)


def test_the_title_is_a_terminal_affordance_and_never_reaches_a_pipe(piped):
    """A heading a script did not ask for is a row it has to learn to skip."""
    assert "Runs" not in "".join(table.render(COLUMNS, ROWS, title="Runs"))


def test_every_column_starts_where_its_heading_does(piped):
    """The whole point of the layer: alignment measured by VISIBLE width, not by `len()`.

    A styled cell padded the wrong way round keeps all its characters and puts them in the wrong
    place, so a substring assertion passes while the table is visibly broken.
    """
    header, *rows = visible(table.render(COLUMNS, ROWS))

    for label, expected in (("TARGET", ("mem0:voyage", "hindsight")), ("STATE", ("done", "running"))):
        start = header.index(label)
        for row in rows:
            assert row[start:].startswith(expected), f"{label} misaligned: {row!r}"


def test_a_pipe_truncates_nothing_however_narrow_the_terminal(piped, monkeypatch):
    """`ellipsis` is a reading aid. A script copying a run id must get the whole handle."""
    monkeypatch.setattr(style, "terminal_width", lambda default=100: 20)
    columns = (table.Column("ID", ellipsis=True),)
    long_id = "20260819-aaaaaaaaaaaaaaaaaaaaaaaa__locomo__deadbeef"

    assert long_id in "\n".join(table.render(columns, [[long_id]]))


def test_nothing_trails_whitespace_on_the_piped_path(piped):
    """Padding the last column would put invisible bytes in a file someone diffs."""
    for line in table.render(COLUMNS, ROWS):
        assert click.unstyle(line) == click.unstyle(line).rstrip()


# --- the boxed rendering ------------------------------------------------------------------------ #

def test_a_terminal_gets_a_bordered_titled_table(boxed):
    joined = "\n".join(visible(table.render(COLUMNS, ROWS, title="Runs")))

    assert joined.count("┏") == 1 and joined.count("└") == 1     # one box, top and bottom
    assert "Runs" in joined
    for row in ROWS:
        assert row[1] in joined


def test_the_boxed_path_truncates_a_column_that_asked_to(boxed, monkeypatch):
    monkeypatch.setattr(style, "terminal_width", lambda default=100: 40)
    columns = (table.Column("ID", ellipsis=True), table.Column("NOTE"))
    long_id = "20260819-aaaaaaaaaaaaaaaaaaaaaaaa__locomo__deadbeef"

    joined = "\n".join(visible(table.render(columns, [[long_id, "a note"]])))
    assert long_id not in joined and "…" in joined


def test_a_boxed_row_is_never_wider_than_the_terminal(boxed, monkeypatch):
    """A box that overflows wraps into rubble; the width is the one thing it must respect."""
    monkeypatch.setattr(style, "terminal_width", lambda default=100: 46)
    columns = (table.Column("A", ellipsis=True), table.Column("B", ellipsis=True))

    for line in visible(table.render(columns, [["x" * 90, "y" * 90]])):
        assert len(line) <= 46, repr(line)


def test_a_truncatable_column_yields_before_the_table_gives_up(boxed, monkeypatch):
    """The narrow-terminal bug, as a test.

    `no_wrap` is RIGID in rich's width solver, so an uncapped 40-character run id claimed half
    an 80-column terminal and left `STATE` one character wide -- a box full of vertical letters.
    The handle must be what gives way, because the reader is looking at the state.
    """
    monkeypatch.setattr(style, "terminal_width", lambda default=100: 58)
    long_id = "supermemory-locomo-20260727-064017-44505"

    lines = visible(table.render(ELLIPSIS_COLUMNS, [[long_id, "hindsight", "running", "2h"]]))
    body = next(line for line in lines if "hindsight" in line)
    assert "…" in body                      # the id was cut
    assert "running" in body                # ...so the state survived whole


def test_a_table_too_wide_even_when_squeezed_falls_back_to_plain_rows(boxed, monkeypatch):
    """A box the screen cannot hold wraps into one character per column, which is worse than
    the rows it replaced. Degrading is the honest answer; rendering rubble is not."""
    monkeypatch.setattr(style, "terminal_width", lambda default=100: 26)

    lines = visible(table.render(ELLIPSIS_COLUMNS, ROWS, title="Runs"))
    assert not any(char in "".join(lines) for char in "┏│└")
    assert lines[0].startswith("ID ")


def test_a_wide_terminal_keeps_the_handle_whole(boxed, monkeypatch):
    """Truncation is a response to a narrow screen, not a habit."""
    monkeypatch.setattr(style, "terminal_width", lambda default=100: 140)
    long_id = "supermemory-locomo-20260727-064017-44505"

    assert long_id in "\n".join(
        visible(table.render(ELLIPSIS_COLUMNS, [[long_id, "h", "running", "2h"]])))


# --- colour, and the gate that silences it ------------------------------------------------------ #

def test_a_pre_styled_cell_keeps_its_colour_through_the_box(boxed):
    """`Text.from_ansi` is the bridge: a caller styles with `style.*` and both paths obey it."""
    joined = "\n".join(table.render(COLUMNS, ROWS))

    assert "\x1b[32m" in joined      # done, green
    assert "\x1b[36m" in joined      # running, cyan


def test_no_color_leaves_the_box_but_takes_the_colour(boxed, monkeypatch):
    """NO_COLOR is about colour. A monochrome bordered table is still a correct rendering."""
    monkeypatch.setenv("NO_COLOR", "1")

    lines = table.render(COLUMNS, ROWS, title="Runs")
    joined = "\n".join(lines)
    assert "\x1b[" not in joined, repr(joined)
    assert "┏" in joined and "Runs" in joined


def test_a_terminal_that_cannot_draw_gets_no_box(monkeypatch):
    """`TERM=dumb` is a TTY whose glyphs would come out as mojibake."""
    monkeypatch.setattr("sys.stdout.isatty", lambda: True, raising=False)
    monkeypatch.setenv("TERM", "dumb")

    assert not style.supports_boxes()


def test_a_pipe_is_not_a_terminal_however_colourful(monkeypatch):
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setattr("sys.stdout.isatty", lambda: False, raising=False)

    assert not style.supports_boxes()


# --- the row/column contract -------------------------------------------------------------------- #

def test_a_row_of_the_wrong_length_is_refused(piped):
    """Silently padding a short row puts a value under the wrong heading."""
    with pytest.raises(ValueError, match="2 cells"):
        table.render(COLUMNS, [["only", "two"]])


def test_a_column_grows_past_its_minimum_to_fit_its_widest_cell(piped):
    """A fixed width shifts every column to its right on exactly the row nobody recognises."""
    columns = (table.Column("ID", width=8), table.Column("NEXT"))

    header, row = visible(table.render(columns, [["a-very-long-identifier", "here"]]))
    assert row.index("here") == header.index("NEXT")


def test_a_heading_wider_than_its_data_still_sizes_the_column(piped):
    columns = (table.Column("DESCRIPTION",), table.Column("NEXT"))

    header, row = visible(table.render(columns, [["x", "here"]]))
    assert row.index("here") == header.index("NEXT")


def test_an_empty_listing_renders_its_headings_and_no_rows(piped):
    assert len(table.render(COLUMNS, [])) == 1


def test_a_bare_string_inherits_its_column_styler(piped):
    """So a caller declares a colour once per column rather than once per cell."""
    lines = table.render((table.Column("X", styler=style.accent),), [["value"]])

    assert "\x1b[36m" in lines[1]


def test_a_cell_overrides_the_column_it_sits_in(piped):
    """Run states differ per row; the column cannot know which colour a cell wants."""
    lines = table.render((table.Column("X", styler=style.accent),),
                         [[table.Cell("value", style.bad)]])

    assert "\x1b[31m" in lines[1] and "\x1b[36m" not in lines[1]


def test_emit_puts_the_listing_on_stdout(piped, capsys):
    """A listing IS the answer, so it is stdout -- narration about it is not."""
    table.emit(COLUMNS, ROWS)

    captured = capsys.readouterr()
    assert captured.err == ""
    assert len(captured.out.strip().splitlines()) == 1 + len(ROWS)
