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
"""One listing, rendered two ways -- a bordered table for a terminal, plain rows for a pipe.

    $ memrank secrets ls                    $ memrank secrets ls | cat
                  Secrets                   NAME            VALUE         SOURCE
    ┏━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┓   OPENAI_API_KEY  sk-…4f2       wallet
    ┃ NAME           ┃ VALUE   ┃ SOURCE ┃   VOYAGE_API_KEY  (not stored)  env only
    ┡━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━┩
    │ OPENAI_API_KEY │ sk-…4f2 │ wallet │
    └────────────────┴─────────┴────────┘

WHY BOTH. 12-Factor #8 says never to border a table, and its reason is piping: one plain row
per entry keeps `grep`, `wc` and `cut` working. That reason evaporates when the destination is
a person. So borders are decided at RENDER time, by `style.supports_boxes()`, exactly as `gh`
decides colour and truncation -- the piped bytes are what they always were, and the terminal
gets something worth looking at.

STYLING IS THE EXISTING VOCABULARY. A cell carries an optional `style.*` styler and the
renderer applies it -- on the plain path through `style.pad`, which pads before styling, and on
the boxed path through `Text.from_ansi`, which parses the same escape codes back into rich's
model. So a caller writes `style.accent` or `style.state_styler(...)` once and both paths obey
it; nobody hand-pads a coloured string, which is the alignment bug this package was built
around.

RICH LIVES HERE AND NOWHERE ELSE. It renders into a buffer, never to a stream, so every line
still leaves through `style.out` and the chokepoint that `tests/term/test_output_streams.py`
enforces is untouched. `tests/term/test_output_streams.py` also pins the import itself to this
module, so the dependency cannot spread into command code.
"""
from __future__ import annotations

import io
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import NamedTuple

from memrank.term import style

Styler = Callable[[str], str]


class Cell(NamedTuple):
    """A value and how it should look. Bare strings are accepted anywhere a cell is expected."""

    text: str
    styler: Styler | None = None


CellLike = str | Cell


@dataclass(frozen=True)
class Column:
    """A column's heading and how its cells are laid out.

    `width` is a MINIMUM on the PLAIN path, and ignored on the boxed one. It keeps a piped
    listing from reflowing between invocations; a bordered table has a border doing that job, so
    padding to it there would only waste screen. Either way the column GROWS to its widest cell --
    a fixed width is what pushes every column after it out of line on exactly the row a reader is
    least likely to recognise (`runs ls` learned this from legacy ids).

    `ellipsis` truncates -- on the boxed path only. Truncation is a human affordance, like
    colour and relative timestamps; the piped bytes stay whole so a script still sees the
    complete handle.
    """

    header: str
    align: str = "<"
    width: int = 0
    styler: Styler | None = None
    ellipsis: bool = False


#: What separates two plain columns. One space, matching what `runs ls` has always emitted --
#: the boxed path is where breathing room comes from, and widening the piped rows would only
#: move every offset in the suite for no reader's benefit.
_GAP = " "


def _cells(row: Sequence[CellLike], columns: Sequence[Column]) -> list[Cell]:
    """Normalise a row: a bare string inherits its column's styler, a `Cell` overrides it."""
    if len(row) != len(columns):
        raise ValueError(f"row has {len(row)} cells, table has {len(columns)} columns")
    return [cell if isinstance(cell, Cell) else Cell(cell, column.styler)
            for cell, column in zip(row, columns, strict=True)]


def _widths(rows: Sequence[Sequence[Cell]], columns: Sequence[Column]) -> list[int]:
    """Each column sized to the widest thing in it -- its heading included."""
    return [max(column.width, len(column.header), *(len(row[i].text) for row in rows), 0)
            if rows else max(column.width, len(column.header))
            for i, column in enumerate(columns)]


def _plain(rows: Sequence[Sequence[Cell]], columns: Sequence[Column]) -> list[str]:
    """The pipeable rendering: one row per entry, no borders, nothing truncated.

    The last column is left unpadded so no line carries trailing whitespace -- a diff-visible
    difference that a reader never sees and a test would fight over.
    """
    widths = _widths(rows, columns)
    last = len(columns) - 1

    def line(cells: Sequence[Cell]) -> str:
        parts = []
        for i, (cell, column, width) in enumerate(zip(cells, columns, widths, strict=True)):
            styler = cell.styler or (lambda text: text)
            if i == last and column.align == "<":
                parts.append(styler(cell.text))
            else:
                parts.append(style.pad(cell.text, width, styler, align=column.align))
        return _GAP.join(parts)

    header = line([Cell(column.header, style.heading) for column in columns])
    return [header, *(line(cells) for cells in rows)]


#: How much of the screen one `ellipsis` column may claim, tried in order until the table fits.
#: A `no_wrap` column is RIGID in rich's width solver -- it never yields -- so an uncapped run id
#: took forty of an eighty-column terminal and left every other column one character wide.
#: Capping makes the handle the thing that gives way, which is right: the reader is looking at
#: the state, and the whole id is a `--json` (or a pipe) away.
_ELLIPSIS_CAPS = (0.34, 0.25, 0.18, 0.12)

#: Below this an ellipsis is all a column would be. A table that needs it has too many columns
#: for this terminal, and the plain rows say so honestly.
_MIN_ELLIPSIS = 10


def _fit(rows: Sequence[Sequence[Cell]], columns: Sequence[Column], title: str | None,
         width: int, cap: int) -> list[str] | None:
    """Render at one ellipsis cap, or ``None`` if the table still cannot fit ``width``."""
    from rich import box
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text

    coloured = style.wants_color()
    table = Table(box=box.HEAVY_HEAD,
                  title=title,
                  title_style="bold" if coloured else "none",
                  header_style="bold" if coloured else "none",
                  border_style="dim" if coloured else "none",
                  expand=False)
    for column in columns:
        # `Column.width` is NOT passed on. It is a minimum that exists to stop a plain listing
        # from reflowing between invocations -- a border already separates these cells, so
        # honouring it here would only pad every column out to a width nothing in it needs.
        table.add_column(column.header,
                         justify="right" if column.align == ">" else "left",
                         no_wrap=column.ellipsis,
                         max_width=cap if column.ellipsis else None,
                         overflow="ellipsis" if column.ellipsis else "fold")
    for cells in rows:
        table.add_row(*(Text.from_ansi(cell.styler(cell.text)) if cell.styler
                        else Text(cell.text) for cell in cells))

    buffer = io.StringIO()
    console = Console(file=buffer, width=width,
                      force_terminal=coloured, no_color=not coloured,
                      highlight=False, soft_wrap=False)
    if table.__rich_measure__(console, console.options).minimum > width:
        return None
    console.print(table)
    return buffer.getvalue().rstrip("\n").splitlines()


def _boxed(rows: Sequence[Sequence[Cell]], columns: Sequence[Column],
           title: str | None) -> list[str] | None:
    """The terminal rendering: rich's HEAVY_HEAD box, the one `modal` and friends use.

    Tries progressively tighter caps on the truncatable columns before giving up, because the
    alternative to a slightly shorter run id is no table at all -- and a 100-column terminal
    fits this listing at a 16-character id and not at a 33-character one.

    Returns ``None`` when even the tightest cap overflows. A box the screen is too small for
    wraps into rubble -- one character per column, which the narrow case really did produce --
    and that is strictly worse than the plain rows it replaced, so the caller falls back.
    """
    if not any(column.ellipsis for column in columns):
        return _fit(rows, columns, title, style.terminal_width(), 0)
    width = style.terminal_width()
    for share in _ELLIPSIS_CAPS:
        cap = int(width * share)
        if cap < _MIN_ELLIPSIS:
            break
        rendered = _fit(rows, columns, title, width, cap)
        if rendered is not None:
            return rendered
    return None


def render(columns: Sequence[Column], rows: Sequence[Sequence[CellLike]], *,
           title: str | None = None) -> list[str]:
    """The listing as lines. Boxed when stdout is a terminal, plain otherwise.

    Returned rather than printed so a caller can test the layout without capturing a stream,
    and so `emit` stays the only thing that talks to `style`.
    """
    normalised = [_cells(row, columns) for row in rows]
    if style.supports_boxes():
        boxed = _boxed(normalised, columns, title)
        if boxed is not None:
            return boxed
    return _plain(normalised, columns)


def emit(columns: Sequence[Column], rows: Sequence[Sequence[CellLike]], *,
         title: str | None = None) -> None:
    """Render the listing and write it to stdout -- a listing IS the command's answer."""
    for line in render(columns, rows, title=title):
        style.out(line)
