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
"""The label/value block a `show` view is made of, and the rules that separate its sections.

    ╭─ hindsight:matched ────────────────────────╮
    │   kind      system                         │
    │   adapter   mem0                           │
    │   depends   voyage-3, qdrant               │
    ╰────────────────────────────────────────────╯

      ── secrets ──────────────────────────────────
      ✔ OPENAI_API_KEY
      ✘ VOYAGE_API_KEY

Every `show` command had grown its own version of this -- `runs show` with a 9-wide padded
label, `evals show` with the width typed into each f-string, `targets show` with a third
answer. Three implementations of "line up these pairs" drift on the first label longer than
the guess, and the one that drifts is whichever was written first.

The label width comes from the LABELS, not from a constant, so adding a field cannot silently
unalign the block above it. Labels are dimmed and values are left at the terminal's default
foreground, which is `style.value`'s standing argument: everything around a value is dimmed or
coloured, so plain IS the emphasis.

Like `table`, this renders boxes only when `style.supports_boxes()` says stdout is a terminal,
and returns lines rather than printing them -- `style.out` stays the only emitter.
"""
from __future__ import annotations

import io
from collections.abc import Sequence

from memrank.term import style

#: A label/value pair. The value may already be styled; it is last on its line, so nothing is
#: padded after it and the usual pad-before-style trap does not apply.
Field = tuple[str, str]


def _pairs(fields: Sequence[Field], indent: int) -> list[str]:
    """The pairs themselves, one per line, labels sized to the longest of them."""
    if not fields:
        return []
    width = max(len(label) for label, _ in fields)
    pad = " " * indent
    return [f"{pad}{style.pad(label, width, style.label)}  {value}" for label, value in fields]


def fields(pairs: Sequence[Field], *, indent: int = 2) -> list[str]:
    """A block of label/value lines. The plain form, used inside and outside a panel alike."""
    return _pairs(pairs, indent)


def rule(title: str, *, indent: int = 0, width: int = 0) -> list[str]:
    """``── secrets ────────────`` on a terminal, a bare bold ``secrets`` on a pipe.

    A section heading that a pipe carries as a word rather than as a run of dashes: the dashes
    are there to give the eye an edge to follow, and a `grep` for the section name should find
    the name, not have to skip a hyphen fence.
    """
    if not style.supports_boxes():
        return [f"{' ' * indent}{style.heading(title)}"]
    span = width or min(style.terminal_width() - indent, 72)
    tail = max(0, span - len(title) - 5)
    return [f"{' ' * indent}{style.dim('──')} {style.heading(title)} {style.dim('─' * tail)}"]


def panel(title: str, pairs: Sequence[Field], *, indent: int = 0) -> list[str]:
    """A titled box around a block of pairs -- the header of a `show` view.

    Off a terminal this degrades to the heading and the pairs, which is what the view printed
    before boxes existed, so a redirected `runs show` is unchanged.
    """
    body = _pairs(pairs, indent=0)
    if not style.supports_boxes():
        return [f"{' ' * indent}{style.heading(title)}",
                *(f"{' ' * (indent + 2)}{line}" for line in body)]

    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text

    coloured = style.wants_color()
    content = Text("\n").join(Text.from_ansi(line) for line in body)
    box = Panel(content, title=Text(title, style="bold not dim" if coloured else "not dim"),
                title_align="left",
                border_style="dim" if coloured else "none",
                padding=(0, 2), expand=False)
    buffer = io.StringIO()
    console = Console(file=buffer, width=style.terminal_width() - indent,
                      force_terminal=coloured, no_color=not coloured,
                      highlight=False, soft_wrap=False)
    console.print(box)
    rendered = buffer.getvalue().rstrip("\n").splitlines()
    return [f"{' ' * indent}{line}" for line in rendered]


def emit(lines: Sequence[str]) -> None:
    """Write a rendered block to stdout -- a `show` view's fields ARE its answer."""
    for line in lines:
        style.out(line)
