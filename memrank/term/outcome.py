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
"""The one renderer for how a command ends (:class:`memrank.outcome.Outcome`).

Every line is built once, as plain words tagged with what each part IS (a mark, a heading, a
command, a link, detail). Decoration is applied only at the end, and only when the stream is a
terminal that wants it -- so a pipe, ``NO_COLOR`` or CI gets exactly the same words with nothing
around them. A finished run's ending is the command's answer and goes to stdout; an ending that
asks something of the reader goes to stderr with the rest of the narration.
"""

from __future__ import annotations

import sys

import typer

from memrank.outcome import Kind, Outcome, Step
from memrank.term import style

#: What each part of a line is; the styling each one gets lives in :func:`_decorate`.
Part = tuple[str, str | None]
Line = list[Part]

#: The mark each look leads with. The check is the one character here that is not ASCII.
CHECK, BANG, CROSS = "✓", "!", "x"
_MARKS = {Kind.DONE: (CHECK, "good"), Kind.ACTION: (BANG, "caution"),
          Kind.BROKEN: (CROSS, "bad")}
_INDENT = "  "


def _step(number: int | None, step: Step) -> list[Line]:
    lead = f"{_INDENT}{number}. " if number is not None else _INDENT
    rows: list[Line] = [[(lead + step.text, None)]] if step.text else []
    pad = " " * (len(lead) + 2)
    return rows + [[(pad, None), (command, "command")] for command in step.commands]


def _steps(steps: tuple[Step, ...]) -> list[Line]:
    rows: list[Line] = [[("What to do", "heading")]]
    numbered = len(steps) > 1
    for index, step in enumerate(steps, start=1):
        rows += _step(index if numbered else None, step)
    return rows


def _paragraph(text: str, role: str | None = None, indent: str = "") -> list[Line]:
    return [[(indent + row, role)] for row in text.splitlines()]


def _head(outcome: Outcome) -> list[Line]:
    rows: list[Line] = []
    if outcome.achieved:
        rows.append([(CHECK, "good"), (f" {outcome.achieved}", None)])
    mark, role = _MARKS[outcome.kind]
    rows.append([(mark, role), (f" {outcome.title}", None)])
    tagged = [(name, value, "identity") for name, value in outcome.identity] + \
        [(name, value, None) for name, value in outcome.facts]
    width = max((len(name) for name, _, _ in tagged), default=0)
    rows += [[(f"{_INDENT}{name:<{width}}  ", "label"), (value, role)]
             for name, value, role in tagged]
    return rows


def _where(outcome: Outcome) -> list[Line]:
    rows: list[Line] = []
    if outcome.saved:
        rows.append([("Saved locally: ", None), (outcome.saved, None)])
    if outcome.link:
        rows.append([(f"{outcome.link.label}: ", None), (outcome.link.url, "link")])
    return rows


def lines(outcome: Outcome) -> list[Line]:
    """The outcome as lines of tagged parts, in the order they are shown."""
    sections: list[list[Line]] = [_head(outcome)]
    if outcome.happened:
        title: Line = [("What happened", "heading")]
        sections.append([title, *_paragraph(outcome.happened, indent=_INDENT)])
    if outcome.steps:
        sections.append(_steps(outcome.steps))
    if outcome.note:
        sections.append(_paragraph(f"Note: {outcome.note}"))
    where = _where(outcome)
    if where and outcome.kind is Kind.DONE and len(sections) == 1:
        sections[0] += where  # a clean finish reads as one block, its link right under it
    elif where:
        sections.append(where)
    if outcome.details:
        source: Line = [(f"Details from {outcome.details.source}:", "dim")]
        sections.append([source, *_paragraph("\n".join(outcome.details.lines), "dim",
                                              _INDENT)])
    rows: list[Line] = []
    for number, section in enumerate(sections):
        if number:
            rows.append([("", None)])
        rows += section
    return rows


def plain(outcome: Outcome) -> str:
    """The outcome exactly as a pipe sees it."""
    return "\n".join("".join(text for text, _ in line) for line in lines(outcome))


#: How each tagged part looks on a terminal; an untagged part is left as it is.
_LOOKS: dict[str, dict[str, object]] = {
    "good": {"fg": typer.colors.GREEN, "bold": True},
    "caution": {"fg": typer.colors.YELLOW, "bold": True},
    "bad": {"fg": typer.colors.RED, "bold": True},
    "heading": {"bold": True},
    "command": {"bold": True},
    "label": {"dim": True},
    "dim": {"dim": True},
    "identity": style.IDENTITY_LOOK,
}


def _decorate(text: str, role: str | None, stream: object) -> str:
    if role == "link":
        return style.link(text, stream)
    look = _LOOKS.get(role or "")
    return typer.style(text, **look) if look and text else text  # type: ignore[arg-type]


def render(outcome: Outcome) -> None:
    """Print the outcome: a finished run to stdout, anything else to stderr."""
    err = outcome.kind is not Kind.DONE
    stream = sys.stderr if err else sys.stdout
    # Anything already written to stdout is flushed first, so the ending stays last in a log
    # that captures both streams.
    sys.stdout.flush()
    decorated = style.decorates(stream)
    for line in lines(outcome):
        text = "".join(_decorate(part, role, stream) if decorated else part for part, role in line)
        (style.say if err else style.out)(text)
