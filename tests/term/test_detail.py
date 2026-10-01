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
"""The label/value block behind every `show` view.

Three commands had grown three answers to "line these pairs up", each with the width typed
into the f-strings. What is tested here is the property that made them drift: the width comes
from the LABELS, so adding a field cannot leave the block below it one column out.
"""
from __future__ import annotations

import click
import pytest

from memrank.term import detail, style

FIELDS = [("kind", "system"), ("adapter", "mem0"), ("depends", "voyage-3, qdrant")]


@pytest.fixture
def boxed(monkeypatch):
    monkeypatch.setattr(style, "supports_boxes", lambda: True)
    monkeypatch.setattr(style, "terminal_width", lambda default=100: 80)


@pytest.fixture
def piped(monkeypatch):
    monkeypatch.setattr(style, "supports_boxes", lambda: False)


def visible(lines: list[str]) -> list[str]:
    return [click.unstyle(line) for line in lines]


# --- the pairs themselves ------------------------------------------------------------------------ #

def test_values_line_up_because_the_width_comes_from_the_labels():
    """The bug this replaced: a label longer than the hardcoded guess pushed one value out."""
    lines = visible(detail.fields(FIELDS))

    starts = {line.index(value) for (_, value), line in zip(FIELDS, lines, strict=True)}
    assert len(starts) == 1, lines


def test_adding_a_longer_label_moves_every_value_together():
    """Not just the new line -- the whole block, which is what "sized from the labels" means."""
    lines = visible(detail.fields([*FIELDS, ("a-much-longer-label", "x")]))

    starts = {line.rindex(value) for (_, value), line
              in zip([*FIELDS, ("a-much-longer-label", "x")], lines, strict=True)}
    assert len(starts) == 1, lines


def test_a_label_is_dimmed_and_its_value_is_not():
    """`style.value`'s standing argument: everything around a value is quieter, so plain leads."""
    line = detail.fields([("kind", "system")])[0]

    assert "\x1b[2m" in line
    assert line.endswith("system")


def test_a_pre_styled_value_survives_intact():
    """Values are last on their line, so nothing is padded after them and colour is safe."""
    assert style.good("✔") in detail.fields([("OPENAI_API_KEY", style.good("✔"))])[0]


def test_no_fields_renders_nothing():
    assert detail.fields([]) == []


# --- rules and panels: structure that a pipe should not carry ------------------------------------ #

def test_a_rule_is_dashes_on_a_terminal(boxed):
    assert "──" in click.unstyle(detail.rule("secrets")[0])


def test_a_piped_rule_is_greppable_by_its_name(piped):
    """A `grep secrets` should find the section, not have to skip a hyphen fence."""
    assert click.unstyle(detail.rule("secrets")[0]) == "secrets"


def test_a_panel_boxes_its_fields_on_a_terminal(boxed):
    lines = visible(detail.panel("mem0:voyage", FIELDS))

    assert lines[0].startswith("╭") and lines[-1].startswith("╰")
    assert "mem0:voyage" in lines[0]
    for label, value in FIELDS:
        assert any(label in line and value in line for line in lines)


def test_a_piped_panel_is_the_heading_and_the_pairs(piped):
    """It degrades to what the view printed before boxes existed, so a redirect is unchanged."""
    lines = visible(detail.panel("mem0:voyage", FIELDS))

    assert lines[0] == "mem0:voyage"
    assert not any(char in "".join(lines) for char in "╭│╰─")
    assert len(lines) == 1 + len(FIELDS)


def test_a_panel_stays_inside_the_terminal(boxed, monkeypatch):
    monkeypatch.setattr(style, "terminal_width", lambda default=100: 40)

    for line in visible(detail.panel("t", [("k", "v" * 80)])):
        assert len(line) <= 40, repr(line)


def test_no_color_keeps_the_panel_and_drops_the_codes(boxed, monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")

    joined = "\n".join(detail.panel("mem0:voyage", FIELDS))
    assert "\x1b[" not in joined
    assert "╭" in joined


def test_emit_puts_a_show_view_on_stdout(piped, capsys):
    """A `show` view's fields ARE the command's answer."""
    detail.emit(detail.fields(FIELDS))

    captured = capsys.readouterr()
    assert captured.err == ""
    assert len(captured.out.strip().splitlines()) == len(FIELDS)
