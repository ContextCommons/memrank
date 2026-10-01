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
"""Showing a run's progress: bars on a terminal, lines everywhere else.

The rendering decisions that carry meaning rather than taste -- a rate shown on the readable side
of one item per second, an unknown shown as unknown rather than zero, and no escape sequences
where something is going to read the output rather than watch it.
"""
from __future__ import annotations

import io
import re

from memrank.term import progress as render

#: Everything here asserts what a reader SEES. Colour is presentation; a test that matched raw
#: bytes would fail on every palette change and still miss the failure that matters -- a styled
#: cell padded in the wrong order, which shifts columns while leaving every substring intact.
ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def plain(text: str) -> str:
    return ANSI.sub("", text)

RECORD = {
    "stage": "ingest", "unit": 3, "units": 10,
    "ingest": {"done": 81, "total": 272, "rate_per_second": 0.1938, "eta_seconds": 985},
    "retrieve": {"done": 924, "total": 4620, "rate_per_second": 3.413, "eta_seconds": 1083},
    "pct": 24, "eta_seconds": 2068, "eta_is_lower_bound": False,
}


def test_a_rate_is_shown_on_its_readable_side():
    """These two differ by ~18x on hindsight. One forced direction makes one of them unreadable:
    ingest as `0.19 docs/s` says nothing a person can hold."""
    assert render.rate(0.1938, "doc") == "5.2 s/doc"
    assert render.rate(3.413, "query") == "3.4/s"


def test_an_unknown_rate_is_not_a_zero():
    """A stage that has not run has no rate. `0.0/s` would read as an engine that has stalled."""
    assert render.rate(None, "doc") == "—"
    assert render.rate(0, "doc") == "—"


def test_durations_are_coarse_on_purpose():
    """A projection from an observed rate; `~34m 12s` would claim precision it does not have."""
    assert render.duration(45) == "~45s"
    assert render.duration(2068) == "~34m"
    assert render.duration(7500) == "~2h 05m", "zero-padded so a column of them aligns"
    assert render.duration(None) == "—"


def test_an_empty_total_renders_an_empty_bar_not_a_full_one():
    """A stage with no work must not announce itself finished -- `none` and `icl` ingest nothing."""
    assert plain(render.bar(0, 0)) == "░" * render.BAR_WIDTH


def test_a_bar_is_always_exactly_its_width(monkeypatch):
    """Colour wraps the two runs of characters; it must never change how many there are, or the
    columns after it walk. Checked across the whole range, not one sample."""
    for done in (0, 1, 7, 19, 20, 40):
        assert len(plain(render.bar(done, 20, colour="active"))) == render.BAR_WIDTH


def test_the_filled_and_empty_halves_are_styled_differently():
    """Where progress ends has to be visible without counting blocks."""
    rendered = render.bar(10, 20, colour="active")

    assert "█" in plain(rendered) and "░" in plain(rendered)
    assert rendered.count("\x1b[") >= 2, "two runs, styled separately"


def test_the_block_shows_all_three_bars_with_their_own_rates_and_etas():
    """The point of the layout: which stage is slow is visible, not inferred."""
    lines = render.block("hindsight × locomo", RECORD)
    body = "\n".join(lines)

    body = plain(body)
    assert "overall" in body and "24%" in body
    assert "81/272" in body and "5.2 s/doc" in body and "~16m left" in body
    assert "924/4620" in body and "3.4/s" in body and "~18m left" in body


#: The same run, judged: 69 of its 152 queries are gradable, and judging has begun.
JUDGED = {**RECORD, "stage": "judge", "pct": None, "eta_is_lower_bound": False,
          "judge": {"done": 14, "total": 69, "rate_per_second": 0.077, "eta_seconds": 715}}


def test_judging_gets_a_bar_of_its_own():
    """~15 minutes that used to render nothing, on a phase `watch` reported as a dead run."""
    body = plain("\n".join(render.block("hindsight × locomo", JUDGED)))

    assert "judge" in body and "14/69" in body and "13.0 s/query" in body
    assert "~11m left" in body


def test_an_unjudged_run_grows_no_empty_judge_bar():
    """A permanently empty fourth bar reads as a phase stuck at zero, not one never asked for."""
    lines = render.block("hindsight × locomo", RECORD)

    assert len(lines) == 4, "title, overall, ingest, retrieve -- as before"
    assert not any("judge" in plain(line) for line in lines)


def test_ingest_keeps_its_empty_bar_when_an_engine_reads_only():
    """`none` and `icl` ingest nothing, and their empty ingest bar is a true statement about the
    engine -- unlike judging, which is a phase the run simply did not have."""
    read_only = {**RECORD, "ingest": {"done": 0, "total": 0, "rate_per_second": None,
                                      "eta_seconds": None}}

    assert any("ingest" in plain(line) for line in render.block("icl × locomo", read_only))


def test_the_moving_stage_is_marked():
    lines = render.block("hindsight × locomo", RECORD)

    assert any(plain(line).strip().startswith("▶ ingest") for line in lines)
    assert not any("▶ retrieve" in plain(line) for line in lines)


def test_an_unprojectable_run_shows_unknown_rather_than_zero():
    """Before retrieval has run, the overall percentage is genuinely unknown -- see
    `progress_model.pct`. Printing 0% would be a claim; `—` is the truth."""
    early = {**RECORD, "pct": None, "eta_seconds": None}

    body = "\n".join(render.block("hindsight × locomo", early))

    assert "—" in plain(body) and "0%" not in plain(body)


def test_a_lower_bound_eta_says_so():
    """During unit 1 the total can only grow, because a stage with work has never run."""
    body = "\n".join(render.block("t", {**RECORD, "eta_is_lower_bound": True}))

    assert "≥~34m left" in plain(body)


def test_the_line_form_carries_no_escape_sequences():
    """What CI, a pipe and `tee` read. An escape sequence there is noise in a log forever."""
    rendered = render.line("hindsight × locomo", RECORD)

    assert "\033" not in rendered
    assert "ingest u3/10" in rendered and "24%" in rendered and "~34m left" in rendered


def test_redraw_moves_the_cursor_only_when_there_was_a_previous_draw():
    """The first draw must not emit a cursor-up, which would eat a line of whatever came before."""
    assert not render.redraw(["a"], 0).startswith("\033")
    assert render.redraw(["a"], 4).startswith("\033[4A")


def test_a_pipe_never_gets_the_redrawing_form():
    class NotATerminal:
        def isatty(self):
            return False

    assert not render.supports_redraw(NotATerminal())


def test_no_color_disables_redraw_on_a_real_terminal(monkeypatch):
    """One environment variable silences colour and cursor control together -- half of each is
    what makes a log full of stray escapes."""
    class Terminal:
        def isatty(self):
            return True

    monkeypatch.setenv("NO_COLOR", "1")

    assert not render.supports_redraw(Terminal())


# --- what `watch` actually draws ---------------------------------------------------------------- #

def _run_dir(tmp_path, run_id, progress):
    import json
    directory = tmp_path / run_id
    directory.mkdir()
    (directory / "status.json").write_text(json.dumps({
        "run_id": run_id, "pid": 1, "target": "hindsight", "benchmark": "locomo", "slice": None,
        "state": "running", "progress": progress, "message": "", "error": None,
        "started_at": "2026-08-04T05:48:35+00:00", "updated_at": "2026-08-04T05:50:00+00:00"}),
        encoding="utf-8")
    return directory


def _run_dir_update(tmp_path, run_id, progress):
    """Rewrite a run's heartbeat, so a second draw has something new to show."""
    import json
    path = tmp_path / run_id / "status.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["progress"] = progress
    path.write_text(json.dumps(data), encoding="utf-8")


def test_a_pipe_gets_one_line_per_change_and_nothing_when_unchanged(tmp_path, monkeypatch, capsys):
    """A 45-minute watch polling every 15s would otherwise put 180 identical lines in a CI log."""
    from memrank.cli import watch as watch_cli

    _run_dir(tmp_path, "r1", RECORD)
    monkeypatch.setattr(watch_cli.progress, "supports_redraw", lambda stream: False)
    display = watch_cli.ProgressDisplay()

    display.show(tmp_path, ["r1"], cloud=None)
    display.show(tmp_path, ["r1"], cloud=None)

    assert capsys.readouterr().err.count("hindsight × locomo") == 1


def test_a_terminal_gets_a_block_that_erases_its_previous_draw(tmp_path, monkeypatch, capsys):
    """Without the cursor-up, fifteen polls leave fifteen stacked blocks.

    And with the WRONG cursor-up it ate a line of whatever was above it: the block is four rows,
    it used to erase five, and the fifth was the last thing the user had been told. The count is
    now rows-1, because the frame's last row deliberately carries no newline -- so the cursor is
    already sitting on it rather than on the row below.
    """
    from memrank.cli import watch as watch_cli

    _run_dir(tmp_path, "r1", RECORD)
    monkeypatch.setattr(watch_cli.progress, "supports_redraw", lambda stream: True)
    display = watch_cli.ProgressDisplay()

    display.show(tmp_path, ["r1"], cloud=None)
    first = capsys.readouterr().err
    _run_dir_update(tmp_path, "r1", {**RECORD, "pct": 25})
    display.show(tmp_path, ["r1"], cloud=None)
    second = capsys.readouterr().err

    rows = len(render.block("t", RECORD))
    assert rows == 4, "the block this test reasons about"
    # Colour codes are expected; a CURSOR MOVEMENT on the first draw is not -- it would eat a
    # line of whatever the user had on screen before.
    assert not re.search(r"\x1b\[\d+A", first), "the first draw must not move the cursor"
    assert re.search(r"\x1b\[(\d+)A", second).group(1) == str(rows - 1)
    assert second.count("\033[2K") == rows, "exactly the rows it drew, and not one more"


def test_an_unchanged_block_is_not_rewritten(tmp_path, monkeypatch, capsys):
    """A watch redraws every 2s for up to 45 minutes. Rewriting an identical frame is ~1300
    pointless repaints, and in a `script` or CI capture it is 1300 copies of the block."""
    from memrank.cli import watch as watch_cli

    _run_dir(tmp_path, "r1", RECORD)
    monkeypatch.setattr(watch_cli.progress, "supports_redraw", lambda stream: True)
    display = watch_cli.ProgressDisplay()

    display.show(tmp_path, ["r1"], cloud=None)
    capsys.readouterr()
    display.show(tmp_path, ["r1"], cloud=None)

    assert capsys.readouterr().err == ""


def test_a_run_with_no_progress_draws_nothing(tmp_path, monkeypatch, capsys):
    """A run from before this existed, or a cloud task that has not published yet. Absence of a
    bar is normal and must be silent -- not an error, and not an empty frame."""
    from memrank.cli import watch as watch_cli

    _run_dir(tmp_path, "r1", {})
    monkeypatch.setattr(watch_cli.progress, "supports_redraw", lambda stream: False)

    watch_cli.ProgressDisplay().show(tmp_path, ["r1"], cloud=None)

    assert capsys.readouterr().err == ""


def test_the_run_is_named_by_its_cell_not_its_id(tmp_path, monkeypatch, capsys):
    """`hindsight × locomo` is what a person watching a sweep is distinguishing between."""
    from memrank.cli import watch as watch_cli

    _run_dir(tmp_path, "20260804-054835__locomo__smoke__ef5f43", RECORD)
    monkeypatch.setattr(watch_cli.progress, "supports_redraw", lambda stream: False)

    watch_cli.ProgressDisplay().show(
        tmp_path, ["20260804-054835__locomo__smoke__ef5f43"], cloud=None)

    assert "hindsight × locomo" in capsys.readouterr().err


# --- alignment, the failure that greps cannot see ------------------------------------------------ #

def test_every_column_lines_up_across_wildly_different_values():
    """THE reason `style.pad` exists.

    ANSI escapes count toward len(), so styling a cell before padding it shifts every column to
    its right -- and every substring assertion in this file still passes, because the characters
    are all present, in the wrong places. This measures VISIBLE width instead.
    """
    tiny = {**RECORD, "stage": "retrieve",
            "ingest": {"done": 1, "total": 9, "rate_per_second": 1000.0, "eta_seconds": 1},
            "retrieve": {"done": 0, "total": 4620, "rate_per_second": None,
                         "eta_seconds": None},
            "pct": 3}
    huge = {**RECORD,
            "ingest": {"done": 999999, "total": 999999, "rate_per_second": 0.001,
                       "eta_seconds": 99999},
            "retrieve": {"done": 4620, "total": 4620, "rate_per_second": 12345.0,
                         "eta_seconds": 0}, "pct": 100}

    for record in (RECORD, tiny, huge):
        stages = [plain(line) for line in render.block("t", record)[2:]]
        starts = [line.index("█") if "█" in line else line.index("░") for line in stages]
        assert len(set(starts)) == 1, f"bars start at different columns: {starts}\n" + "\n".join(stages)


def test_no_color_leaves_the_output_entirely_plain(monkeypatch):
    """The chokepoint's whole promise. Asserted rather than assumed, because every helper added
    here has to route through `style._fragment` for it to hold."""
    monkeypatch.setenv("NO_COLOR", "1")

    body = "\n".join(render.block("hindsight × locomo", RECORD))

    assert "\x1b" not in body
    assert body == plain(body)


# --- the live region: the failures that made the block dissolve ---------------------------------- #
#
# All of these are regressions for something seen on a real terminal. The block used to erase one
# row too many (eating the line above it) and never truncated its ~70-column lines, so on anything
# narrower each line wrapped into two rows while the count stayed at one -- and from that frame on,
# the redraw was overwriting rows that held something else.

CUU = re.compile(r"\x1b\[(\d+)A")


class _Screen(io.StringIO):
    """A terminal of a stated size. The size is what these tests are varying, so it is stated
    rather than inherited from whatever ran the suite."""

    def __init__(self, columns: int = 100, lines: int = 40) -> None:
        super().__init__()
        self.columns, self.lines = columns, lines

    def isatty(self) -> bool:
        return True


def _region(screen):
    return render.LiveRegion(screen, width=lambda: screen.columns, height=lambda: screen.lines)


def _rows_drawn(frame: str) -> list[str]:
    """The visible rows of a frame, with the escape machinery around them removed."""
    body = frame.split("\033[G")[-1]
    return [plain(row) for row in body.split("\033[K")[0].split("\n")]


def test_display_width_ignores_escapes_and_counts_wide_characters():
    """Three ways len() lies about a rendered line, and each one desynchronises a redraw."""
    from memrank.term import style

    assert render.display_width(style.accent("abc")) == 3, "escapes print nothing"
    assert render.display_width("×") == 1
    assert render.display_width("日本語") == 6, "East Asian wide characters take two columns each"
    assert render.display_width("é") == 1, "a combining accent rides the letter before it"


def test_truncate_cuts_by_column_and_closes_the_style_it_cut():
    """`text[:n]` would count escape bytes as visible and could cut a sequence in half, putting a
    raw `[3` on screen. And a cut inside a colour leaks it across the rest of the row."""
    from memrank.term import style

    cut = render.truncate(style.accent("abcdefghij"), 4)

    assert plain(cut) == "abcd"
    assert cut.endswith("\033[0m"), "a truncated style must be closed"
    assert render.truncate("日本語", 3) == "日", "a wide character does not half-fit"


def test_no_line_can_wrap_which_is_what_keeps_the_cursor_count_honest(monkeypatch):
    """THE fix. A cursor-up count counts LOGICAL lines but `\\033[nA` moves SCREEN ROWS, and the
    two are equal only if nothing wraps. The block is ~70 columns; on a 40-column terminal every
    line used to become two rows while the count stayed at one, and the redraw walked up into the
    middle of its own previous frame -- the "random pads" this replaces."""
    monkeypatch.setattr(render.style, "wants_color", lambda: True)
    screen = _Screen(columns=40)
    region = _region(screen)

    region.draw(render.block("hindsight × locomo", RECORD))

    rows = _rows_drawn(screen.getvalue())
    assert len(rows) == 4, "four logical lines"
    for row in rows:
        assert render.display_width(row) <= screen.columns - 1, f"would wrap: {row!r}"


def test_a_narrow_terminal_does_not_drift_across_frames(monkeypatch):
    """The visible symptom: each frame landing a row lower than the last, leaving a trail."""
    monkeypatch.setattr(render.style, "wants_color", lambda: True)
    screen = _Screen(columns=40)
    region = _region(screen)

    region.draw(render.block("hindsight × locomo", RECORD))
    screen.truncate(0), screen.seek(0)
    region.draw(render.block("hindsight × locomo", {**RECORD, "pct": 25}))

    frame = screen.getvalue()
    assert CUU.search(frame).group(1) == "3", "up three rows to the top of a four-row block"
    assert frame.count("\033[2K") == 4, "clearing exactly the four it drew"


def test_a_frame_taller_than_the_terminal_is_clipped(monkeypatch):
    """A row that scrolled off the top can never be reached by a cursor-up again, so the count
    would be wrong from then on -- permanently. Better to show fewer runs than to break."""
    monkeypatch.setattr(render.style, "wants_color", lambda: True)
    screen = _Screen(columns=100, lines=6)
    region = _region(screen)

    region.draw([f"run {n}" for n in range(20)])

    rows = _rows_drawn(screen.getvalue())
    assert len(rows) == screen.lines
    assert rows[-1].strip() == "…", "and it says that it clipped"


def test_a_resize_invalidates_the_frame_it_measured(monkeypatch):
    """The same text at a new width is a different set of rows. Caching on content alone would
    hold the old frame and never repaint after a resize -- which is why no SIGWINCH handler is
    needed: the size is re-read every frame and a change is a cache miss."""
    monkeypatch.setattr(render.style, "wants_color", lambda: True)
    screen = _Screen(columns=100)
    region = _region(screen)
    lines = render.block("hindsight × locomo", RECORD)

    region.draw(lines)
    screen.truncate(0), screen.seek(0)
    screen.columns = 50
    region.draw(lines)

    assert screen.getvalue() != "", "an unchanged block at a new width still has to be redrawn"
    for row in _rows_drawn(screen.getvalue()):
        assert render.display_width(row) <= 49


def test_print_above_keeps_the_line_and_puts_the_block_back(monkeypatch):
    """`watch` narrates while the block is live -- `done`, `fetching ...`. A bare write there
    scrolled the screen under a block that was still counting its old rows."""
    monkeypatch.setattr(render.style, "wants_color", lambda: True)
    screen = _Screen()
    region = _region(screen)
    region.draw(render.block("hindsight × locomo", RECORD))
    screen.truncate(0), screen.seek(0)

    region.print_above("r1: done")

    frame = screen.getvalue()
    assert "r1: done\n" in plain(frame), "the line is written permanently"
    assert "hindsight × locomo" in plain(frame), "and the block is drawn again beneath it"


def test_the_cursor_is_hidden_while_drawing_and_always_given_back(monkeypatch):
    """A hidden cursor OUTLIVES the process: leaving one behind hands the user a shell that looks
    broken and that quitting the program does not fix."""
    monkeypatch.setattr(render.style, "wants_color", lambda: True)
    screen = _Screen()
    region = _region(screen)

    region.draw(["one line"])
    assert "\033[?25l" in screen.getvalue()

    region.close()
    assert "\033[?25h" in screen.getvalue()

    before = screen.getvalue()
    region.close()
    assert screen.getvalue() == before, "close is idempotent -- a finally may reach it twice"


def test_a_region_on_a_pipe_emits_no_escape_sequences():
    """The whole promise of the non-terminal path: a CI log carries the numbers, never cursor
    motion. `print_above` degrades to an ordinary line rather than vanishing."""
    region = render.LiveRegion(io.StringIO())

    assert not region.live
    region.draw(render.block("hindsight × locomo", RECORD))
    region.print_above("r1: done")
    region.close()

    assert region._stream.getvalue() == "", "nothing is drawn in place"


def test_a_download_during_a_watch_draws_through_the_block(monkeypatch):
    """Two in-place drawers on one stream each erase rows the other wrote. The byte meter becomes
    a footer the block composes, which is what came apart when artifacts started landing."""
    monkeypatch.setattr(render.style, "wants_color", lambda: True)
    screen = _Screen()
    region = _region(screen)
    region.draw(render.block("hindsight × locomo", RECORD))

    meter = render.DownloadProgress("cell.json", 10_000_000, stream=screen, region=region)
    meter.advance(5_000_000)
    tail = plain(_rows_drawn(screen.getvalue())[-1])

    assert "50%" in tail, "the transfer is shown under the block"
    assert "hindsight × locomo" in plain(screen.getvalue()), "which is still drawn"

    screen.truncate(0), screen.seek(0)
    meter.finish()
    redrawn = plain(screen.getvalue())
    assert "hindsight × locomo" in redrawn, "the block outlives the transfer"
    assert "100%" not in redrawn, "and the finished footer is retired rather than left standing"
