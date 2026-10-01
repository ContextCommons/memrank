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
"""Rendering a run's progress: bars on a terminal, lines everywhere else.

    hindsight x locomo                              unit 3/10
      overall     ████████░░░░░░░░░░░░   24%                     ~34m left
      ▶ ingest    ███████░░░░░░░░░░░░░   81/272     5.2 s/doc    ~16m left
        retrieve  ████░░░░░░░░░░░░░░░░   924/4620   3.4/s        ~18m left

EVERY BAR CARRIES ITS OWN RATE AND ITS OWN REMAINING TIME. One combined figure hides whether a
slow run is slow at ingest or slow at retrieval -- which on hindsight differ by ~18x and are the
first thing worth knowing about an engine.

TWO OUTPUTS, ONE SOURCE. A terminal gets a block redrawn in place; a pipe, a log or CI gets
append-only lines with no escape sequences. Both render the same record, so what you read in a CI
log is what you would have watched live.
"""
from __future__ import annotations

import os
import re
import sys
import threading
import time
import unicodedata
from typing import Any

from memrank.term import fmt, style

BAR_WIDTH = 20
FILLED, EMPTY = "█", "░"

#: Each stage and the item its rate is quoted in, in the order a run reaches them. The order
#: mirrors :data:`memrank.runs.progress.STAGES`; the nouns live here because they are how a
#: number is read, not part of the model.
STAGE_NOUNS: tuple[tuple[str, str], ...] = (
    ("ingest", "doc"), ("retrieve", "query"), ("judge", "query"))

#: Stages a run may legitimately not have at all, and which are therefore hidden when empty.
OPTIONAL_STAGES = frozenset({"judge"})

#: Shown under the bars, but only where a key is actually being read -- advertising `q` on a pipe
#: or a background job promises something that will not happen.
QUIT_HINT = style.dim("  press q to detach (the run keeps going)")

#: Rendered where a number is genuinely unknown -- a stage that has not run, a run too young to
#: project. Never a zero, which would read as a measurement.
UNKNOWN = fmt.UNKNOWN


def bar(done: int, total: int, width: int = BAR_WIDTH, *, colour: str = "") -> str:
    """A fixed-width bar. An unknown or empty total renders empty rather than full -- dividing by
    zero into a full bar would announce a finished stage that never started.

    Any work at all shows at least one block. 12 of 272 documents is 0.9 of a block and would
    round to nothing, leaving a stage that has been running for a minute looking untouched -- the
    distinction between "started" and "not started" matters more here than the first 4%.

    ``colour`` styles the FILLED portion only -- ``"active"`` while a stage is moving, ``"done"``
    once it is complete. The empty portion is always dimmed, so the boundary between what has
    happened and what has not is visible without counting blocks. The bar's WIDTH never changes:
    styling wraps the two runs of characters rather than being interleaved per block, so the
    escape codes cannot desynchronise the column.
    """
    if total <= 0:
        return style.dim(EMPTY * width)
    filled = min(width, int(width * done / total))
    if done > 0:
        filled = max(1, filled)
    head, tail = FILLED * filled, EMPTY * (width - filled)
    paint = {"active": style.accent, "done": style.good}.get(colour, lambda text: text)
    return paint(head) + style.dim(tail)


#: Number rendering lives in :mod:`memrank.term.fmt`, which `runs show` also uses. Two
#: implementations of "how do you print a duration" drift on the first edge case, and these surfaces
#: show the same quantities minutes apart.
rate = fmt.rate
duration = fmt.approx_duration


def remaining(seconds: float | None, *, prefix: str = "") -> str:
    """``~34m left``, or a bare ``—`` when there is nothing to project.

    ``— left`` reads as a duration whose value went missing; a bare dash reads as what it is.
    """
    if seconds is None:
        return UNKNOWN
    return f"{prefix}{duration(seconds)} left"


def _stage_line(name: str, stage: dict[str, Any], *, noun: str, active: bool) -> str:
    """One stage's bar, counts, rate and remaining time.

    Colour says one thing per element: cyan for the stage moving now, green for one that has
    finished, dim for the rate and the ETA because they qualify the counts rather than compete
    with them. Every padded cell goes through `style.pad`, which pads before styling -- the other
    order misaligns the column while every substring assertion still passes.
    """
    done, total = stage.get("done", 0), stage.get("total", 0)
    complete = total > 0 and done >= total
    marker = style.accent("▶") if active else " "
    return (f"  {marker} {style.pad(name, 9, style.label)} "
            f"{bar(done, total, colour='done' if complete else 'active' if active else '')}  "
            f"{style.pad(f'{done}/{total}', 12)} "
            f"{style.pad(rate(stage.get('rate_per_second'), noun), 11, style.unit)} "
            f"{style.unit(remaining(stage.get('eta_seconds')))}")


def block(title: str, progress: dict[str, Any]) -> list[str]:
    """The full multi-line block for one run. Returned as lines so the caller owns the cursor."""
    pct = progress.get("pct")
    overall_bar = bar(pct or 0, 100, colour="done" if pct == 100 else "active")
    # "≥" while a stage that still has work has never run: its cost is unknown, so the total can
    # only grow. Saying so beats a confident number that understates a hindsight run by ~20 min.
    prefix = "≥" if progress.get("eta_is_lower_bound") else ""
    overall_eta = remaining(progress.get("eta_seconds"), prefix=prefix)
    unit_of = f"unit {progress.get('unit', 0)}/{progress.get('units', 0)}"
    lines = [
        f"{style.heading(title)}{'':<8}{style.unit(unit_of)}",
        f"    {style.pad('overall', 9, style.label)} {overall_bar}  "
        f"{style.pad((str(pct) + '%') if pct is not None else UNKNOWN, 12)} {'':<11} "
        f"{style.unit(overall_eta)}",
    ]
    for name, noun in STAGE_NOUNS:
        stage = progress.get(name) or {}
        active = progress.get("stage") == name
        # Judging is the one stage a run can simply not have, and an unjudged run must not carry
        # a permanently empty fourth bar -- that reads as a phase stuck at zero rather than one
        # never asked for. Ingest and retrieve always show: `no-context` and `fixed-context` ingest nothing, and
        # their empty ingest bar is a true statement about a read-only engine (see `bar`).
        if name in OPTIONAL_STAGES and not active and not stage.get("total", 0):
            continue
        lines.append(_stage_line(name, stage, noun=noun, active=active))
    return lines


def line(title: str, progress: dict[str, Any]) -> str:
    """One append-only line -- what a pipe, a log or CI reads.

    Same record, same numbers, no escape sequences: a CI log shows what you would have watched.
    """
    pct = progress.get("pct")
    stage = progress.get("stage") or UNKNOWN
    current = progress.get(stage) or {}
    prefix = "≥" if progress.get("eta_is_lower_bound") else ""
    return (f"{title}  {stage} u{progress.get('unit', 0)}/{progress.get('units', 0)}  "
            f"{current.get('done', 0)}/{current.get('total', 0)}  "
            f"{(str(pct) + '%') if pct is not None else UNKNOWN}  "
            f"{remaining(progress.get('eta_seconds'), prefix=prefix)}")


#: How often a download redraws. Every chunk would be tens of thousands of writes on a 271 MB
#: cell, and on an append-only stream that is a flooded log rather than progress.
_DOWNLOAD_EVERY_BYTES = 2_000_000
_DOWNLOAD_EVERY_SECONDS = 0.5


class DownloadProgress:
    """Bytes as they land, for a transfer big enough that silence reads as a hang.

    A 271 MB artifact printed its name and then nothing for minutes, which is indistinguishable
    from a stall -- and until artifacts are served by redirect, some of those transfers genuinely do
    die partway. ``download_artifact`` has carried an ``on_chunk`` hook for this since it was
    written; this is the renderer it was waiting for.

    Uses the same bar, rate and remaining-time vocabulary as the run views, so a download reads
    like the rest of the tool rather than inventing a second dialect. On a terminal it redraws one
    line; on a pipe it appends at intervals, because escape sequences in a CI log are noise.

    ONE METER PER FETCH, not per file. Artifacts come down concurrently, so ``files`` lets a
    single instance count a whole run's transfer, and every public method is locked. The rendered
    line already omitted ``name`` whenever the total was known, so a fetch-wide meter needed a
    lock and a file counter rather than a second class.
    """

    def __init__(self, name: str, total: int | None, *, stream: Any = None,
                 now: Any = None, region: Any = None, files: int = 0) -> None:
        self.name = name
        self.total = total or 0
        self.done = 0
        # A whole fetch, not one file. `fetch_artifacts` downloads a run's artifacts concurrently,
        # and one meter per file would have N of them fighting over the single footer slot below.
        # So one meter counts the fetch: the listing carries every size, so the denominator is
        # known before a byte moves and the ETA covers the thing the user is actually waiting for.
        self.files = files
        self._files_done = 0
        # Every mutable field above plus the write itself. Concurrent workers call `advance`, and
        # two composing a line and emitting escape sequences on one stream interleave into rows
        # no later frame repairs -- the same reason `LiveRegion` holds a lock.
        self._lock = threading.Lock()
        self._stream = stream if stream is not None else sys.stderr
        self._now = now or time.monotonic
        self._started = self._now()
        self._last_at = 0.0
        self._last_done = 0
        self._drawn = 0
        # A transfer that happens WHILE a block is live must not move the cursor itself: two
        # bookkeepers on one stream each erase rows the other drew. Given a region, this becomes a
        # footer line the region composes, and the region stays the only thing holding the cursor.
        self._region = region if (region is not None and region.live) else None
        self._redraw = self._region is None and supports_redraw(self._stream)

    def advance(self, count: int) -> None:
        """Record ``count`` more bytes, drawing only when enough has changed to be worth it.

        ``count`` may be NEGATIVE: a resumed transfer whose server ignored the ``Range`` header
        restarts the body, and `_download_direct` rewinds the meter by the bytes already counted.
        Only the DISPLAY clamps at zero; the accumulator is left honest.
        """
        with self._lock:
            self.done += count
            if self._should_draw():
                self._draw()

    def file_done(self) -> None:
        """Record one artifact of a multi-file fetch as landed."""
        with self._lock:
            self._files_done += 1

    def finish(self) -> None:
        """Draw the completed state once, so the last thing shown is not a stale 97%."""
        with self._lock:
            self._finish_locked()

    def _finish_locked(self) -> None:
        self._draw(force=True)
        if self._region is not None:
            # The transfer is over; the block below it is not, so the footer is retired rather
            # than left showing a finished download under a run that is still going.
            self._region.set_footer([])
            return
        if self._redraw:
            self._stream.write("\n")
            self._stream.flush()

    def _should_draw(self) -> bool:
        elapsed = self._now() - self._last_at
        return (self.done - self._last_done >= _DOWNLOAD_EVERY_BYTES
                or elapsed >= _DOWNLOAD_EVERY_SECONDS)

    def _line(self) -> str:
        elapsed = max(self._now() - self._started, 1e-9)
        per_second = self.done / elapsed
        # No total means no bar and no projection: a percentage invented from an unknown
        # denominator is worse than admitting the size is not known.
        done = max(0, self.done)
        if self.total <= 0:
            return (f"  {self.name}  {fmt.size(done)}{self._files()}  "
                    f"{fmt.rate(per_second, 'B')}")
        left = (self.total - done) / per_second if per_second > 0 else None
        pct = min(100, int(100 * done / self.total))
        return (f"  {bar(done, self.total, colour='active')} {pct:3d}%  "
                f"{fmt.size(done)}/{fmt.size(self.total)}{self._files()}  {remaining(left)}")

    def _files(self) -> str:
        """The file counter, and only for a fetch that has more than one.

        Guarded on ``> 1`` so a single-artifact transfer renders exactly the line it always has --
        a count of ``1/1`` is noise, and every existing render is a one-file transfer.
        """
        return f"  {self._files_done}/{self.files} files" if self.files > 1 else ""

    def _draw(self, *, force: bool = False) -> None:
        line = self._line()
        if self._region is not None:
            self._region.set_footer([line])
            self._last_at = self._now()
            self._last_done = self.done
            return
        if self._redraw:
            self._stream.write(redraw([line], self._drawn))
            self._drawn = 1
        elif force or self._last_at == 0.0 or self._should_draw():
            self._stream.write(line + "\n")
        self._stream.flush()
        self._last_at = self._now()
        self._last_done = self.done


def redraw(lines: list[str], previous: int) -> str:
    """``lines`` preceded by enough cursor-up to overwrite ``previous`` lines of a prior draw.

    Kept here rather than in the caller so the one place that emits escape sequences is the one
    place :func:`supports_redraw` guards.
    """
    up = f"\033[{previous}A\033[J" if previous else ""
    return up + "\n".join(lines)


def supports_redraw(stream: Any) -> bool:
    """Whether to redraw in place: a real terminal, and not NO_COLOR.

    Reuses the NO_COLOR gate :mod:`memrank.term.style` already owns, so one environment variable
    silences colour and cursor control together rather than half of each.
    """
    return bool(getattr(stream, "isatty", lambda: False)()) and style.wants_color()


# --- the live region: how a block can be redrawn without ever corrupting the screen ------------- #
#
# THE ONE RULE, and everything below follows from it: a cursor-up count is a count of LOGICAL
# lines, but `\033[nA` moves SCREEN ROWS. Those two numbers are equal if, and only if, no line
# wraps. So every line entering the region is truncated to the terminal's width first, and the
# arithmetic is then correct by construction rather than by luck. This is what rich, tqdm,
# cli-progress and Docker's progress writer all do; indicatif takes the other road and recomputes
# the wrapped height per frame, which buys soft-wrapping nobody here wants.
#
# The block used to be written with a trailing newline and erased with `len(lines) + 1` -- one row
# too many, which ate the line above it -- and its ~70-column lines were never truncated at all, so
# on any narrower terminal each one wrapped into two rows while the count stayed at one. Those are
# the two failures this replaces.

#: Matches a CSI escape sequence. These are ZERO WIDTH on screen and yet count toward `len()`,
#: which is why measuring and slicing both have to go through here.
_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")

_HIDE_CURSOR, _SHOW_CURSOR = "\033[?25l", "\033[?25h"
#: DEC 2026. Makes a multi-line frame land atomically instead of tearing; terminals that do not
#: know the mode ignore an unknown private sequence, so this costs nothing where it is unsupported.
_SYNC_ON, _SYNC_OFF = "\033[?2026h", "\033[?2026l"
#: Erase the whole row (EL 2) -- and note it does NOT move the cursor, which is why every use of it
#: is paired with an explicit column reset.
_ERASE_ROW = "\033[2K"
#: Erase rightward from the cursor: kills the tail of a longer previous frame on the final row.
_ERASE_TAIL = "\033[K"
#: Column 1 of the current row (CHA). Preferred over `\r`, which log capture and CI pipelines
#: sometimes rewrite.
_COLUMN_ONE = "\033[G"

#: Shown in place of the rows that did not fit. A frame taller than the terminal scrolls its top
#: rows away, and a row that has scrolled off can never be reached by a cursor-up again -- the count
#: would be wrong forever after.
_CLIPPED = "  …"

#: What a width lookup falls back to. 80 is the historical default and the safe guess: too NARROW
#: only truncates, while too wide wraps, which is the failure this module exists to prevent.
_FALLBACK_WIDTH = 80
_FALLBACK_HEIGHT = 24


def display_width(text: str) -> int:
    """How many COLUMNS ``text`` occupies once drawn.

    Three corrections over ``len()``, each of which has its own way of breaking a redraw: escape
    sequences are stripped (they print nothing but count toward the length), combining marks count
    zero (they attach to the character before them), and East Asian wide and fullwidth characters
    count two. The block renders `×`, `▶`, `█` and `░` and titles carry whatever a target is
    called, so none of the three is hypothetical here.
    """
    total = 0
    for char in _ANSI.sub("", text):
        if unicodedata.combining(char):
            continue
        total += 2 if unicodedata.east_asian_width(char) in ("W", "F") else 1
    return total


def truncate(text: str, width: int) -> str:
    """``text`` cut to ``width`` COLUMNS, keeping its escape sequences.

    Slicing with ``text[:width]`` would count escape bytes as visible and cut a sequence in half,
    putting a raw `[3` on screen. This walks the string instead, passing escapes through at zero
    width and stopping when the next visible character would not fit.

    A cut inside a styled run is closed with a reset, so a truncated colour cannot leak across the
    rest of the screen -- the same guard tqdm's `disp_trim` applies for the same reason.
    """
    if width <= 0:
        return ""
    if display_width(text) <= width:
        return text
    out: list[str] = []
    column, styled, index = 0, False, 0
    while index < len(text):
        escape = _ANSI.match(text, index)
        if escape:
            sequence = escape.group()
            out.append(sequence)
            if sequence.endswith("m"):
                styled = sequence not in ("\033[0m", "\033[m")
            index = escape.end()
            continue
        char = text[index]
        size = 0 if unicodedata.combining(char) else (
            2 if unicodedata.east_asian_width(char) in ("W", "F") else 1)
        if column + size > width:
            break
        out.append(char)
        column += size
        index += 1
    if styled:
        out.append("\033[0m")
    return "".join(out)


def _terminal_size(stream: Any) -> os.terminal_size:
    """The size of the terminal behind ``stream``, or a safe default.

    Measured against THE STREAM BEING DRAWN TO, which is the difference from
    :func:`memrank.term.style.terminal_width`: that one asks stdout because it sizes tables, and a
    progress bar on stderr with stdout piped would get its fallback -- a width wider than the real
    terminal, which is precisely the case that wraps.
    """
    columns = os.environ.get("COLUMNS")
    if columns and columns.isdigit() and int(columns) > 0:
        return os.terminal_size((int(columns), _FALLBACK_HEIGHT))
    try:
        return os.get_terminal_size(stream.fileno())
    except (OSError, ValueError, AttributeError):
        return os.terminal_size((_FALLBACK_WIDTH, _FALLBACK_HEIGHT))


def region_width(stream: Any) -> int:
    """Columns available to a live region on ``stream``. Never below 20: below that nothing
    renders usefully and the arithmetic is all that is left to protect."""
    return max(20, _terminal_size(stream).columns)


def region_height(stream: Any) -> int:
    """Rows available to a live region on ``stream``."""
    return max(2, _terminal_size(stream).lines)


class LiveRegion:
    """A block of lines redrawn in place, which owns the cursor for as long as it is open.

    Everything that writes to the same stream while the region is live must come THROUGH it --
    :meth:`print_above` for a log line, :meth:`set_footer` for a transient extra line -- because a
    second writer moves the cursor without the region knowing, and from that point its count of
    what is on screen describes rows that hold something else.

    The frame invariant, borrowed from indicatif and rich: the cursor rests on the LAST drawn row,
    which is deliberately NOT newline-terminated. Erasing is therefore ``n-1`` rows up, not ``n``,
    and a block at the bottom of the screen does not scroll the terminal on every frame.

    Locked, because the cloud watch runs an event-stream thread per run: two sets of interleaved
    cursor movements produce garbage no later frame can repair.
    """

    def __init__(self, stream: Any = None, *, width: Any = None, height: Any = None) -> None:
        self._stream = stream if stream is not None else sys.stderr
        # Injected the way `DownloadProgress` injects its clock: a test can state the terminal it
        # is rendering for instead of inheriting whatever ran the suite.
        self._width = width or (lambda: region_width(self._stream))
        self._height = height or (lambda: region_height(self._stream))
        self._lock = threading.RLock()
        self._live = supports_redraw(self._stream)
        self._base: list[str] = []
        self._footer: list[str] = []
        self._rows = 0
        self._frame = ""
        self._last_width = 0
        self._hidden = False

    @property
    def live(self) -> bool:
        """Whether this region redraws at all. False for a pipe, CI or NO_COLOR, where every
        method below degrades to appending plain text or to doing nothing."""
        return self._live

    def draw(self, lines: list[str]) -> None:
        """Replace the region's body with ``lines``."""
        with self._lock:
            self._base = list(lines)
            self._render()

    def set_footer(self, lines: list[str]) -> None:
        """Lines pinned under the body -- a download in flight, and nothing else so far.

        Separate from the body so the two writers do not overwrite each other's content: the watch
        loop owns :meth:`draw`, a transfer owns this, and the region composes them.
        """
        with self._lock:
            self._footer = list(lines)
            self._render()

    def print_above(self, text: str) -> None:
        """Write ``text`` permanently above the region, then redraw it.

        The one safe way to emit a log line while a block is live -- erase, print, redraw -- and the
        same primitive tqdm exposes as `external_write_mode` and indicatif as `MultiProgress::println`.
        With no region live this is an ordinary line on stderr, which is what it should be.
        """
        with self._lock:
            if not self._live:
                style.say(text)
                return
            self._stream.write(self._erase() + _ERASE_ROW)
            self._rows, self._frame = 0, ""
            self._stream.write(text + "\n")
            self._stream.flush()
            self._render()

    def close(self, *, persist: bool = True) -> None:
        """Release the cursor. Idempotent, and safe to call from a `finally`.

        ``persist`` keeps the last frame on screen and moves past it; False erases it. Showing the
        cursor again is the part that must not be skipped on any path -- a hidden cursor outlives
        the process and the user is left with a shell that looks broken.
        """
        with self._lock:
            if not self._live:
                return
            if self._rows:
                if persist:
                    self._stream.write("\n")
                else:
                    self._stream.write(self._erase() + _ERASE_ROW)
                self._rows = 0
            if self._hidden:
                self._stream.write(_SHOW_CURSOR)
                self._hidden = False
            self._frame = ""
            self._stream.flush()

    def _render(self) -> None:
        """Draw body + footer as one frame. Assumes the lock is held."""
        if not self._live:
            return
        # Read ONCE per frame and use the same number to truncate and to count: querying twice
        # across a resize measures one width and erases with another.
        width, height = self._width(), self._height()
        lines = [truncate(line, width - 1) for line in self._base + self._footer]
        if len(lines) > height:
            lines = lines[:max(1, height - 1)] + [truncate(style.dim(_CLIPPED), width - 1)]
        frame = "\n".join(lines)
        # A width change invalidates the cache as surely as a content change: the same text at a
        # new width is a different set of rows. Also why no SIGWINCH handler is needed -- the size
        # is re-read on every frame, so a resize is picked up by the next one.
        if frame == self._frame and width == self._last_width:
            return
        out = [_SYNC_ON]
        if not self._hidden:
            out.append(_HIDE_CURSOR)
            self._hidden = True
        out.extend((self._erase(), frame, _ERASE_TAIL, _SYNC_OFF))
        self._stream.write("".join(out))
        self._stream.flush()
        self._rows, self._frame, self._last_width = len(lines), frame, width

    def _erase(self) -> str:
        r"""Clear the previous frame and leave the cursor at column 1 of its first row.

        Walks UP ``n-1`` rows (the cursor is on the last one, which carries no newline), clears
        each row on the way down, then walks back up. `\033[2K` erases a whole row without moving
        the cursor, so the column is meaningless until the closing `\033[G` fixes it.
        """
        rows = self._rows
        if rows == 0:
            return _COLUMN_ONE
        up = f"\033[{rows - 1}A" if rows > 1 else ""
        body = (_ERASE_ROW + "\033[1B") * (rows - 1) + _ERASE_ROW
        return up + body + up + _COLUMN_ONE
