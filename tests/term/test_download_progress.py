"""A large download reports bytes as they land.

A 271 MB artifact printed its name and then nothing for minutes -- indistinguishable from a hang,
and some of those transfers really do die partway, so silence hid where. `download_artifact` has
carried an `on_chunk` hook since it was written and nothing ever passed it; these pin the renderer
it was waiting for.
"""

from __future__ import annotations

import io
import re
import threading

from memrank.term.progress import DownloadProgress


class _Tty(io.StringIO):
    def isatty(self) -> bool:
        return True


class _Clock:
    """A hand-cranked clock, so throttling is tested by time rather than by sleeping."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def test_a_pipe_appends_lines_and_never_moves_the_cursor():
    """Cursor control in a CI log is noise. Colour is a separate question, owned by `style` and
    its NO_COLOR gate -- what must not appear here is a redraw, which would overwrite log lines."""
    out, clock = io.StringIO(), _Clock()
    meter = DownloadProgress("cell.json", 10_000_000, stream=out, now=clock)
    for _ in range(5):
        clock.t += 1.0
        meter.advance(2_000_000)
    meter.finish()

    text = out.getvalue()
    assert not re.search(r"\033\[\d*[AJ]", text)   # no cursor-up, no erase-to-end
    assert text.count("\n") >= 5                   # each update on its own line


def test_a_terminal_redraws_one_line():
    out, clock = _Tty(), _Clock()
    meter = DownloadProgress("cell.json", 10_000_000, stream=out, now=clock)
    for _ in range(4):
        clock.t += 1.0
        meter.advance(2_000_000)
    meter.finish()

    assert "\033[" in out.getvalue()          # cursor-up, i.e. redrawing in place


def test_small_chunks_do_not_each_draw():
    """Tens of thousands of writes on a 271 MB file is its own failure."""
    out, clock = io.StringIO(), _Clock()
    meter = DownloadProgress("cell.json", 10_000_000, stream=out, now=clock)
    for _ in range(500):                      # 500 x 1 KB, no time passing
        meter.advance(1000)

    assert out.getvalue().count("\n") <= 2    # throttled, not one line per chunk


def test_progress_counts_and_projects():
    out, clock = io.StringIO(), _Clock()
    meter = DownloadProgress("cell.json", 4_000_000, stream=out, now=clock)
    clock.t += 1.0
    meter.advance(2_000_000)

    line = out.getvalue()
    assert "50%" in line
    assert "2.0 MB/4.0 MB" in line
    assert "left" in line                     # an ETA, not just a count


def test_an_unknown_size_shows_bytes_without_inventing_a_percentage():
    """A percentage of an unknown denominator is worse than admitting the size is unknown."""
    out, clock = io.StringIO(), _Clock()
    meter = DownloadProgress("cell.json", None, stream=out, now=clock)
    clock.t += 1.0
    meter.advance(3_000_000)

    line = out.getvalue()
    assert "3.0 MB" in line
    assert "%" not in line


def test_a_restart_rewinds_the_count():
    """A server that ignores Range re-sends what was already counted; without a rewind the meter
    would report more than the file's size."""
    out, clock = io.StringIO(), _Clock()
    meter = DownloadProgress("cell.json", 8_000_000, stream=out, now=clock)
    meter.advance(4_000_000)
    meter.advance(-4_000_000)                 # what run_api_client sends on a restart
    clock.t += 1.0
    meter.advance(8_000_000)

    assert meter.done == 8_000_000


# ------------------------------------------------------------------ #
# One meter for a whole fetch, driven by several workers at once
# ------------------------------------------------------------------ #

def test_bytes_from_many_threads_are_all_counted():
    """A fetch's artifacts download concurrently into ONE meter, so the accumulator is contended.
    Contention here is structural, not timed: a barrier releases every worker at the same point."""
    meter = DownloadProgress("run", 8_000, stream=io.StringIO(), now=_Clock())
    ready = threading.Barrier(8, timeout=10)

    def work():
        ready.wait()
        for _ in range(1000):
            meter.advance(1)

    threads = [threading.Thread(target=work) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert meter.done == 8000


def test_concurrent_draws_emit_whole_lines():
    """Two workers composing a line and writing escape sequences on one stream interleave into
    rows no later frame repairs. Every write must therefore arrive intact."""
    out, clock = _Tty(), _Clock()
    meter = DownloadProgress("run", 800_000, stream=out, now=clock, files=8)
    ready = threading.Barrier(8, timeout=10)

    def work():
        ready.wait()
        for _ in range(10):
            clock.t += 1.0
            meter.advance(10_000)
        meter.file_done()

    threads = [threading.Thread(target=work) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    meter.finish()

    drawn = [row for row in re.split(r"\033\[\d*[AJ]", out.getvalue()) if row.strip()]
    assert drawn and all("files" in row and "%" in row for row in drawn)


def test_a_multi_file_fetch_shows_how_many_have_landed():
    out, clock = io.StringIO(), _Clock()
    meter = DownloadProgress("run", 4_000_000, stream=out, now=clock, files=1541)
    for _ in range(3):
        clock.t += 1.0
        meter.advance(1_000_000)
        meter.file_done()
    meter.finish()

    assert "3/1541 files" in out.getvalue()


def test_a_single_file_transfer_renders_exactly_as_it_always_did():
    """`files` defaults to 0 and a count of 1/1 is noise, so one transfer's line is unchanged."""
    out, clock = io.StringIO(), _Clock()
    DownloadProgress("cell.json", 4_000_000, stream=out, now=clock, files=1).finish()

    assert "files" not in out.getvalue()


def test_a_rewound_transfer_never_reports_negative_bytes():
    """A resumed download whose server ignored `Range` restarts the body, and `_download_direct`
    rewinds the meter by what it already counted. The accumulator stays honest; the display does
    not show a negative size."""
    out, clock = io.StringIO(), _Clock()
    meter = DownloadProgress("cell.json", 4_000_000, stream=out, now=clock)
    meter.advance(1_000_000)
    clock.t += 1.0
    meter.advance(-1_000_000)
    meter.finish()

    assert "-" not in out.getvalue().replace("—", "")


def test_a_fetch_whose_listing_lacks_a_size_shows_no_bar():
    """An invented percentage is worse than admitting the size is not known."""
    out, clock = io.StringIO(), _Clock()
    meter = DownloadProgress("run", 0, stream=out, now=clock, files=12)
    clock.t += 1.0
    meter.advance(50_000)
    meter.finish()

    text = out.getvalue()
    assert "%" not in text and "0/12 files" in text
