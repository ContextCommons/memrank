"""Semantic terminal styling and the two output streams -- the one chokepoint the CLI emits through.

Color is presentation only: every escape code goes through ``typer.style``/``typer.echo``,
so Click strips it when the stream is not a TTY, and the ``NO_COLOR`` convention
(https://no-color.org) is honored here, at the chokepoint. Nothing styled here may be
stored (heartbeats, receipts, ``--json``) -- style at the echo site, never in the data.
Fragments destined for width-padded table cells must be padded before styling;
``state()`` does this itself via its ``width`` argument.

**Which stream, and why it is decided here.** stdout carries the command's ANSWER -- run ids,
``--json``, log content, the table a listing was asked for. Everything else -- progress,
confirmations, hints, warnings, errors -- is stderr, so that piping a command yields its answer
and nothing else. Two emitters, :func:`out` and :func:`say`, so the choice is made once per line
by naming what the line IS, and no caller reaches for ``typer.echo`` and picks a stream by
accident. ``tests/term/test_output_streams.py`` enumerates the package to keep it that way.
"""
from __future__ import annotations

import logging
import os
import sys
from collections.abc import Callable, Sequence

import typer

#: Run-state -> color, one map for `runs ls`, `ps`, `status`, and `watch`.
#: `unknown` and `—` are deliberately absent: an unstyled cell reads as "no claim".
_STATE_COLORS: dict[str, str] = {
    "submitted": typer.colors.YELLOW,
    "queued": typer.colors.YELLOW,
    "running": typer.colors.CYAN,
    "done": typer.colors.GREEN,
    "failed": typer.colors.RED,
    "stale": typer.colors.MAGENTA,
}


def wants_color() -> bool:
    """The NO_COLOR convention: any non-empty value disables styling entirely.

    Read at call time, not import: tests and wrapper scripts set it per invocation.

    Public because it is not only colour that the variable silences: `watch`'s in-place redraw
    emits cursor control, which is terminal decoration by the same argument, and
    :func:`memrank.term.progress.supports_redraw` asks here rather than re-reading the
    environment -- one gate, so NO_COLOR cannot silence half of each.
    """
    return not os.environ.get("NO_COLOR")


def supports_boxes() -> bool:
    """Whether stdout can carry box drawing -- the gate that makes a listing a TABLE.

    Borders are a HUMAN affordance, applied at render time like truncation and relative
    timestamps. 12-Factor #8 says never to border a table, and the reason it gives is piping:
    ``grep``, ``wc`` and ``awk`` want one plain row per entry. Both are satisfied by deciding
    here -- a terminal gets the boxed table, a pipe gets the same rows unadorned -- which is the
    same split ``gh`` documents for its own output.

    Deliberately NOT gated on :func:`wants_color`. NO_COLOR is about colour
    (https://no-color.org); a monochrome bordered table is still a correct rendering of it.
    That differs from :func:`memrank.term.progress.supports_redraw`, which DOES ask
    ``wants_color`` -- and the difference is real: redraw emits cursor MOTION, which a reader
    who asked for plain output did not ask to watch, while a box is static glyphs.

    ``TERM=dumb`` is refused separately because such a terminal is a TTY that cannot draw
    the characters.
    """
    return bool(getattr(sys.stdout, "isatty", lambda: False)()) and os.environ.get("TERM") != "dumb"


def decorates(stream: object) -> bool:
    """Whether ``stream`` gets colour and clickable links, rather than the same words plain.

    Stricter than :func:`wants_color`, whose escapes Click strips from a pipe on its own: a
    hyperlink escape is not one Click knows to strip, so this asks the stream itself. CI logs
    are recorded rather than read live, and get the plain words even from a pseudo-terminal.
    """
    tty = bool(getattr(stream, "isatty", lambda: False)())
    return (tty and wants_color() and not os.environ.get("CI")
            and os.environ.get("TERM") != "dumb")


def link(url: str, stream: object = None) -> str:
    """``url`` blue and clickable (OSC 8) where ``stream`` is a terminal; plain elsewhere."""
    if not decorates(sys.stdout if stream is None else stream):
        return url
    return typer.style(f"\033]8;;{url}\033\\{url}\033]8;;\033\\", fg=typer.colors.BLUE)


def terminal_width(default: int = 100) -> int:
    """The width a boxed table may fill. One source, so table and panel agree.

    Capped rather than uncapped: a maximised terminal is 300 columns wide and a table stretched
    across all of them puts a run's id and its score at opposite ends of the desk.
    """
    try:
        columns = os.get_terminal_size(sys.stdout.fileno()).columns
    except (OSError, ValueError, AttributeError):
        return default
    return max(40, min(columns, 160))


def _fragment(text: str, *, fg: str | None = None, dim: bool = False, bold: bool = False) -> str:
    """Every styled string passes through here -- the single gate NO_COLOR closes."""
    if not wants_color():
        return text
    # None, not False: Click renders `dim=False` as "normal intensity" (SGR 22), which cancels
    # the bold it has just emitted, so every bold fragment used to render plain.
    return typer.style(text, fg=fg, dim=dim or None, bold=bold or None)


def out(text: str = "", *, nl: bool = True) -> None:
    """Write one line of the command's ANSWER to stdout.

    What a caller redirects: ids, ``--json``, log content, the rows of a listing. If a line
    would still be worth having with the terminal gone, it belongs here. ``nl=False`` for
    content that already carries its own newlines -- a log file printed whole.
    """
    typer.echo(text, nl=nl)


def say(text: str = "") -> None:
    """Write one line of NARRATION to stderr -- progress, confirmation, a hint about what to type.

    Narration is about producing the answer rather than being it, so it must not land in the
    file a caller redirected stdout into. ``error``/``warn``/``note`` are its labelled forms.
    """
    typer.echo(text, err=True)


def error(message: str) -> None:
    """Echo ``error: <message>`` to stderr in red.

    No stream parameter: a surface that "owns stdout errors" is a surface whose failures land in
    the file a caller was capturing an answer into.
    """
    typer.echo(_fragment(f"error: {message}", fg=typer.colors.RED), err=True)


def step(statement: str, commands: Sequence[str] = ()) -> None:
    """Tell the user what they need to do: the statement, then each command on its own line.

    The missing-step voice, deliberately unlike :func:`error`: no red and no ``error:`` label,
    because a step the user has not taken yet is not something that broke. Commands are indented
    and bold so they can be found and copied whole.
    """
    if statement:
        say(statement)
    for command in commands:
        say(f"  {bold(command)}")


def warn(message: str) -> None:
    """Echo ``warning: <message>`` to stderr in yellow."""
    typer.echo(_fragment(f"warning: {message}", fg=typer.colors.YELLOW), err=True)


def note(message: str) -> None:
    """Echo ``note: <message>`` to stderr, dimmed."""
    typer.echo(_fragment(f"note: {message}", dim=True), err=True)


class _NarrationHandler(logging.Handler):
    """Render a `memrank.*` log record as terminal narration: WARNING+ via :func:`warn`,
    the rest via :func:`say`."""

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if record.levelno >= logging.WARNING:
            warn(message)
        else:
            say(message)


def install_log_bridge() -> None:
    """Give the library's log records a terminal voice -- the CLI's half of a contract.

    Library modules that narrate outside the eval loop's observer (a dataset download in
    ``benchmarks``, a survivable adapter warning) speak stdlib ``logging`` and nothing
    else, so an embedding application hears them through ordinary logging config -- or not
    at all. The CLI installs this bridge at startup so a terminal user sees the same lines
    as before. Idempotent: commands re-enter the root callback freely.
    """
    logger = logging.getLogger("memrank")
    if any(isinstance(h, _NarrationHandler) for h in logger.handlers):
        return
    logger.setLevel(logging.INFO)
    logger.addHandler(_NarrationHandler())


def good(text: str) -> str:
    """Style a success fragment (``✓``, ``done``, ``HIT``) green."""
    return _fragment(text, fg=typer.colors.GREEN)


def bad(text: str) -> str:
    """Style a failure fragment (``✗``, ``failed``, ``MISS``) red."""
    return _fragment(text, fg=typer.colors.RED)


def caution(text: str) -> str:
    """Style an attention fragment (``!`` advisories) yellow."""
    return _fragment(text, fg=typer.colors.YELLOW)


#: The look of WHOSE and WHICH: the organisation a run is recorded to and the run's own id.
#: One constant, so the run's opening lines and its ending (:mod:`memrank.term.outcome`) match.
IDENTITY_LOOK: dict[str, object] = {"fg": typer.colors.YELLOW, "bold": True}

#: What every line ``memrank run`` says about itself opens with, as W&B's ``wandb:`` does.
PREFIX = "memrank:"


def identity(text: str) -> str:
    """An identity the reader must not misread -- an org slug, a run id, a login. Bold yellow,
    so which organisation a run lands in is the first thing the eye finds."""
    return _fragment(text, fg=typer.colors.YELLOW, bold=True)


def prefixed(text: str) -> str:
    """``memrank: <text>``, the prefix bold blue: a line the tool says about the run, set apart
    from the agent's and the evaluation's own output."""
    return f"{_fragment(PREFIX, fg=typer.colors.BLUE, bold=True)} {text}"


def dim(text: str) -> str:
    """Dim a secondary fragment (placeholders, sources)."""
    return _fragment(text, dim=True)


def bold(text: str) -> str:
    """Embolden a header row."""
    return _fragment(text, bold=True)


def heading(text: str) -> str:
    """A section or column heading. Bold rather than coloured: headings are structure, and a
    colour here would compete with the colours that carry meaning below them."""
    return _fragment(text, bold=True)


def label(text: str) -> str:
    """The NAME of a thing, in a label/value pair. Dimmed so the value it introduces is what the
    eye lands on -- in a metrics block the labels are scaffolding, not content."""
    return _fragment(text, dim=True)


def value(text: str) -> str:
    """A measured value. Left at the terminal's default foreground on purpose.

    Everything around it is dimmed or coloured, so plain IS the emphasis -- and it stays readable
    on light and dark themes alike, which a fixed bright colour would not.
    """
    return text


def accent(text: str) -> str:
    """Marks what the reader should look at FIRST -- the active stage, the current row."""
    return _fragment(text, fg=typer.colors.CYAN)


def unit(text: str) -> str:
    """A unit or qualifier trailing a value (``s/doc``, ``~4m left``). Dimmed: it qualifies the
    number without competing with it."""
    return _fragment(text, dim=True)


def pad(text: str, width: int, styler: Callable[[str], str] = lambda s: s, *,
        align: str = "<") -> str:
    """Pad to ``width``, THEN style. The only correct order, made the only convenient one.

    ANSI escapes count toward ``len()``, so styling before padding silently destroys column
    alignment -- and every substring-matching test still passes, because the characters are all
    there in the wrong places. The module docstring has warned about this since it was written;
    `state()` was the one place that obeyed it. This is that discipline as a function, so a caller
    cannot get the order wrong by forgetting a rule.

    ``align`` takes a format spec (``<`` or ``>``). It exists because right-aligned columns are
    the ones most likely to be padded by hand afterwards -- which is the same bug wearing a
    different hat, and it was made in this file's first caller.
    """
    return styler(f"{text:{align}{width}}")


def state_styler(value: str) -> Callable[[str], str]:
    """A styler bound to ``value``'s colour, safe to apply to a PADDED cell.

    :func:`state` looks its colour up from the text it is handed, so it cannot colour
    ``"done     "`` -- the lookup misses and the cell silently loses its colour. A table pads
    before it styles (it must; ANSI counts toward ``len()``), so it needs the colour decided
    from the raw value and applied later. Same map, both callers.
    """
    color = _STATE_COLORS.get(value)
    if color is None:
        return lambda text: text
    return lambda text: _fragment(text, fg=color)


def state(value: str, *, width: int = 0) -> str:
    """Color a run state, padding BEFORE styling so ``len()``-sized columns stay aligned.

    Args:
        value: The state token (``running``, ``done``, ...). Unknown states pass
            through unstyled.
        width: Left-pad to this width first; the ANSI codes wrap the padded cell, so
            hand-rolled tables that compute widths with ``len()`` on the raw value keep
            their columns.
    """
    padded = f"{value:<{width}}" if width else value
    color = _STATE_COLORS.get(value)
    return _fragment(padded, fg=color) if color else padded
