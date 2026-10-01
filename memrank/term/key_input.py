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
"""A quit key for long-running foreground commands, without wrecking the terminal.

`memrank watch` blocks for up to 45 minutes redrawing progress bars, and detaching from one is a
normal thing to want. Reading a single keypress means putting the terminal in cbreak mode, which
is process-global state that OUTLIVES the process -- so the interesting part of this module is not
reading the key, it is the three conditions under which it refuses to try.

**stdin must be a terminal.** Under CI or a pipe there is nothing to read and nothing to set.

**The process must be in the foreground.** A backgrounded ``memrank watch ... &`` that calls
``tcsetattr`` takes SIGTTOU and the job STOPS -- the command appears to hang, which is strictly
worse than having no quit key. The check is cheap and never fires interactively, which is exactly
why it is easy to leave out.

**The terminal is restored on every exit path**, including exceptions. A watch that exits without
restoring leaves the shell in cbreak with no echo: it looks like a broken terminal, and closing
the program does not fix it.

``q`` and ``Q`` only. Not ESC -- it is the first byte of every arrow-key escape sequence, so a
stray arrow would end a 45-minute watch.
"""
from __future__ import annotations

import contextlib
import os
import select
import sys
import time
from collections.abc import Iterator
from typing import Any

#: What ends a watch. ESC is deliberately absent; see the module docstring.
QUIT_KEYS = ("q", "Q")


def can_listen(stream: Any = None) -> bool:
    """Whether a keypress can be read from ``stream`` without harming the session.

    False for a pipe, a closed stdin, a platform without termios, and -- the one that is easy to
    miss -- a process that is not in the foreground of its controlling terminal.
    """
    stream = sys.stdin if stream is None else stream
    try:
        import termios  # noqa: F401  (presence is the check: absent on Windows)

        fileno = stream.fileno()
        if not stream.isatty():
            return False
        # Reading or setting terminal modes from a background job raises SIGTTIN/SIGTTOU, whose
        # default disposition stops the job. Better to have no quit key than a stopped one.
        return os.getpgrp() == os.tcgetpgrp(fileno)
    except Exception:  # noqa: BLE001 - any failure here means "do not touch the terminal"
        return False


@contextlib.contextmanager
def raw_mode(stream: Any = None) -> Iterator[bool]:
    """Put ``stream`` in cbreak for the block, restoring it however the block ends.

    Yields whether the mode was actually entered, so a caller can decide whether to advertise a
    key that may not be listening.
    """
    stream = sys.stdin if stream is None else stream
    if not can_listen(stream):
        yield False
        return
    import termios
    import tty

    fileno = stream.fileno()
    saved = termios.tcgetattr(fileno)
    try:
        tty.setcbreak(fileno)
        # cbreak leaves ECHO ON, so every key pressed during a watch was painted into whatever
        # row the cursor happened to be on -- i.e. into the middle of the progress block, shifting
        # the rows the redraw is counting. `q` is READ here, never displayed; the same goes for
        # the arrow keys and stray typing that a 45-minute watch collects.
        mode = termios.tcgetattr(fileno)
        mode[3] &= ~termios.ECHO          # [3] is lflag
        termios.tcsetattr(fileno, termios.TCSADRAIN, mode)
        yield True
    finally:
        # TCSADRAIN, not TCSANOW: let queued output flush before the mode changes, so a final
        # redraw is not truncated mid-escape-sequence.
        termios.tcsetattr(fileno, termios.TCSADRAIN, saved)


def _readable(stream: Any, timeout: float) -> bool:
    """Whether ``stream`` has a byte waiting within ``timeout``. Injectable so tests need no fd."""
    ready, _, _ = select.select([stream], [], [], timeout)
    return bool(ready)


def wait_for_quit(timeout: float, stream: Any = None, listening: bool = True,
                  readable: Any = _readable) -> bool:
    """Wait up to ``timeout`` seconds; return True if a quit key arrived.

    Replaces a plain sleep so the key answers immediately rather than at the end of a 15-second
    poll interval -- a quit key that takes fifteen seconds reads as one that does not work.

    With no terminal to read (``listening`` False), this IS a plain sleep: the caller's loop keeps
    its cadence and simply cannot be interrupted.

    ``readable`` is injected rather than called directly so a test can decide what is waiting
    instead of racing a real descriptor -- a keypress test that depends on select's timing is a
    flake, and this one WAS one before the seam existed.
    """
    stream = sys.stdin if stream is None else stream
    if not listening:
        time.sleep(timeout)
        return False
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        if not readable(stream, remaining):
            return False
        # A key that is not a quit key is consumed and ignored, and the wait CONTINUES for its
        # remaining time -- returning early would turn any keystroke into a busy poll loop.
        if stream.read(1) in QUIT_KEYS:
            return True
