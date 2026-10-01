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
"""Reading a quit key without wrecking the terminal.

`watch` blocks for up to 45 minutes, so detaching has to be possible. Reading one keypress means
cbreak mode, which is process-global state that OUTLIVES the process -- so what these cover is not
the reading but the three refusals: not a terminal, not in the foreground, and restore-always.
"""
from __future__ import annotations

import pytest

from memrank.term import key_input


class FakeStream:
    """A stdin stand-in. ``tty`` and ``foreground`` are what the module gates on."""

    def __init__(self, *, tty: bool = True, keys: str = "", fd: int = 99):
        self.tty, self._keys, self._fd = tty, list(keys), fd

    def isatty(self):
        return self.tty

    def fileno(self):
        return self._fd

    def read(self, n):
        return self._keys.pop(0) if self._keys else ""


@pytest.fixture
def foreground(monkeypatch):
    """Pretend this process owns the terminal."""
    monkeypatch.setattr(key_input.os, "getpgrp", lambda: 1)
    monkeypatch.setattr(key_input.os, "tcgetpgrp", lambda fd: 1)


def test_a_pipe_is_never_touched(foreground):
    """Under CI stdin is a pipe or closed; setting terminal modes there is meaningless."""
    assert not key_input.can_listen(FakeStream(tty=False))


def test_a_background_job_is_never_touched(monkeypatch):
    """THE one that is easy to omit, because it never fires interactively. A backgrounded
    `memrank watch ... &` calling tcsetattr takes SIGTTOU and the job STOPS -- the command appears to
    hang, which is worse than having no quit key at all."""
    monkeypatch.setattr(key_input.os, "getpgrp", lambda: 1)
    monkeypatch.setattr(key_input.os, "tcgetpgrp", lambda fd: 2)  # the terminal belongs elsewhere

    assert not key_input.can_listen(FakeStream())


def test_a_terminal_we_own_is_listened_to(foreground):
    assert key_input.can_listen(FakeStream())


def test_a_stream_with_no_fileno_is_refused(foreground):
    """pytest's captured stdin, and anything else that is not a real file descriptor."""
    class NoFileno:
        def isatty(self):
            return True

        def fileno(self):
            raise OSError("not a real stream")

    assert not key_input.can_listen(NoFileno())


#: The lflag bit the fake terminal below carries, and the index termios keeps lflag at.
_ECHO_BIT, _LFLAG = 0b1000, 3


def _fake_termios(restored: list, saved: list):
    """A terminal that records every mode written to it.

    Models the fields `raw_mode` actually touches -- a double that returned an opaque token would
    pass while the code under it did arithmetic on the wrong index.
    """
    return type("termios", (), {
        "tcgetattr": staticmethod(lambda fd: list(saved)),
        "tcsetattr": staticmethod(lambda fd, when, state: restored.append(list(state))),
        "TCSADRAIN": 1, "ECHO": _ECHO_BIT})


def test_the_terminal_is_restored_even_when_the_body_raises(foreground, monkeypatch):
    """The failure this guards outlives the test process: a shell left in cbreak with no echo
    looks broken, and quitting the program does not fix it."""
    restored: list = []
    saved = [0, 0, 0, _ECHO_BIT, 0, 0, []]
    monkeypatch.setitem(__import__("sys").modules, "tty",
                        type("tty", (), {"setcbreak": staticmethod(lambda fd: None)}))
    monkeypatch.setitem(__import__("sys").modules, "termios", _fake_termios(restored, saved))

    with pytest.raises(RuntimeError):
        with key_input.raw_mode(FakeStream()):
            raise RuntimeError("the watch blew up")

    assert restored[-1] == saved, "the mode the terminal arrived in is the mode it leaves in"


def test_keys_are_read_but_never_echoed(foreground, monkeypatch):
    """cbreak leaves ECHO on, so every key pressed during a watch was painted into whatever row
    the cursor was on -- the middle of the progress block -- shifting the rows the redraw counts."""
    restored: list = []
    saved = [0, 0, 0, _ECHO_BIT, 0, 0, []]
    monkeypatch.setitem(__import__("sys").modules, "tty",
                        type("tty", (), {"setcbreak": staticmethod(lambda fd: None)}))
    monkeypatch.setitem(__import__("sys").modules, "termios", _fake_termios(restored, saved))

    with key_input.raw_mode(FakeStream()) as listening:
        assert listening

    assert restored[0][_LFLAG] & _ECHO_BIT == 0, "echo is off for the duration of the block"
    assert restored[-1][_LFLAG] & _ECHO_BIT, "and back on afterwards"


#: A stream that always has a byte ready -- so these assert key HANDLING, not select's timing.
READY = lambda stream, timeout: True        # noqa: E731
NEVER = lambda stream, timeout: False       # noqa: E731


def test_q_ends_the_wait(foreground):
    assert key_input.wait_for_quit(5, FakeStream(keys="q"), readable=READY)


def test_uppercase_q_too(foreground):
    assert key_input.wait_for_quit(5, FakeStream(keys="Q"), readable=READY)


def test_no_keypress_at_all_just_times_out(foreground):
    assert not key_input.wait_for_quit(5, FakeStream(), readable=NEVER)


def test_escape_is_not_a_quit_key(foreground):
    """ESC is the first byte of every arrow-key sequence. Binding it would let a stray arrow end
    a 45-minute watch."""
    assert "\033" not in key_input.QUIT_KEYS


def test_another_key_does_not_end_the_wait(foreground):
    """Consumed and ignored; the wait continues for its remaining time rather than returning
    early, which would turn any keystroke into a busy poll."""
    assert not key_input.wait_for_quit(0.05, FakeStream(keys="xyz"), readable=READY)


def test_with_nothing_listening_it_is_just_a_sleep():
    """A CI watch keeps its cadence; it simply cannot be interrupted."""
    import time

    start = time.monotonic()
    result = key_input.wait_for_quit(0.05, FakeStream(tty=False), listening=False)

    assert result is False
    assert time.monotonic() - start >= 0.04
