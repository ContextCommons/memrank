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
"""How a command ends, as data: done, needs your action, or broken -- and what to do next.

A run used to end on whatever the last exception said, so a finished run whose judging was
refused by Anthropic closed on a red "bug in memrank". An :class:`Outcome` is the ending as the
reader needs it: one line saying where they are, what happened in plain words, numbered steps
with each command on its own line, the run's link, and the raw detail from another system kept
at the bottom. :mod:`memrank.term.outcome` is the one renderer; this module holds no styling, so
the same outcome reads the same with or without a terminal.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Kind(Enum):
    """The three looks an ending has."""

    #: Everything asked for was done.
    DONE = "done"
    #: Nothing in memrank broke; the user has a step to take (a key, a login, their agent).
    ACTION = "action"
    #: Something in memrank itself failed, and it should be reported.
    BROKEN = "broken"


@dataclass(frozen=True)
class Step:
    """One thing to do: a sentence, then the commands that do it, each on its own line."""

    text: str
    commands: tuple[str, ...] = ()


@dataclass(frozen=True)
class Link:
    """A call to action with a URL: ``View results: <url>``."""

    label: str
    url: str


@dataclass(frozen=True)
class Details:
    """Raw words from another system -- a provider's payload, an HTTP error -- kept verbatim.

    ``source`` names the system (``Anthropic``, ``the memrank API``), so the reader knows whose
    words they are reading. Never a key: callers pass only what that system said.
    """

    source: str
    lines: tuple[str, ...]


@dataclass(frozen=True)
class Outcome:
    """One command's ending.

    Attributes:
        kind: Which of the three looks it takes.
        title: The line saying where the reader is (``Judging couldn't run``).
        achieved: What did get done, shown as a check above the title when the ending is only
            partly a success (``Your agent answered all 500 questions``).
        identity: ``(label, value)`` rows naming whose and which -- a run's organisation and
            id -- shown first among the facts and highlighted, so they cannot be misread.
        facts: ``(label, value)`` rows under the title -- a finished run's score and the rest.
        happened: What happened, in plain words.
        steps: What to do, in order.
        note: One more thing worth knowing, after the steps.
        saved: Where the run's files are on this machine.
        link: The run's page.
        details: Raw detail from another system, shown last and dimmed.
    """

    kind: Kind
    title: str
    achieved: str | None = None
    identity: tuple[tuple[str, str], ...] = ()
    facts: tuple[tuple[str, str], ...] = ()
    happened: str | None = None
    steps: tuple[Step, ...] = ()
    note: str | None = None
    saved: str | None = None
    link: Link | None = None
    details: Details | None = None

