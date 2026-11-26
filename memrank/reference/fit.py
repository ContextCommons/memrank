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
"""Cutting a history that is longer than the reader's input limit, from the oldest end.

The full-context agent sends the whole history first. Only when the provider refuses it with its
own count -- ``prompt is too long: N tokens > M maximum``, the words Anthropic uses directly and
that Bedrock passes back through an OpenAI-compatible router -- does it drop the oldest turns, in
proportion to the excess, and ask again. The provider's count is the only one used: memrank does
not tokenize, so a history that fits is never touched and the cut is the same for the same
history and question. Any other refusal is not a length the agent can act on and is raised.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from memrank.connect.template import SESSION_DATE

#: How the provider says the prompt is over its input limit, with its own token counts.
TOO_LONG = re.compile(r"prompt is too long: (\d+) tokens > (\d+) maximum")

#: How many times a refused history is cut and asked again before the refusal is raised.
MAX_FIT_ROUNDS = 3

#: The fraction of the provider's maximum left unused: room for the answer (the reader's 512
#: output tokens) and for the cut text counting at a slightly different rate than the whole.
FIT_MARGIN = 0.02


@dataclass(frozen=True)
class Overflow:
    """The provider's own count of a refused prompt and its maximum, in tokens."""

    tokens: int
    maximum: int


def overflow(exc: BaseException) -> Overflow | None:
    """The token counts in a "prompt is too long" refusal, or ``None`` for any other failure."""
    found = TOO_LONG.search(str(exc))
    if found is None:
        return None
    return Overflow(tokens=int(found.group(1)), maximum=int(found.group(2)))


def chars_to_drop(over: Overflow, prompt_chars: int) -> int:
    """How many characters to drop so the prompt fits, at the refused prompt's chars per token."""
    target = over.maximum * (1 - FIT_MARGIN)
    excess_tokens = over.tokens - target
    return max(1, math.ceil(excess_tokens * prompt_chars / over.tokens))


def is_header(message: dict[str, Any]) -> bool:
    """A session's dated header, which belongs to the turns after it rather than being one."""
    return (message.get("role") == "system"
            and str(message.get("content") or "").startswith(SESSION_DATE))


def cut_oldest(history: list[dict[str, Any]], chars: int) -> list[dict[str, Any]] | None:
    """``history`` without its oldest whole turns totalling at least ``chars`` characters.

    The session header of the first kept turn is kept with it. ``None`` when nothing would be left.
    """
    dropped = 0
    for start, message in enumerate(history):
        if dropped >= chars and message.get("role") != "system":
            header = [m for m in history[:start] if is_header(m)][-1:]
            return header + history[start:]
        if message.get("role") != "system":
            dropped += len(str(message.get("content") or "")) + 1  # and its line break
    return None


def turns(history: list[dict[str, Any]]) -> tuple[int, int]:
    """How many turns ``history`` holds and how many characters they carry."""
    said = [str(m.get("content") or "") for m in history if m.get("role") != "system"]
    return len(said), sum(len(text) for text in said)
