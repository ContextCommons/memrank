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
"""Reading an answer out of an agent's structured reply, by JMESPath (https://jmespath.org).

Shared by the HTTP connector, whose ask response is JSON, and the command connector when its
spec says the ask command prints JSON. The answer must be a string. The retrieval time is
optional: a spec that names no path for it never reads one, and a reply where the path finds
nothing reported none for that answer. A value that is there but not a non-negative number is
the agent's error, never read as zero or skipped.

The token usage is optional the same way, on an ask reply and on a feed reply: a list of entries,
one per paid component or billed call (:class:`~memrank.service.protocol.TokenUsage`)::

    [{"component": "reader", "provider": "anthropic", "model": "claude-haiku-4-5",
      "input_tokens": 1834, "output_tokens": 12},
     {"component": "embedder", "provider": "openai", "model": "text-embedding-3-small",
      "input_tokens": 9}]

Absent is "not reported"; ``[]`` reports that no paid component was called. Anything else that is
not such a list is the agent's error, naming the reply.
"""

from __future__ import annotations

import math
from typing import Any

import jmespath
from pydantic import TypeAdapter, ValidationError

from memrank.connect.base import AgentError, Answer
from memrank.service.protocol import TokenUsage

_ENTRIES = TypeAdapter(list[TokenUsage])


class UsageReader:
    """Reads the agent's reported token usage from a reply, where its spec says it is."""

    def __init__(self, usage: str | None) -> None:
        self.path = usage
        self._usage = None if usage is None else jmespath.compile(usage)

    def read(self, payload: Any, where: str) -> list[TokenUsage] | None:
        """The usage in ``payload``; None when the spec names no path or the reply has none."""
        if self._usage is None:
            return None
        value = self._usage.search(payload)
        if value is None:
            return None  # not reported for this step
        try:
            return _ENTRIES.validate_python(value)
        except ValidationError as exc:
            raise AgentError(
                f"{self.path!r} found {value!r} in {where}; the usage must be a list of "
                "{component, provider, model, input_tokens, output_tokens} entries with "
                f"non-negative whole token counts, or absent ({exc.error_count()} problems: "
                f"{exc.errors()[0]['msg']})") from exc


def combined(parts: list[list[TokenUsage] | None]) -> list[TokenUsage] | None:
    """One step's usage from its replies (a feed of several requests): every entry reported,
    or None when no reply reported any."""
    reported = [part for part in parts if part is not None]
    return None if not reported else [entry for part in reported for entry in part]


class ReplyReader:
    """Reads the answer, and the agent's retrieval time and token usage when its spec names
    them, from a reply."""

    def __init__(self, answer: str, retrieval_ms: str | None = None,
                 usage: str | None = None) -> None:
        self.answer_path, self.retrieval_path = answer, retrieval_ms
        self._answer = jmespath.compile(answer)
        self._retrieval = None if retrieval_ms is None else jmespath.compile(retrieval_ms)
        self.usage = UsageReader(usage)

    def read(self, payload: Any, where: str) -> Answer:
        """The answer in ``payload``; ``where`` names the reply in an error."""
        answer = self._answer.search(payload)
        if not isinstance(answer, str):
            raise AgentError(f"{self.answer_path!r} found {answer!r} in {where}, not an answer "
                             "string")
        retrieval = None if self._retrieval is None else _milliseconds(
            self._retrieval.search(payload), self.retrieval_path, where)
        return Answer(answer, retrieval, self.usage.read(payload, where))


def _milliseconds(value: Any, path: str | None, where: str) -> float | None:
    if value is None:
        return None  # not reported for this answer
    number = isinstance(value, (int, float)) and not isinstance(value, bool)
    if not number or not math.isfinite(value) or value < 0:
        raise AgentError(f"{path!r} found {value!r} in {where}; the retrieval time must be a "
                         "non-negative number of milliseconds, or absent")
    return float(value)
