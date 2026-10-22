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
"""

from __future__ import annotations

import math
from typing import Any

import jmespath

from memrank.connect.base import AgentError, Answer


class ReplyReader:
    """Reads the answer, and the agent's retrieval time when its spec names one, from a reply."""

    def __init__(self, answer: str, retrieval_ms: str | None = None) -> None:
        self.answer_path, self.retrieval_path = answer, retrieval_ms
        self._answer = jmespath.compile(answer)
        self._retrieval = None if retrieval_ms is None else jmespath.compile(retrieval_ms)

    def read(self, payload: Any, where: str) -> Answer:
        """The answer in ``payload``; ``where`` names the reply in an error."""
        answer = self._answer.search(payload)
        if not isinstance(answer, str):
            raise AgentError(f"{self.answer_path!r} found {answer!r} in {where}, not an answer "
                             "string")
        if self._retrieval is None:
            return Answer(answer)
        return Answer(answer, _milliseconds(self._retrieval.search(payload),
                                            self.retrieval_path, where))


def _milliseconds(value: Any, path: str | None, where: str) -> float | None:
    if value is None:
        return None  # not reported for this answer
    number = isinstance(value, (int, float)) and not isinstance(value, bool)
    if not number or not math.isfinite(value) or value < 0:
        raise AgentError(f"{path!r} found {value!r} in {where}; the retrieval time must be a "
                         "non-negative number of milliseconds, or absent")
    return float(value)
