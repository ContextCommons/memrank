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
"""What a provider call was billed, as the provider's own response says, for an agent to report.

A ``Completer`` returns only text, so the senders record each successful call's usage here
instead, into whatever :func:`collecting` block the calling thread has open; with none open (the
judge, every other caller) recording does nothing. An agent wraps one step's model calls in
``collecting("reader")`` and reports what it holds (:class:`~memrank.service.protocol.TokenUsage`).

A response that carries no usage leaves the collection incomplete: the agent then reports
nothing for the step rather than a partial count that would read as the whole.
"""

from __future__ import annotations

import contextlib
import contextvars
from collections.abc import Iterator
from dataclasses import dataclass, field

from memrank.service.protocol import TokenUsage


@dataclass
class Collected:
    """The usage of the calls made inside one :func:`collecting` block."""

    component: str
    entries: list[TokenUsage] = field(default_factory=list)
    #: False once a call's response carried no usage to record.
    complete: bool = True

    def reported(self) -> list[TokenUsage] | None:
        """The entries to report, or None when any call's usage is unknown."""
        return list(self.entries) if self.complete else None


_current: contextvars.ContextVar[Collected | None] = contextvars.ContextVar(
    "provider_usage", default=None)


@contextlib.contextmanager
def collecting(component: str) -> Iterator[Collected]:
    """Collect the usage of every provider call made in this block, as ``component``'s."""
    collected = Collected(component)
    token = _current.set(collected)
    try:
        yield collected
    finally:
        _current.reset(token)


def record(provider: str, model: str, input_tokens: int | None,
           output_tokens: int | None) -> None:
    """Record one call's billed tokens; ``input_tokens`` None means the response gave none."""
    collected = _current.get()
    if collected is None:
        return
    if input_tokens is None:
        collected.complete = False
        return
    collected.entries.append(TokenUsage(component=collected.component, provider=provider,
                                        model=model, input_tokens=input_tokens,
                                        output_tokens=output_tokens))


def record_openai(provider: str, model: str, body: object) -> None:
    """Record a Chat Completions response's ``usage`` (``prompt_tokens`` already counts cached
    input)."""
    usage = body.get("usage") if isinstance(body, dict) else None
    if not isinstance(usage, dict) or not isinstance(usage.get("prompt_tokens"), int):
        record(provider, model, None, None)
        return
    completion = usage.get("completion_tokens")
    record(provider, model, usage["prompt_tokens"],
           completion if isinstance(completion, int) else None)
