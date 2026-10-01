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
"""What every connector is, and the two ways a call to an agent can fail.

The distinction between the two failures is the whole retry policy. :class:`AgentUnreachable`
means the request provably never reached the agent -- the connection was refused or never
established -- so sending it again cannot feed anything twice. :class:`AgentError` means the
agent may have seen the request; it is recorded as a failed step and never retried.
"""

from __future__ import annotations

from typing import Any, Protocol

from memrank.errors import MemrankError
from memrank.service.protocol import Step


class ConnectorConfigError(MemrankError):
    """An agent spec that cannot work as written. Raised before any step runs."""


class AgentUnreachable(MemrankError):
    """The request never reached the agent. Safe to retry."""


class AgentError(MemrankError):
    """The agent was reached and the call failed. Recorded as a failure, never retried.

    ``status`` and ``body`` are the agent's HTTP answer when there was one, kept apart from the
    message so the runner can say whose side a failure is on (:mod:`memrank.loop.explain`)
    without parsing prose.
    """

    def __init__(self, message: str, *, status: int | None = None, body: Any = None) -> None:
        super().__init__(message)
        self.status = status
        self.body = body


class AgentRateLimited(AgentError):
    """The agent answered 429: it refused the request unprocessed, so waiting and sending it
    again is safe. ``retry_after_s`` is the pause it asked for, when it named one."""

    def __init__(self, message: str, *, retry_after_s: float | None, body: Any = None) -> None:
        super().__init__(message, status=429, body=body)
        self.retry_after_s = retry_after_s


class Connector(Protocol):
    """Carries steps to one agent. Each method either succeeds or raises one of the above."""

    def reset(self, step: Step) -> None: ...

    def feed(self, step: Step) -> None: ...

    def ask(self, step: Step) -> str: ...

    def close(self) -> None: ...
