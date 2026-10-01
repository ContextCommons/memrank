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
"""The declarative HTTP connector: any agent with an HTTP API, described rather than coded.

Each of reset, feed and ask is a templated request (see :mod:`memrank.connect.template`). The
answer is pulled out of the ask response with a JMESPath expression (https://jmespath.org), the
query language AWS's CLI uses, so ``choices[0].message.content`` means what it means there.

``feed: chat`` replays every past turn through the ask request, one at a time, for an agent
that has no endpoint to load history into; the replies are discarded.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any, Literal

import httpx
import jmespath
from pydantic import BaseModel, ConfigDict, Field

from memrank.adapters.errors import body_excerpt, safe_json
from memrank.connect import template
from memrank.connect.base import (
    AgentError,
    AgentRateLimited,
    AgentUnreachable,
    ConnectorConfigError,
)
from memrank.rate_limit import RateLimitGate, retry_after_seconds
from memrank.service.protocol import Step

#: A body is shown in an error up to this length; agents can return pages of HTML on a 500.
_ERROR_BODY_CHARS = 300


class Auth(BaseModel):
    """A credential sent on every request, read from the runner's environment by name."""

    model_config = ConfigDict(extra="forbid")

    env: str
    header: str = "Authorization"
    prefix: str = "Bearer "


class Call(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: str = "POST"
    path: str
    body: Any = None


class FeedCall(Call):
    #: How much history one request carries: a turn, a session, or the whole case.
    per: Literal["turn", "session", "case"] = "session"


class AskCall(Call):
    answer: str = Field(description="JMESPath to the answer in the JSON response")


class HttpSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_url: str
    auth: Auth | None = None
    timeout_s: float = Field(default=600.0, gt=0)
    vars: dict[str, Any] = Field(default_factory=dict)
    reset: Call | None = None
    feed: FeedCall | Literal["chat"]
    ask: AskCall


def _headers(auth: Auth | None) -> dict[str, str]:
    if auth is None:
        return {}
    value = os.environ.get(auth.env)
    if not value:
        raise ConnectorConfigError(f"the agent spec reads its credential from ${auth.env}, "
                                   "which is not set; export it and retry")
    return {auth.header: f"{auth.prefix}{value}"}


def _refusal(method: str, path: str, response: httpx.Response) -> AgentError:
    """The agent's error answer, whole: its status and its own words, redacted and capped."""
    message = f"{method} {path} returned {response.status_code}: {body_excerpt(response)}"
    body = safe_json(response)
    if response.status_code == 429:
        return AgentRateLimited(message, body=body,
                                # it reads an exception's ``response``; this has only that
                                retry_after_s=retry_after_seconds(
                                    SimpleNamespace(response=response)))  # type: ignore[arg-type]
    return AgentError(message, status=response.status_code, body=body)


#: How many times one request is sent while the agent answers 429. A 429 is a request refused
#: unprocessed, so sending it again feeds nothing twice; the bound is what stops a limit that
#: never lifts from spinning forever -- the last refusal is then recorded as the step's failure.
RATE_LIMIT_ATTEMPTS = 6

#: The pause after a 429 that names none, doubling per attempt (1, 2, 4, 8, 16 s).
RATE_LIMIT_BACKOFF_S = 1.0


class HttpConnector:
    """Carries steps to an agent over HTTP, as its spec describes.

    Safe to share between the run's lanes: one ``httpx.Client``, and one :class:`RateLimitGate`,
    so a 429 seen by any lane pauses every lane -- the limit is the agent's, not one lane's
    (:mod:`memrank.rate_limit` argues why the pause must be shared).
    """

    def __init__(self, spec: HttpSpec, transport: httpx.BaseTransport | None = None,
                 gate: RateLimitGate | None = None) -> None:
        self.spec = spec
        self._answer = jmespath.compile(spec.ask.answer)
        self._http = httpx.Client(base_url=spec.base_url, timeout=spec.timeout_s,
                                  headers=_headers(spec.auth), transport=transport)
        self.gate = gate or RateLimitGate()

    def close(self) -> None:
        self._http.close()

    def _send(self, call: Call, variables: dict[str, Any]) -> httpx.Response:
        """One request, waiting out the agent's rate limit on the shared gate.

        Per request, not per step: a feed is several requests, and repeating the whole step after
        the third was refused would feed the first two sessions twice.
        """
        for attempt in range(RATE_LIMIT_ATTEMPTS):
            self.gate.wait()
            try:
                return self._send_once(call, variables)
            except AgentRateLimited as limited:
                if attempt == RATE_LIMIT_ATTEMPTS - 1:
                    raise
                self.gate.pause(limited.retry_after_s or RATE_LIMIT_BACKOFF_S * 2 ** attempt)
        raise AssertionError("unreachable: the last attempt returns or raises")

    def _send_once(self, call: Call, variables: dict[str, Any]) -> httpx.Response:
        path = template.render(call.path, variables)
        body = template.render(call.body, variables)
        try:
            response = self._http.request(call.method, path, json=body)
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise AgentUnreachable(f"could not connect to {self.spec.base_url}: {exc}") from exc
        except httpx.HTTPError as exc:
            raise AgentError(f"{call.method} {path} failed after sending: {exc!r}") from exc
        if response.is_error:
            raise _refusal(call.method, path, response)
        return response

    def reset(self, step: Step) -> None:
        if self.spec.reset is not None:
            self._send(self.spec.reset, template.base_vars(step, self.spec.vars))

    def feed(self, step: Step) -> None:
        sessions = step.sessions or []
        feed = self.spec.feed
        if feed == "chat":
            for turn in (t for s in sessions for t in s.turns):
                self._ask(self._replayed(step, template.turn_vars(step, turn, self.spec.vars)))
        elif feed.per == "case":
            self._send(feed, template.case_vars(step, self.spec.vars))
        elif feed.per == "session":
            for session in sessions:
                self._send(feed, template.session_vars(step, session, self.spec.vars))
        else:
            for turn in (t for s in sessions for t in s.turns):
                self._send(feed, template.turn_vars(step, turn, self.spec.vars))

    def _replayed(self, step: Step, turn: dict[str, Any]) -> dict[str, Any]:
        """Ask-shaped variables whose question is one past turn, attributed."""
        line = f"{turn['speaker']}: {turn['text']}"
        return {**template.base_vars(step, self.spec.vars), "question": line,
                "question_id": None, "timestamp": turn["timestamp"],
                "messages": [{"role": "user", "content": line}]}

    def ask(self, step: Step) -> str:
        return self._ask(template.ask_vars(step, self.spec.vars))

    def _ask(self, variables: dict[str, Any]) -> str:
        response = self._send(self.spec.ask, variables)
        try:
            payload = response.json()
        except ValueError as exc:
            raise AgentError(f"ask response is not JSON: {response.text[:_ERROR_BODY_CHARS]}") \
                from exc
        answer = self._answer.search(payload)
        if not isinstance(answer, str):
            raise AgentError(f"{self.spec.ask.answer!r} found {answer!r} in the ask response, "
                             "not an answer string")
        return answer
