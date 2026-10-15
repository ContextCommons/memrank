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
"""TypeSafe System One over ``httpx``: one request of typed questions, its answers, cached.

Written here rather than taken from the TypeSafe SDK for the reasons the ``openai`` adapter is:
one POST and its errors is all a judge needs. ``429`` is a rate limit -- its ``retry-after`` is
honoured through a :class:`~memrank.rate_limit.RateLimitGate` every worker shares -- and ``529``
is overload, retried with exponential backoff, as are other transient answers. Anything else is
a :class:`~memrank.judging.providers.ProviderError`; there is no fallback to another judge.

The cache stores the response under a hash of who answered (provider, endpoint, model), the
question set's version and the canonical request, so a rerun replays its verdicts. A fresh call
is not documented to be bit-identical, which is why the replay matters.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from memrank import rate_limit
from memrank.atomic_json import staging_path
from memrank.judging.providers.errors import ProviderError
from memrank.judging.providers.spec import ResolvedJudge

#: Where System One answers, below the provider's ``api_base``.
SYSTEMONE_PATH = "/v1/systemone"
#: TypeSafe's request id header, quoted in an error so a failure can be traced with them.
REQUEST_ID_HEADER = "x-typesafe-request-id"

READ_TIMEOUT_S = 120.0
#: Rate limited (429), overloaded (529), timed out (408) and other server-side failures are
#: worth retrying; a refusal of the key or the request is not.
_RETRY_LIMITED = 429
_ATTEMPTS = 8
_BACKOFF_BASE_S = 1.0


def _is_transient(status: int) -> bool:
    return status == 408 or status >= 500


@dataclass(frozen=True)
class Decision:
    """One System One response: who answered, every typed answer, and what it read."""

    model: str
    answers: dict[str, Any]
    usage: dict[str, Any]
    request_id: str | None


def error_message(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text.strip()[:300] or response.reason_phrase
    detail = body.get("detail") if isinstance(body, dict) else None
    if isinstance(detail, dict) and detail.get("message"):
        return str(detail["message"])
    return str(detail or body)[:300]


def _refusal(response: httpx.Response, provider: str) -> ProviderError:
    return ProviderError(error_message(response), provider=provider,
                         status_code=response.status_code,
                         request_id=response.headers.get(REQUEST_ID_HEADER),
                         retry_after=rate_limit.retry_after_from_headers(response.headers))


def _decision(response: httpx.Response, provider: str) -> Decision:
    """The response as a :class:`Decision`; one that is not a System One answer raises."""
    try:
        body = response.json()
        decision = Decision(model=str(body["model"]), answers=dict(body["answers"]),
                            usage=dict(body.get("usage") or {}),
                            request_id=response.headers.get(REQUEST_ID_HEADER))
    except (ValueError, KeyError, TypeError) as exc:
        raise ProviderError(f"the reply is not a System One answer: {response.text[:300]!r}",
                            provider=provider, status_code=response.status_code) from exc
    return decision


class JevClient:
    """Sends typed questions to one System One model, counted, cached and rate-limit gated.

    Thread-safe: one client serves every judge worker of a run, so the gate it holds is the
    run's and a 429 seen by one worker pauses them all.
    """

    def __init__(self, target: ResolvedJudge, api_key: str, *, cache_dir: Path | None,
                 transport: httpx.BaseTransport | None = None,
                 gate: rate_limit.RateLimitGate | None = None,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.target = target
        self._cache_dir = cache_dir
        self._gate = gate or rate_limit.RateLimitGate()
        self._sleep = sleep
        self._client = httpx.Client(base_url=target.api_base, transport=transport,
                                    timeout=httpx.Timeout(READ_TIMEOUT_S),
                                    headers={"Authorization": f"Bearer {api_key}"})
        self._lock = threading.Lock()
        self._requests = 0

    @property
    def requests(self) -> int:
        """Requests that reached TypeSafe and were answered; cache hits are not counted."""
        return self._requests

    def decide(self, state: Any, questions: dict[str, Any], *, question_set: str) -> Decision:
        """The answers to ``questions`` about ``state``, from the cache when it holds them."""
        body = {"model": self.target.model, "state": state, "questions": questions}
        path = self._cache_path(body, question_set)
        if path is not None and path.exists():
            return Decision(**json.loads(path.read_text(encoding="utf-8")))
        decision = self._send(body)
        with self._lock:
            self._requests += 1
        if path is not None:
            _store(path, decision)
        return decision

    def _cache_path(self, body: dict[str, Any], question_set: str) -> Path | None:
        if self._cache_dir is None:
            return None
        canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        named = (f"{self.target.provider}|{self.target.api_base}|{self.target.model}|"
                 f"{question_set}|{canonical}")
        return self._cache_dir / f"typesafe-{hashlib.sha256(named.encode()).hexdigest()}.json"

    def _send(self, body: dict[str, Any]) -> Decision:
        """One answered request, retrying a rate limit, an overload or a transient failure."""
        provider = self.target.provider
        for attempt in range(_ATTEMPTS):
            self._gate.wait()
            response = self._client.post(SYSTEMONE_PATH, json=body)
            if response.is_success:
                return _decision(response, provider)
            status = response.status_code
            retryable = status == _RETRY_LIMITED or _is_transient(status)
            if not retryable or attempt == _ATTEMPTS - 1:
                raise _refusal(response, provider)
            backoff = min(_BACKOFF_BASE_S * 2 ** attempt, rate_limit.MAX_PAUSE_S)
            if status == _RETRY_LIMITED:
                # Account-wide: every worker waits, for as long as TypeSafe asked.
                stated = rate_limit.retry_after_from_headers(response.headers)
                self._gate.pause(stated if stated is not None else backoff)
            else:
                self._sleep(backoff)
        raise AssertionError("unreachable: the last attempt returns or raises")


def _store(path: Path, decision: Decision) -> None:
    """Staged then renamed and owner-only, as the text judge's cache is: a reader that caught a
    partial write would read it as a verdict."""
    tmp = staging_path(path)
    tmp.write_text(json.dumps(decision.__dict__, sort_keys=True), encoding="utf-8")
    try:
        tmp.chmod(0o600)
    except OSError:
        pass
    tmp.replace(path)
