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
"""The ``openai`` adapter: Chat Completions over ``httpx``, for OpenAI and compatible servers.

Written here rather than taken from the ``openai`` package or LiteLLM: one POST and its errors is
all a judge needs, and neither dependency is worth its supply-chain and release churn for that.
No structured-output mode is requested -- some router backends answer JSON mode with ``{ }`` --
so judges keep memrank's own strict-JSON prompt and parser.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any

import httpx

from memrank.judging.providers.errors import ProviderError
from memrank.judging.providers.spec import ResolvedJudge
from memrank.judging.providers.types import Send

#: Above the router's own ~60 s gateway timeout on slow backends, so that one is what is reported.
READ_TIMEOUT_S = 120.0
#: Transient answers retried, as the ``anthropic`` SDK does for the Anthropic path.
_TRANSIENT = frozenset({408, 409})
_RETRIES = 2
_BACKOFF_BASE_S = 1.0
#: The code OpenAI puts on a 429 that means "no credit", which waiting does not fix.
_NO_QUOTA = "insufficient_quota"


def _is_transient(status: int) -> bool:
    return status in _TRANSIENT or status >= 500


def _error_of(response: httpx.Response) -> dict[str, Any]:
    try:
        body = response.json()
    except ValueError:
        return {"message": response.text.strip() or response.reason_phrase}
    said = body.get("error") if isinstance(body, dict) else None
    return said if isinstance(said, dict) else {"message": str(body)}


def _refusal(response: httpx.Response, provider: str) -> ProviderError:
    from memrank import rate_limit

    error = _error_of(response)
    no_quota = _NO_QUOTA in (error.get("code"), error.get("type"))
    return ProviderError(
        str(error.get("message") or response.reason_phrase), provider=provider,
        status_code=response.status_code, request_id=response.headers.get("x-request-id"),
        retry_after=rate_limit.retry_after_from_headers(response.headers),
        quota_exhausted=no_quota)


def _text(response: httpx.Response, provider: str) -> str:
    """The reply's text; a reply with none raises with its ``finish_reason``, never parses empty."""
    try:
        choice = response.json()["choices"][0]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise ProviderError(f"the reply is not a chat completion: {response.text[:300]!r}",
                            provider=provider, status_code=response.status_code) from exc
    content = (choice.get("message") or {}).get("content")
    if not isinstance(content, str) or not content.strip():
        raise ProviderError(
            f"the reply has no text (finish_reason={choice.get('finish_reason')!r}); a "
            f"reasoning model may have spent the whole output budget",
            provider=provider, status_code=response.status_code)
    return content


def openai_send(target: ResolvedJudge, api_key: str, *, max_tokens: int,
                transport: httpx.BaseTransport | None = None,
                sleep: Callable[[float], None] = time.sleep) -> Send:
    """A sender for ``target``; ``transport`` and ``sleep`` are for tests."""
    client = httpx.Client(base_url=target.api_base, timeout=httpx.Timeout(READ_TIMEOUT_S),
                          headers={"Authorization": f"Bearer {api_key}"}, transport=transport)

    def send(model: str, system: str, user: str, sampling: Mapping[str, Any]) -> str:
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": user}]
        body = {"model": model, "messages": messages, target.max_tokens_field: max_tokens,
                **sampling}
        for attempt in range(_RETRIES + 1):
            response = client.post("/chat/completions", json=body)
            if response.is_success:
                return _text(response, target.provider)
            if not _is_transient(response.status_code) or attempt == _RETRIES:
                raise _refusal(response, target.provider)
            sleep(_BACKOFF_BASE_S * 2 ** attempt)
        raise AssertionError("unreachable: the last attempt returns or raises")

    return send
