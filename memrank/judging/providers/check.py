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
"""Checking, before anything runs, that a key reaches the judge's model -- for free where it can.

A built-in provider is asked for the one model (``GET /v1/models/{model}`` on Anthropic,
``GET /models/{model}`` on OpenAI); a named endpoint is asked for its models listing, which must
list the model. Neither is billed. An endpoint registered without a listing is checked with one
1-token completion instead -- recorded on the endpoint when it was registered, never decided by
retrying here.

A refusal about the key or the model is an answer and comes back as words; anything else -- an
outage, an unreachable host -- is raised. No message here carries an endpoint's URL.
"""

from __future__ import annotations

import httpx

from memrank.accounts.org_endpoints import models_request
from memrank.errors import MemrankError
from memrank.judging.providers.errors import ProviderError
from memrank.judging.providers.spec import ResolvedJudge, provider_label

#: Statuses that answer "not this key" or "not this model", rather than "not now".
_REFUSALS = (400, 401, 403, 404)

#: A models lookup answers in well under this.
_CHECK_TIMEOUT_S = 30.0

#: The one-token prompt an endpoint without a models listing is checked with.
_PING = ("Reply with the word OK.", "OK?")


class JudgeUnreachable(MemrankError):
    """The judge's provider could not be reached to check the key; nothing ran."""


def _lookup(target: ResolvedJudge, key: str) -> tuple[str, dict[str, str]]:
    """The URL and headers that ask ``target``'s provider about its model, or list them all."""
    url, headers = models_request(target.adapter, target.api_base, key)
    return (url, headers) if target.endpoint is not None else (f"{url}/{target.model}", headers)


def _listed(response: httpx.Response, model: str) -> bool:
    try:
        rows = response.json().get("data")
    except (ValueError, AttributeError):
        return False
    return isinstance(rows, list) and any(
        isinstance(row, dict) and row.get("id") == model for row in rows)


def _ping(target: ResolvedJudge, key: str) -> str | None:
    """One 1-token completion, for an endpoint with no models listing."""
    from memrank.judging.providers import ADAPTERS

    send = ADAPTERS[target.adapter](target, key, max_tokens=1)
    try:
        send(target.model, *_PING, {})
    except ProviderError as exc:
        if exc.status_code not in _REFUSALS:
            raise
        return f"{exc.message} ({exc.status_code})"
    return None


def pasted_key_refusal(target: ResolvedJudge, key: str) -> str | None:
    """Why a key typed at the terminal will not judge with ``target``, or ``None``.

    Anthropic's is one tiny completion, the only check that finds an account with no credit;
    every other provider's is the free model check.
    """
    if target.adapter == "anthropic" and target.endpoint is None:
        from memrank.judging.client import key_refusal

        return key_refusal(target, key)
    return model_refusal(target, key)


def model_refusal(target: ResolvedJudge, key: str, *, models_listing: bool = True,
                  transport: httpx.BaseTransport | None = None) -> str | None:
    """``None`` when ``key`` reaches ``target``'s model, else why not, in a few words.

    Raises:
        JudgeUnreachable: The provider could not be reached; its URL is never in the message.
        ProviderError: The provider answered with something other than a refusal.
    """
    if not models_listing:
        return _ping(target, key)
    url, headers = _lookup(target, key)
    who = provider_label(target.provider)
    try:
        with httpx.Client(timeout=_CHECK_TIMEOUT_S, transport=transport) as client:
            response = client.get(url, headers=headers)
    except httpx.HTTPError as exc:
        # The exception's own text can carry the URL, so only its kind is reported.
        raise JudgeUnreachable(f"could not reach {who} to check the judge "
                               f"({type(exc).__name__})") from exc
    if response.status_code in (401, 403):
        return f"the key was not accepted ({response.status_code})"
    if response.status_code == 404 and target.endpoint is None:
        return f"it has no model {target.model!r}"
    if response.status_code >= 400:
        raise ProviderError(response.reason_phrase or "the models lookup failed",
                            provider=target.provider, status_code=response.status_code)
    if target.endpoint is not None and not _listed(response, target.model):
        return f"its models listing does not include {target.model!r}"
    return None
