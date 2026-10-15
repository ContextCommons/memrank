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
"""Checking, before anything runs, that a TypeSafe key reaches the Jev model a run names.

``GET /v1/models`` is free and settles the key. It lists only the moving aliases, and TypeSafe
documents that versioned ids are accepted whether or not they are listed, so a versioned model
the listing does not name is settled by the one request that can: a single Noul over a few
words, a few dozen input tokens at TypeSafe's per-token price.
"""

from __future__ import annotations

import httpx

from memrank.judging.providers.errors import ProviderError
from memrank.judging.providers.spec import ResolvedJudge

_CHECK_TIMEOUT_S = 30.0
MODELS_PATH = "/v1/models"
_PROBE = {"state": "OK", "questions": {"ok": {"type": "noul",
                                              "instructions": "Does the state say OK?"}}}


def _listed(response: httpx.Response, model: str) -> bool:
    try:
        rows = response.json().get("models")
    except (ValueError, AttributeError):
        return False
    return isinstance(rows, list) and any(
        isinstance(row, dict) and row.get("name") == model for row in rows)


def _model_refusal(client: httpx.Client, target: ResolvedJudge) -> str | None:
    """``None`` when TypeSafe answers ``target.model``, else why not."""
    from memrank.judging.jev.client import SYSTEMONE_PATH, error_message

    response = client.post(SYSTEMONE_PATH, json={"model": target.model, **_PROBE})
    if response.is_success:
        return None
    if response.status_code in (400, 404, 422):
        return f"it has no model {target.model!r} ({error_message(response)})"
    raise ProviderError(error_message(response), provider=target.provider,
                        status_code=response.status_code)


def jev_refusal(target: ResolvedJudge, key: str,
                transport: httpx.BaseTransport | None = None) -> str | None:
    """``None`` when ``key`` reaches ``target``'s Jev model, else why not, in a few words.

    Raises:
        httpx.HTTPError: TypeSafe could not be reached; the caller names it without the URL.
        ProviderError: TypeSafe answered with something other than a refusal.
    """
    with httpx.Client(base_url=target.api_base, timeout=_CHECK_TIMEOUT_S, transport=transport,
                      headers={"Authorization": f"Bearer {key}"}) as client:
        listing = client.get(MODELS_PATH)
        if listing.status_code in (401, 403):
            return f"the key was not accepted ({listing.status_code})"
        if listing.status_code >= 400:
            raise ProviderError(listing.reason_phrase or "the models lookup failed",
                                provider=target.provider, status_code=listing.status_code)
        if _listed(listing, target.model):
            return None
        return _model_refusal(client, target)
