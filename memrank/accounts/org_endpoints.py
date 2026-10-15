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
"""An organisation's named LLM endpoints, from the client side (ATO-2503).

The API calls that read and write ``/orgs/{org}/endpoints``, and the one call made to an endpoint
itself: the ``/models`` check that ``memrank endpoints add`` makes before saving anything.

An endpoint's base URL may be an internal host. Nothing here prints it, puts it in an error or
logs it: every message names the endpoint by its name.
"""
from __future__ import annotations

import re
from typing import Any

import httpx

from memrank.accounts.run_secrets import api_message, session_rejected
from memrank.errors import ActionRequired, MemrankError

#: What an endpoint may be called: short, lowercase, and free of the `:` a judge spec splits on.
NAME_PATTERN = re.compile(r"^[a-z][a-z0-9-]{0,31}$")

#: Provider names a judge spec already means something by. An endpoint called `anthropic` would
#: make `anthropic:claude-haiku-4-5` ambiguous, and `unknown`/`custom` label recorded judges
#: (plan 2026-10-15-judge-providers, Steps 1 and 2).
RESERVED_NAMES = frozenset({"anthropic", "openai", "typesafe", "custom", "unknown"})

#: The adapters an endpoint may speak through.
ADAPTERS = ("openai", "anthropic")

#: An organisation secret is spent as an environment variable, so its name must be one.
_KEY_REF_PATTERN = re.compile(r"^[A-Z_][A-Z0-9_]*$")

#: The ``anthropic-version`` header Anthropic's models listing requires.
_ANTHROPIC_VERSION = "2023-06-01"

#: Statuses that mean "this endpoint has no models listing", as opposed to a refusal or an outage.
_NO_LISTING = (404, 405)

#: How long the one-off ``/models`` check may take; a listing answers in well under this.
_CHECK_TIMEOUT_SECONDS = 30.0


class EndpointError(MemrankError):
    """An endpoint could not be saved, read or checked. Never swallowed."""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


def name_problem(name: str) -> str | None:
    """Why ``name`` cannot name an endpoint, or None when it can."""
    if not NAME_PATTERN.match(name):
        return ("an endpoint name is 1-32 characters of a-z, 0-9 and '-', "
                "starting with a letter")
    if name in RESERVED_NAMES:
        return f"{name!r} is a built-in provider name; choose another"
    return None


def base_url_problem(api_base: str) -> str | None:
    """Why ``api_base`` is not a base URL, or None. Never repeats the value: an error is the
    likeliest place a private URL leaks."""
    if not re.match(r"^https?://\S+$", api_base):
        return "the base URL must be an http:// or https:// URL"
    return None


def key_ref_problem(key_ref: str) -> str | None:
    """Why ``key_ref`` cannot name an org secret, or None."""
    if not _KEY_REF_PATTERN.match(key_ref):
        return "key_ref is the NAME of an org secret, e.g. ATOMIC_ROUTER_API_KEY"
    return None


def _refusal(response, org: str, doing: str, owner_command: str) -> Exception:
    """The error for a refused endpoints call: sign in again, ask an owner, or the API's own."""
    if response.status_code == 401:
        return session_rejected()
    if response.status_code == 403:
        return ActionRequired(f"Only an owner of {org} can {doing}. Ask an owner to run:",
                              owner_command)
    return EndpointError(f"could not {doing} for {org!r}: {api_message(response)}",
                         status=response.status_code)


def list_endpoints(http, org: str) -> list[dict[str, Any]]:
    """``org``'s endpoints: name, adapter, key_ref, models_listing, id, revision, fingerprint."""
    response = http.get(f"/orgs/{org}/endpoints")
    if response.status_code >= 400:
        raise _refusal(response, org, "list its endpoints", f"memrank endpoints ls --org {org}")
    return response.json()


def save_endpoint(http, org: str, entry: dict[str, Any]) -> dict[str, Any]:
    """Register ``entry`` for ``org``, or replace the entry of that name. Answers its public view."""
    response = http.post(f"/orgs/{org}/endpoints", json=entry)
    if response.status_code >= 400:
        raise _refusal(response, org, "register endpoints",
                       f"memrank endpoints add {entry['name']} --org {org} ...")
    return response.json()


def delete_endpoint(http, org: str, name: str) -> None:
    """Remove ``org``'s endpoint ``name``."""
    response = http.delete(f"/orgs/{org}/endpoints/{name}")
    if response.status_code >= 400:
        raise _refusal(response, org, "remove endpoints",
                       f"memrank endpoints rm {name} --org {org}")


def resolve_endpoint(http, org: str, name: str) -> dict[str, Any]:
    """What a client needs to call ``org``'s endpoint ``name``, ``api_base`` included.

    Raises:
        EndpointError: With ``status=404`` when the org has no such endpoint, so a caller can
            name the endpoints it does have.
    """
    response = http.get(f"/orgs/{org}/endpoints/{name}/resolved")
    if response.status_code >= 400:
        raise _refusal(response, org, "read its endpoints", f"memrank endpoints ls --org {org}")
    return response.json()


def models_request(adapter: str, api_base: str, key: str) -> tuple[str, dict[str, str]]:
    """The URL and headers of ``adapter``'s models listing, as its vendor documents it."""
    if adapter == "anthropic":
        return (f"{api_base}/v1/models",
                {"x-api-key": key, "anthropic-version": _ANTHROPIC_VERSION})
    return f"{api_base}/models", {"Authorization": f"Bearer {key}"}


def has_models_listing(*, name: str, adapter: str, api_base: str, key: str,
                       transport: httpx.BaseTransport | None = None) -> bool:
    """Whether endpoint ``name`` answers ``GET /models`` with ``key``; free on every vendor.

    Returns:
        True when the listing answered, False when the endpoint has none (404 or 405).

    Raises:
        EndpointError: The key was refused (401, 403), the endpoint failed, or it could not be
            reached. The message names the endpoint and never its URL.
    """
    url, headers = models_request(adapter, api_base, key)
    try:
        with httpx.Client(timeout=_CHECK_TIMEOUT_SECONDS, transport=transport) as client:
            response = client.get(url, headers=headers)
    except httpx.HTTPError as exc:
        # The exception's own text can carry the URL, so only its kind is reported.
        raise EndpointError(f"could not reach endpoint {name!r} ({type(exc).__name__}); check "
                            f"the base URL and that this machine can reach it") from exc
    if response.status_code in (401, 403):
        raise EndpointError(f"endpoint {name!r} refused the key ({response.status_code}); "
                            f"nothing was saved", status=response.status_code)
    if response.status_code in _NO_LISTING:
        return False
    if response.status_code >= 400:
        raise EndpointError(f"endpoint {name!r} answered its models listing with "
                            f"{response.status_code}; nothing was saved",
                            status=response.status_code)
    return True
