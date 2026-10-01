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
"""Loading an org's BYOK credentials into a run.

The bridge between "who am I" and "what does this run spend". Every failure here is loud: a
run that started with a *partial* set of an org's credentials would fail deep inside an
engine with an unrelated-looking error, or -- worse -- complete against a fallback key and
publish a number attributed to the wrong configuration.
"""
from __future__ import annotations

from memrank.errors import ActionRequired, MemrankError
from memrank.outcome import Step

#: The command every "sign in first" step names.
SIGN_IN = "memrank auth login"


class OrgSecretsError(MemrankError):
    """Raised when an org's credentials cannot be loaded or saved. Never swallowed.

    ``status`` is the API's HTTP status, so a caller can tell a refusal it has a next step for
    (401, 403) from a failure it does not.
    """

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


def session_rejected() -> ActionRequired:
    """The step when the API refuses the stored session: sign in again."""
    return ActionRequired("Your memrank session was not accepted; it may have expired.",
                          steps=(Step("Sign in again:", (SIGN_IN,)),))


def api_message(response) -> str:
    """The API's message, or its raw body when the shape is not the one we document."""
    try:
        detail = response.json().get("detail")
    except ValueError:
        return response.text
    return detail.get("message", str(detail)) if isinstance(detail, dict) else str(detail)


def save_org_secret(http, org: str, name: str, value: str) -> None:
    """Store ``name`` for ``org`` through the API -- the one path every org key is saved by.

    Raises:
        ActionRequired: The session was refused, or the caller's role may not save keys.
        OrgSecretsError: Any other refusal, with the API's own message -- a credential that did
            not store must not read as though it did.
    """
    response = http.post(f"/orgs/{org}/secrets", json={"name": name, "value": value})
    if response.status_code == 401:
        raise session_rejected()
    if response.status_code == 403:
        raise ActionRequired(f"Only an owner of {org} can save its keys. Ask an owner to run:",
                             f"memrank secrets set {name} --org {org}")
    if response.status_code >= 400:
        raise OrgSecretsError(f"could not store {name} for {org!r}: {api_message(response)}",
                              status=response.status_code)


def load_org_secrets(http, org: str) -> dict[str, str]:
    """Fetch ``org``'s credentials as ``{ENV_VAR: value}``.

    Args:
        http: An authenticated ``httpx.Client``-shaped object bound to the API.
        org: The org slug the run was launched for.

    Returns:
        The org's credentials. May be empty if the org has none stored -- that is a real
        answer, not an error, and the engine's own preflight will say which key is missing.

    Raises:
        OrgSecretsError: If the caller is not signed in, or is not a member of ``org``. Both
            are refusals to guess: continuing would silently spend somebody else's key.
    """
    response = http.get(f"/orgs/{org}/secrets/resolved")
    if response.status_code == 401:
        raise OrgSecretsError("not signed in -- run `memrank auth login`", status=401)
    if response.status_code == 403:
        raise OrgSecretsError(f"you are not a member of org {org!r}, or your role may not read "
                              "its saved credentials (an owner's may)", status=403)
    if response.status_code != 200:
        raise OrgSecretsError(
            f"could not load credentials for {org!r}: {response.status_code} {response.text}",
            status=response.status_code)
    return response.json()
