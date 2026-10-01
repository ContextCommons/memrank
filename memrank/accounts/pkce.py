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
"""PKCE (RFC 7636) -- the S256 binding between an authorization request and its redemption.

Why this exists rather than a bare authorization code: the code travels back through the
user's browser to a loopback port, where any other local process could in principle race for
it. PKCE makes the code alone worthless -- redeeming it also requires the ``code_verifier``,
which the CLI generated, never sent on the outbound request, and holds only in memory.

Only S256 is implemented. RFC 7636 also defines ``plain``, which offers no protection at all
and exists for clients that cannot compute SHA-256; supporting it here would add a
downgrade an attacker could ask for.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import secrets

# 32 bytes -> 43 base64url characters, exactly the RFC 7636 section 4.1 minimum length.
_VERIFIER_BYTES = 32


def _b64url(raw: bytes) -> str:
    """base64url without padding, as RFC 7636 section 4.2 requires."""
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def make_verifier() -> str:
    """A fresh ``code_verifier``. Held by the CLI in memory; never written to disk."""
    return _b64url(secrets.token_bytes(_VERIFIER_BYTES))


def challenge_for(verifier: str) -> str:
    """The ``code_challenge`` to send with the authorization request: S256 of ``verifier``."""
    return _b64url(hashlib.sha256(verifier.encode()).digest())


def verify(verifier: str, challenge: str) -> bool:
    """Whether ``verifier`` is the one that produced ``challenge``.

    Compared with ``compare_digest`` so the check does not leak, through timing, how much of
    a guessed verifier was correct.
    """
    if not verifier:
        return False
    return hmac.compare_digest(challenge_for(verifier), challenge)
