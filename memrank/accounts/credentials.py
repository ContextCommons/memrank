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
"""Client-side storage for the CLI token.

Resolution order, highest first:

1. ``MEMRANK_TOKEN`` -- CI, containers, and SSH sessions must be able to supply a token without
   anything installed. It wins outright so an override is always possible.
2. A ``0600`` file beside the local wallet.
3. The OS keyring -- only when ``auth.keyring`` is turned on.

**Why the file first, when a keychain is better at rest.** memrank installs as a Python entry
point, so the process asking macOS for the secret is ``python`` -- the dialog names the
interpreter, not the tool, which reads like malware to anyone paying attention. Keychain ACLs
bind to that binary, so every ``uv tool install`` mints a new identity and the prompt returns
(keyring#619). A security control that trains users to click through a scary dialog on every
upgrade is not the control it looks like. ``gh`` gets a friendly, stable prompt because it ships
a signed native binary; we cannot.

So this follows the majority of comparable tools -- ``aws``, ``gcloud``, ``kubectl``, ``npm`` all
keep credentials in permission-protected files -- and leaves the keychain available to anyone who
wants it: ``memrank config set auth.keyring true``.

**What that costs, stated plainly**: at rest the token is readable by anything running as this
user, where the keychain would have prompted. It is a 30-day credential scoped to submitting
evaluations against one org and revocable server-side, and ``secrets.json`` next to it already
holds provider keys under the same 0600 model -- so this is consistent with what memrank already
does, not a new exposure. It is still a real reduction, not a free win.

The keyring is injected so tests never touch the real keychain: a test suite that writes to
a developer's login keyring is one that fails differently on every machine.
"""
from __future__ import annotations

import os
from pathlib import Path

from memrank.errors import MemrankError

#: Namespace under which the token is filed in the OS keyring.
SERVICE = "memrank"
ACCOUNT = "cli-token"


class CredentialError(MemrankError):
    """The credential store exists but would not answer. NOT the same as "not signed in"."""


def _is_missing_backend(exc: BaseException) -> bool:
    """Whether ``exc`` means "there is no keyring on this machine" -- not a failure.

    Matched by exception NAME rather than by importing ``keyring.errors``: keyring is imported
    lazily (a CLI that never authenticates should not pay for it), and both the test suite and
    injected fakes substitute stand-in modules that have no ``errors`` submodule. The default
    when nothing matches is to treat the exception as a real failure, which is the direction
    that tells the truth -- a store that broke should never be read as an empty one.
    """
    return any(base.__name__ == "NoKeyringError" for base in type(exc).__mro__)


def default_path() -> Path:
    """Where the fallback credential file lives, beside the existing local wallet."""
    return Path.home() / ".memrank" / "credentials"


class CredentialStore:
    """Reads and writes the CLI token, preferring the most secure location available."""

    def __init__(self, keyring=None, path: Path | None = None) -> None:
        """Build a store.

        Args:
            keyring: The keyring module (or a stand-in). Passing one is itself a statement that
                this store should use it -- that is how tests and callers opt in explicitly.
                ``None`` means "consult ``auth.keyring``", and the real module is imported only
                if that says yes, so an install that never opts in never loads it.
            path: Credential file location. ``None`` uses :func:`default_path`.
        """
        self._injected_keyring = keyring
        self._path = path or default_path()

    @property
    def _use_keyring(self) -> bool:
        """Whether this store touches the OS keyring at all."""
        if self._injected_keyring is not None:
            return True
        from memrank import settings

        return (settings.get("auth.keyring") or "").strip().lower() in ("1", "true", "yes")

    @property
    def _keyring(self):
        """The keyring module, imported only when something is actually going to use it."""
        if self._injected_keyring is not None:
            return self._injected_keyring
        import keyring as real_keyring

        return real_keyring

    def load(self) -> str | None:
        """The token, or ``None`` if the user is not logged in.

        Raises:
            CredentialError: When a keyring exists but would not answer -- a denied prompt, a
                locked keychain. Distinct from ``None`` on purpose: "nothing is stored" and
                "the store refused" are different facts, and conflating them is how a
                permission problem gets reported as being logged out. `gh` has this bug open
                (cli/cli#13317): a failed keychain read there produces silent unauthenticated
                requests.
        """
        from memrank import config

        from_env = config.cli_token_from_env()
        if from_env:
            return from_env
        if self._use_keyring:
            stored = self._from_keyring()
            if stored:
                return stored
        return self._from_file()

    def save(self, token: str) -> None:
        """Persist ``token``: the 0600 file, or the keyring when ``auth.keyring`` is on."""
        if self._use_keyring:
            try:
                self._keyring.set_password(SERVICE, ACCOUNT, token)
                return
            except Exception:  # noqa: BLE001 -- a keyring that will not take it still leaves
                pass                                                    # the file as an answer
        self._write_file(token)

    def clear(self) -> None:
        """Forget the token everywhere this store could have put it.

        The keyring is cleared even when ``auth.keyring`` is off, unlike every read path. A
        session stored under a previous setting -- or a previous version, when the keyring was
        the default -- would otherwise survive a logout, and a credential the user believes is
        gone is worse than a dialog they asked for by typing `logout`.
        """
        try:
            self._keyring.delete_password(SERVICE, ACCOUNT)
        except Exception:  # noqa: BLE001 -- nothing stored there, or no backend at all
            pass
        self._path.unlink(missing_ok=True)

    def _from_keyring(self) -> str | None:
        try:
            return self._keyring.get_password(SERVICE, ACCOUNT)
        except Exception as exc:  # noqa: BLE001 -- sorted into "absent" and "refused" below
            if _is_missing_backend(exc):
                # A machine with no keyring at all is a supported configuration -- that is what
                # the 0600 file exists for. Fall through to it.
                return None
            raise CredentialError(
                f"the credential store refused to answer ({exc}). Your session may be fine -- "
                f"this machine could not read it. Approve the keychain prompt, or set "
                f"MEMRANK_TOKEN for this shell."
            ) from exc

    def _from_file(self) -> str | None:
        if not self._path.exists():
            return None
        content = self._path.read_text(encoding="utf-8").strip()
        return content or None

    def _write_file(self, token: str) -> None:
        """Write the token 0600, creating the directory if needed.

        The mode is set on the file descriptor at creation rather than chmod'ed afterwards,
        so there is no window in which the file exists with default permissions.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self._path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(token)
