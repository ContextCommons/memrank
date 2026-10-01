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
"""Standing user preferences -- the defaults that make routing flags disappear.

Distinct from :mod:`memrank.config`, which answers *what is this deployment wired to* (database
URLs, buckets, the API address) from an env file services override wholesale and that holds
credentials. This answers *what does this user want by default*: per-user rather than
per-checkout, written by a command rather than by hand, and never holding a secret. A preference
command rewriting a file full of DSNs is the accident this separation prevents.

Values resolve environment -> file -> shipped default, and every read reports **which** answered.
The environment wins because ``MEMRANK_API_URL=... memrank ...`` is how the runbooks work and must
keep working; a setting whose source is invisible is one the user cannot reason about when it
surprises them.

The store is ``config.json`` beside the local wallet in
:func:`memrank.secrets.wallet.config_dir`, written with the same atomic ``0600`` replace.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from memrank import config
from memrank.atomic_json import write_json
from memrank.errors import MemrankError
from memrank.secrets import wallet

_FILENAME = "config.json"

#: Where a resolved value came from, in precedence order.
ENV, FILE, DEFAULT = "env", "file", "default"


@dataclass(frozen=True)
class Setting:
    """One configurable key: how it is spelled everywhere, and what it means."""

    key: str
    env: str
    default: str | None
    help: str
    #: The value is an os.pathsep-separated list of directories. Stored absolute: a relative
    #: entry means "this directory, from here" when typed, but read back later it would mean
    #: "wherever the process happens to be standing" -- the cwd-dependence runs_root() documents.
    search_path: bool = False


#: Every key the CLI honours. A key absent here cannot be set -- a typo that stored happily
#: would be a preference that silently never applies.
SETTINGS: tuple[Setting, ...] = (
    Setting("defaults.org", "MEMRANK_ORG", None,
            "org a submission runs under (and is recorded against)"),
    # Shipped `none`, NOT `cloud`: logged out, everything must still work locally. `auth login`
    # writes `cloud`, so a machine that signed in is cloud-first and a fresh clone is not.
    Setting("defaults.on", "MEMRANK_ON", "none",
            "where submissions run: none | local | cloud"),
    # Defaulted, unlike the others, because an installed CLI has no env file: without it the
    # first command a new user runs fails on an environment variable they have never heard of.
    # This is not the no-fallbacks rule bending -- that rule stops a MISSING setting being
    # guessed at (a localhost default would send a credential to whatever is listening). Which
    # platform a build talks to is a decision this build makes, and `config ls` shows it as
    # `default` alongside its source, so nothing is hidden and both overrides still win.
    #
    # PRODUCTION since 2026-08-25, when api.memrank.ai went live. It was api-staging until
    # then because there was no other option. The consequence worth knowing rather than
    # discovering: an already-installed CLI keeps talking to staging until it is upgraded
    # (the API's cli_contract check bounds what that can break, but does not redirect it).
    #
    # `submit --on cloud` against prod works, and is BYOK: a submitted run spends the
    # credential of the ORG it is recorded against, resolved from accounts.org_secrets and
    # injected by ARN (memrank/api/runs.py, `"credentials": "org"`), so an org with no
    # provider key stored cannot launch one. The shared /memrank/prod/* provider keys are
    # placeholders and nothing on the submission path reads them; do not read them as the
    # thing a cloud run spends (tech-debt.md).
    Setting("api.url", "MEMRANK_API_URL", "https://api.memrank.ai",
            "base URL of the memrank API this CLI talks to"),
    # Off by default: the OS keychain is better at rest, but memrank runs as a Python entry
    # point, so the macOS dialog names `python` rather than the tool and returns after every
    # reinstall (the ACL binds to the interpreter). Available to anyone who wants it and will
    # accept that; see memrank/accounts/credentials.py for the full trade.
    Setting("auth.keyring", "MEMRANK_KEYRING", "false",
            "store the session token in the OS keyring instead of a 0600 file"),
    # On by default, and it costs a signed-out machine nothing: the hook is silent without a
    # session, so local-first use never notices it. Signed in, every finished run reaches the
    # org -- which is the point of a shared universe, and the reason `runs sync` is a
    # reconciler rather than the way runs normally get there.
    Setting("sync.auto", "MEMRANK_SYNC_AUTO", "true",
            "sync each finished local run into the org universe when signed in"),
    # Where a user's own target descriptors live, beside the translators they launch. A setting
    # rather than a `targets link` verb: this is the primitive, and a command would be sugar over
    # it -- worth choosing once hand-wiring has shown what the ergonomic should be. Separated by
    # os.pathsep, following PATH/PYTHONPATH, because a search path is what this is.
    Setting("targets.path", "MEMRANK_TARGETS_PATH", None,
            "extra directories to read target descriptors from (os.pathsep-separated)",
            search_path=True),
)

_BY_KEY = {s.key: s for s in SETTINGS}


class UnknownSetting(MemrankError, KeyError):
    """A key that is not in the table. Raised rather than stored.

    Keeps ``KeyError`` so lookup call sites read naturally, and gains ``MemrankError`` so an
    escape prints as one line rather than a traceback.
    """


def setting(key: str) -> Setting:
    """The :class:`Setting` for ``key``, or :class:`UnknownSetting` naming what exists."""
    try:
        return _BY_KEY[key]
    except KeyError as exc:
        raise UnknownSetting(
            f"unknown setting {key!r}; known keys: {', '.join(sorted(_BY_KEY))}") from exc


def store_path() -> Path:
    """Absolute path to the settings file."""
    return wallet.config_dir() / _FILENAME


def _load() -> dict[str, str]:
    path = store_path()
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _save(values: dict[str, str]) -> None:
    """Write the whole file atomically, ``0600`` -- the wallet's pattern, for the same reason."""
    write_json(store_path(), values, mode=0o600, sort_keys=True, trailing_newline=True)


def resolve(key: str) -> tuple[str | None, str]:
    """The effective value of ``key`` and the source that supplied it.

    Returns:
        ``(value, source)`` where source is ``env``, ``file`` or ``default``. A ``None`` value
        with source ``default`` means nothing has answered this key at all.
    """
    spec = setting(key)
    # Through config, so a value in the CLI's own env file counts exactly as it does everywhere
    # else -- and so this module is not one more place reading os.environ for settings.
    config._ensure_dotenv()
    from_env = os.environ.get(spec.env)
    if from_env:
        return from_env, ENV
    stored = _load().get(key)
    if stored is not None:
        return stored, FILE
    return _shipped_default(spec), DEFAULT


def _shipped_default(spec: Setting) -> str | None:
    """The default this install ships with.

    The standalone build is built for one site and names that site's API in its record
    (``tools/internal/binary/build.py --api-url``), so a staging build reaches the staging API
    with no setup. A Python install, or a build whose record names no API, ships the table's.
    """
    if spec.key == "api.url":
        from memrank.provenance.install import standalone_build

        build = standalone_build()
        if build is not None and build.get("api_url"):
            return build["api_url"]
    return spec.default


def get(key: str) -> str | None:
    """The effective value of ``key``, without its source."""
    return resolve(key)[0]


def _absolute_search_path(value: str) -> str:
    """Each entry expanded and resolved against the cwd -- fixed at the moment of the write."""
    parts = [part for part in value.split(os.pathsep) if part]
    return os.pathsep.join(str(Path(part).expanduser().resolve()) for part in parts)


def put(key: str, value: str) -> str:
    """Store ``value`` for ``key`` and return what was stored. Refuses keys outside the table."""
    spec = setting(key)
    if spec.search_path:
        value = _absolute_search_path(value)
    values = _load()
    values[key] = value
    _save(values)
    return value


def resolved() -> list[tuple[Setting, str | None, str]]:
    """Every setting with its effective value and source, in table order."""
    return [(spec, *resolve(spec.key)) for spec in SETTINGS]
