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
"""Local credentials pool -- the operator's key "wallet".

A single flat plaintext JSON file (``${MEMRANK_CONFIG_DIR}/secrets.json``, default
``~/.config/memrank``) holding the API keys you *have*, reused by any engine's requirements. A value
may be a literal key or an ``${ENV_VAR}`` reference (resolved by :mod:`memrank.config`). Written
``0600`` and trivially inspectable (``cat`` it).

POC store: no encryption, local only. Secure backends (OS keyring, external managers) are a deferred,
drop-in evolution behind the same read API.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from memrank.atomic_json import write_json

_DIR_ENV = "MEMRANK_CONFIG_DIR"
_FILENAME = "secrets.json"


def config_dir() -> Path:
    """The memrank config directory: ``$MEMRANK_CONFIG_DIR`` else ``~/.config/memrank``."""
    return Path(os.environ.get(_DIR_ENV) or (Path.home() / ".config" / "memrank"))


def store_path() -> Path:
    """Absolute path to ``secrets.json``."""
    return config_dir() / _FILENAME


def _load() -> dict[str, Any]:
    path = store_path()
    if not path.exists():
        return {"secrets": {}}
    data = json.loads(path.read_text(encoding="utf-8"))
    data.setdefault("secrets", {})
    return data


def _save(data: dict[str, Any]) -> None:
    write_json(store_path(), data, mode=0o600, sort_keys=True, trailing_newline=True)


def get(name: str) -> str | None:
    """The stored value for secret ``name`` (a literal or ``${VAR}`` reference), or ``None``."""
    return _load()["secrets"].get(name)


def put(name: str, value: str) -> None:
    """Store (or overwrite) secret ``name``."""
    data = _load()
    data["secrets"][name] = value
    _save(data)


def delete(name: str) -> bool:
    """Remove secret ``name``; return whether it existed."""
    data = _load()
    existed = data["secrets"].pop(name, None) is not None
    if existed:
        _save(data)
    return existed


def names() -> list[str]:
    """Sorted names of every stored secret."""
    return sorted(_load()["secrets"])
