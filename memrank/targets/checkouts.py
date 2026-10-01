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
"""Where this machine keeps the checkout each source target binds.

A source target evaluates a working tree -- the engine as it exists on someone's disk right now.
That location cannot live in the descriptor: memrank knows nothing about the repository holding
the descriptor, the engine repository knows nothing about being evaluated, and the checkout may sit
anywhere. An absolute path committed to the file is true on exactly one machine, which is how a
collaborator's first run ended in ``this machine cannot run here`` naming a path in someone else's
home directory.

KEYED BY TARGET REF, not by repository. Two arms of the *same* repository -- a second worktree, a
commit under comparison -- are two targets, and keying on the repository would collapse them onto one
path, which is precisely the variable under test. Ansible reaches the same conclusion with
``host_vars/<host>.yml``, Nix with ``--override-input <name> <path>``, Bazel with
``--override_repository=<name>=<path>``: the value is keyed on the instance, and the field keeps its
plain name. It also means a descriptor copied to a new name gets a new key by construction rather
than by remembering to rename a variable.

A value may be a literal path or an ``${ENV_VAR}`` reference, resolved through
:mod:`memrank.config` -- the same contract :mod:`memrank.secrets.wallet` offers, so an operator who
would rather keep the path in the environment can store the reference and get both.

Local and uncommitted by construction, like ``go.work`` and ``.cargo/config.toml``, which exist for
this same reason in other ecosystems.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from memrank.atomic_json import write_json

_DIR_ENV = "MEMRANK_CONFIG_DIR"
_FILENAME = "checkouts.json"


def config_dir() -> Path:
    """The memrank config directory: ``$MEMRANK_CONFIG_DIR`` else ``~/.config/memrank``."""
    return Path(os.environ.get(_DIR_ENV) or (Path.home() / ".config" / "memrank"))


def store_path() -> Path:
    """Absolute path to ``checkouts.json``."""
    return config_dir() / _FILENAME


def _load() -> dict[str, Any]:
    path = store_path()
    if not path.exists():
        return {"checkouts": {}}
    data = json.loads(path.read_text(encoding="utf-8"))
    data.setdefault("checkouts", {})
    return data


def _save(data: dict[str, Any]) -> None:
    write_json(store_path(), data, sort_keys=True, trailing_newline=True)


def get(ref: str) -> str | None:
    """The stored location for target ``ref``, unresolved, or ``None`` if it is not linked."""
    return _load()["checkouts"].get(ref)


def resolve(ref: str) -> str | None:
    """The location for target ``ref`` as a usable path, or ``None`` if it is not linked.

    An ``${ENV_VAR}`` value is resolved here rather than at the call site, so a stored reference and
    a stored literal are indistinguishable to everything downstream.
    """
    from memrank import config

    stored = get(ref)
    if stored is None:
        return None
    return config.expand_ref(stored, noun="checkout link")


def put(ref: str, path: str) -> str:
    """Link target ``ref`` to ``path`` and return what was stored.

    A literal path is stored expanded and absolute -- a relative entry means "this directory" when
    typed and "wherever the process is standing" when read back, the cwd-dependence
    ``settings._absolute_search_path`` documents. An ``${ENV_VAR}`` reference is stored verbatim,
    since it is not a path yet.
    """
    stored = path if path.startswith("${") else str(Path(path).expanduser().resolve())
    data = _load()
    data["checkouts"][ref] = stored
    _save(data)
    return stored


def delete(ref: str) -> bool:
    """Unlink target ``ref``; return whether it was linked."""
    data = _load()
    existed = data["checkouts"].pop(ref, None) is not None
    if existed:
        _save(data)
    return existed


def links() -> list[tuple[str, str]]:
    """Every link as ``(target_ref, stored_value)``, sorted by ref."""
    return sorted(_load()["checkouts"].items())
