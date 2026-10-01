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
"""The one way memrank writes a JSON file atomically -- stage to a temp file, rename over.

The rename is what makes the write atomic: a reader (``memrank ps``, ``watch``, the leaderboard
site) either sees the whole old file or the whole new one, never a half-written one.

**Why this module exists rather than four hand-rolled copies.** Every copy staged through a
*fixed* temp name derived from the target (``status.json`` -> ``status.json.tmp``). That is only
safe with a single writer, and the run heartbeat has ten: :func:`memrank.runner._run_units_concurrent`
evaluates units in a thread pool and each worker publishes progress per document and per query.
Two threads wrote the same temp file, the first ``replace`` consumed it, and the second raised
``FileNotFoundError`` -- killing a LongMemEval run 975/1232 documents in, from a *progress update*.
The staging name here is unique per process AND per thread, so no writer can consume another's
in-flight file, and a lock serialises stage-then-rename so the last write wins as a whole record
rather than as an interleaving of two.

The temp name is dot-prefixed so a crashed writer's leftover reads as a hidden scratch file rather
than a run artifact -- see the comment in :mod:`memrank.runs.registry` about ``status.json`` being
swallowed as a result cell the moment it appeared in a run directory.

Failures propagate. A JSON file that cannot be written means a broken disk (full, read-only,
unmounted), and callers must not continue against a store they failed to update.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

#: Serialises stage-then-rename across threads. One lock for all paths rather than one per path:
#: these writes are small and rare enough that the contention costs nothing measurable, and a
#: per-path registry is another shared dict to keep thread-safe for no gain.
_WRITE_LOCK = threading.Lock()


def staging_path(path: Path) -> Path:
    """The temp file ``path`` is staged through -- unique per process and per thread.

    Both halves matter. The pid separates a local run from a concurrent CLI writing the same
    store; the thread id separates the eval workers, which is the collision that actually
    happened.
    """
    return path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")


def write_json(path: Path, data: Any, *, mode: int | None = None, indent: int | None = 2,
               sort_keys: bool = False, trailing_newline: bool = False) -> None:
    """Serialise ``data`` to ``path`` atomically, creating parent directories as needed.

    Args:
        path: Destination file. Its parent is created if absent.
        data: Any JSON-serialisable value.
        mode: Permission bits applied to the staged file *before* the rename and to the
            destination after, so the content is never briefly world-readable. ``None`` leaves
            the umask default.
        indent: ``json.dumps`` indent. ``None`` writes the compact form, for a file
            no person reads -- see `runs.checkpoint.write`.
        sort_keys: Sort object keys -- for stores a human diffs (the wallet, settings).
        trailing_newline: Append a newline, for files meant to be ``cat``-ed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, indent=indent, sort_keys=sort_keys)
    if trailing_newline:
        text += "\n"
    tmp = staging_path(path)
    with _WRITE_LOCK:
        tmp.write_text(text, encoding="utf-8")
        if mode is not None:
            os.chmod(tmp, mode)
        tmp.replace(path)
    if mode is not None:
        os.chmod(path, mode)
