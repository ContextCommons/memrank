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
"""Where runs live: one folder per run, and the small interface the service is written against.

A run's folder (``results/<run-id>/`` by default) is its working record: ``run.json`` holds
the request and the step state, and the runner adds ``result.json`` and ``report.html``. Resuming reads the same folder, so
there is no second place progress could disagree with. Files are written whole and renamed into
place (:func:`memrank.atomic_json.write_json`), so a crash never leaves half a state.

The interface stays because a hosted service will put a database behind the same calls; this
store serves one process, and a lock makes that process's requests for one run take turns.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Protocol

from pydantic import ValidationError

from memrank.atomic_json import write_json
from memrank.errors import MemrankError
from memrank.service.machine import RunState
from memrank.service.protocol import RunCreate

RUN_FILE = "run.json"


class RunNotFound(MemrankError, KeyError):
    """No run with this id."""


class RunUnreadable(MemrankError):
    """A run folder whose saved state this version cannot read."""


class RunStore(Protocol):
    """The persistence the service needs, and nothing else."""

    def create(self, run_id: str, request: RunCreate, state: RunState) -> None: ...

    def load(self, run_id: str) -> tuple[RunCreate, RunState]: ...

    def save_state(self, run_id: str, state: RunState) -> None: ...

    def transaction(self, run_id: str) -> Any: ...


class FolderRunStore:
    """Runs as folders under ``root``, one process at a time."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._lock = threading.RLock()
        #: Each run as last read or written by this store, so a step neither parses nor
        #: re-reads a file that grows with the run. The file stays the record: it is written
        #: whole on every save, and what a new process reads.
        self._runs: dict[str, tuple[RunCreate, RunState]] = {}

    def run_dir(self, run_id: str) -> Path:
        if not run_id or "/" in run_id or run_id.startswith("."):
            raise RunNotFound(f"{run_id!r} is not a run id")
        return self.root / run_id

    @contextmanager
    def transaction(self, run_id: str) -> Iterator[None]:
        """Hold the run while reading and writing it."""
        with self._lock:
            yield

    def create(self, run_id: str, request: RunCreate, state: RunState) -> None:
        folder = self.run_dir(run_id)
        folder.mkdir(parents=True, exist_ok=False)
        self._write_run(run_id, request, state)

    def load(self, run_id: str) -> tuple[RunCreate, RunState]:
        """The run's request and state. The state is shared: change a copy, then save it."""
        with self._lock:
            if run_id not in self._runs:
                self._runs[run_id] = self._read(run_id)
            return self._runs[run_id]

    def _read(self, run_id: str) -> tuple[RunCreate, RunState]:
        path = self.run_dir(run_id) / RUN_FILE
        if not path.is_file():
            raise RunNotFound(f"no run {run_id!r}: {path} does not exist")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            state = RunState.model_validate(data["state"])
        except (json.JSONDecodeError, KeyError, ValidationError) as exc:
            raise RunUnreadable(
                f"run {run_id} was saved in a shape this memrank cannot read, so it cannot be "
                f"resumed; start a new run. {path}: {exc}") from exc
        return RunCreate.model_validate(data["request"]), state

    def save_state(self, run_id: str, state: RunState) -> None:
        with self._lock:
            request, _ = self.load(run_id)
            self._write_run(run_id, request, state)
            self._runs[run_id] = (request, state)

    def _write_run(self, run_id: str, request: RunCreate, state: RunState) -> None:
        # Compact: it is rewritten on every step and read by memrank alone, and an indented
        # dump goes through json's pure-Python encoder, many times slower on a large run.
        write_json(self.run_dir(run_id) / RUN_FILE,
                   {"request": request.model_dump(mode="json"),
                    "state": state.model_dump(mode="json")}, indent=None)
