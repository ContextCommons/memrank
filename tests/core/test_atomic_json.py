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
"""The atomic JSON writer, and the rule that nobody hand-rolls a second one.

Every prior copy staged through a fixed temp name derived from the target, which is only safe
with one writer. See :mod:`memrank.atomic_json` for the run that died of it.
"""
from __future__ import annotations

import json
import re
import stat
import threading
from pathlib import Path

from memrank import atomic_json

#: The hand-rolled pattern this module replaced: stage to a name derived from the target, then
#: rename. `leaderboard.bundle` is allowed -- its `os.replace` promotes an already-staged
#: generation directory, and its tmp name already carries pid + a random token.
#: Package-relative, not bare filenames: `bundle.py` alone would silently exempt any future
#: module that happens to share the name.
_STAGING_PATTERN = re.compile(r"""\.json\.tmp|with_suffix\(["']\.tmp""")
_ALLOWED = {Path("atomic_json.py"), Path("leaderboard/bundle.py")}


def test_writes_json_and_creates_parents(tmp_path):
    path = tmp_path / "nested" / "store.json"
    atomic_json.write_json(path, {"b": 1, "a": 2}, sort_keys=True)
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": 2, "b": 1}
    assert list(path.parent.glob("*.tmp")) == []


def test_trailing_newline_and_mode(tmp_path):
    path = tmp_path / "secrets.json"
    atomic_json.write_json(path, {"k": "v"}, mode=0o600, trailing_newline=True)
    assert path.read_text(encoding="utf-8").endswith("}\n")
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_staging_name_is_unique_per_thread(tmp_path):
    """The collision that killed a run: two threads staging the same target file."""
    path = tmp_path / "status.json"
    names = []
    barrier = threading.Barrier(4)

    def stage():
        barrier.wait()
        names.append(atomic_json.staging_path(path).name)

    threads = [threading.Thread(target=stage) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(set(names)) == 4, names


def test_staging_name_is_hidden_and_not_a_result_cell(tmp_path):
    """A crashed writer's leftover must not read as a run artifact (see run_registry.cell_files)."""
    name = atomic_json.staging_path(tmp_path / "target__bench.json").name
    assert name.startswith(".") and name.endswith(".tmp")


def test_no_other_module_hand_rolls_atomic_json_staging():
    """Fails closed when a new call site copies the pattern instead of using the chokepoint."""
    package = Path(atomic_json.__file__).parent
    offenders = [rel for p in package.rglob("*.py")
                 if (rel := p.relative_to(package)) not in _ALLOWED
                 and _STAGING_PATTERN.search(p.read_text("utf-8"))]
    assert offenders == [], f"hand-rolled atomic write; use atomic_json.write_json: {offenders}"
