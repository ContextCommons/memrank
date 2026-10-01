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
"""What a finished run leaves in its folder: ``result.json`` for programs, ``report.html`` for people.

Both are written from the same :class:`~memrank.service.protocol.RunResult`, so they cannot
disagree. Written whole and renamed into place, so an interrupted write leaves the previous
file or none, never half of one.
"""

from __future__ import annotations

from pathlib import Path

from memrank.atomic_json import staging_path, write_json
from memrank.loop.report_html import render
from memrank.service.protocol import RunResult

RESULT_FILE = "result.json"
REPORT_FILE = "report.html"


def write_outputs(folder: Path, result: RunResult) -> None:
    """Write ``result.json`` and ``report.html`` into ``folder``."""
    write_json(folder / RESULT_FILE, result.model_dump(mode="json"))
    report = folder / REPORT_FILE
    staged = staging_path(report)
    staged.write_text(render(result), encoding="utf-8")
    staged.replace(report)
