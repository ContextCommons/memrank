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
"""One run's RECORD: what it measured and under what conditions, minus the bulk.

The record is the small, durable half of a run -- every cell with its receipt (config hash,
environment stamp, provenance pins) plus the run summary. Kilobytes, against cells that reach
300 MB. It is what `memrank runs sync` pushes to the org, and it is the only thing the browser
can read, because a page render cannot stream a 300 MB artifact out of S3.

**WHY THIS IS ITS OWN MODULE.** It used to live inside :mod:`memrank.cli.sync`, reachable only
by syncing -- which is a thing only a LOCAL run does. `PUT /orgs/{org}/runs/{id}` refuses a
cloud-born run outright (it must not overwrite what the platform launched), so a cloud run's
record stayed empty forever and its numbers existed only as S3 objects. Every cloud run -- which
is every run the browser submits -- therefore rendered as a row with a score and nothing behind
it. The cloud task now builds the same record with this function and uploads it as
``record.json`` beside the cells, and the API ingests that on the first terminal read.

Same function on both paths deliberately: a record assembled two ways is two records, and the
difference would surface as a browser and a terminal disagreeing about one run.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from memrank.runs import registry

#: Dropped from every cell before upload. Measured at 47 MB in one locomo cell against 1.2 KB
#: of receipt beside it -- this is the difference between syncing a record and syncing a dataset.
ARTIFACT_KEYS = ("per_query", "ingested_documents")

#: What :func:`build` is written to when a run uploads it, and what the API reads back. A name
#: without ``__`` in it on purpose: `registry.cell_files` matches cells POSITIVELY on
#: ``<target>__<benchmark>.json``, so this lands in a run directory alongside them without ever
#: being mistaken for one.
RECORD_FILENAME = "record.json"


def build(run_dir: Path) -> dict[str, Any]:
    """One run's record: every cell minus its artifacts, plus the summary.

    The receipt rides inside each cell untouched -- it is the part that makes a number mean
    something later (what config produced it, on what machine, against which engine build).

    Args:
        run_dir: A directory holding a finished run's cell artifacts and its summary.

    Returns:
        ``{"cells": [...], "summary": {...} | None}``.
    """
    cells = []
    for path in registry.cell_files(run_dir):
        cell = json.loads(path.read_text(encoding="utf-8"))
        for key in ARTIFACT_KEYS:
            cell.pop(key, None)
        cells.append(cell)
    summaries = sorted(run_dir.glob("summary__*.json"))
    summary = json.loads(summaries[0].read_text(encoding="utf-8")) if summaries else None
    return {"cells": cells, "summary": summary}
