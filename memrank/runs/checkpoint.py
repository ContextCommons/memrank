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
"""What a run knew immediately before it started judging, written to disk.

A run holds every retrieved document in memory and writes its artifact exactly once, after judging.
So a crash in the judge stage -- the LAST stage, and the one with the least test exposure -- discards
the ingest and retrieval that paid for it. On 2026-08-12 that cost 4h50m: 272 ingests and 4,620
retrieves completed, and the process died three seconds before the first judge call with nothing on
disk but a progress heartbeat.

This module writes the cell as it stands before judging, so `memrank ops rejudge` can finish it.
The checkpoint is deliberately NOT a partial artifact: `cloud_submit` joins evaluate and upload with
`&&` so a failed run cannot publish something a reader mistakes for a result, and a distinctly named
file keeps that rule rather than weakening it.

DOCUMENTS ARE STORED ONCE IN A TABLE, AND REFERENCED BY INDEX. An engine that returns its whole
bank for every query serialises the same text once per query: a real 15.13 MB artifact held 8.63 MB
of retrieved text carrying 0.07 MB of distinct content.

Storing the text once is only half of it. A first version keyed documents by sha256 and referenced
them by hash, and cut that artifact's rows from 12.89 MB to 7.03 MB -- a disappointing 1.8x, because
the REFERENCES are the bulk, not the text: 152 queries x 204 results is ~31,000 of them, and a
64-character digest is no smaller than the ~193-character document it stood in for. An integer index
is what actually collapses it.

Documents are keyed on `(id, content)`, never on either alone. In that same artifact 14 memory ids
carried more than one content, because the engine consolidates memories while it is being queried;
keying on id alone would silently pick one version, and keying on content alone would render a
consolidation as one memory vanishing and another appearing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from memrank.atomic_json import write_json
from memrank.errors import MemrankError

#: The checkpoint's own format version. Bump when the shape changes in a way an older `rejudge`
#: would misread -- a resumed run must never be graded by code that disagrees about what it holds.
SCHEMA_VERSION = 1

#: Named once so the runner, the cloud uploader and `registry`'s exclusion agree.
CHECKPOINT_FILE = "retrieval.json"


class CheckpointError(MemrankError):
    """A checkpoint cannot be read, or does not describe the run it is being resumed into."""


#: The key a retrieved entry uses to name its document -- an integer index into ``documents``.
_REF = "d"


def pack(per_query: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split drill rows into a document table and rows that index into it.

    Returns ``(documents, rows)``. Each document is stored once as ``{"id", "content"}``; each
    ``retrieved`` entry keeps its own fields (``score`` and anything an adapter adds) and names its
    document by integer index.

    THE INDEX IS THE POINT, not just the deduplication. A first version stored content by sha256
    and referenced it by hash, which cut a real 12.89 MB artifact only to 7.03 MB -- because the
    entries, not the text, are the bulk: 152 queries x 204 results is ~31,000 references, and a
    64-character hex digest is no smaller than the ~193-character document it replaces. Indexing
    into a table makes each reference a couple of characters.

    Keyed on ``(id, content)``, never on either alone. Two documents can share an id with different
    content -- 14 did in that artifact, because the engine consolidates memories while it is being
    queried -- and two can share content under different ids. Collapsing either would rewrite what
    the run actually retrieved.
    """
    index: dict[tuple[str, str], int] = {}
    documents: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for row in per_query:
        referenced = []
        for doc in row.get("retrieved") or []:
            key = (str(doc.get("id")), doc.get("content") or "")
            position = index.get(key)
            if position is None:
                position = index[key] = len(documents)
                documents.append({"id": doc.get("id"), "content": doc.get("content") or ""})
            extras = {k: v for k, v in doc.items() if k not in ("id", "content")}
            referenced.append({**extras, _REF: position})
        rows.append({**row, "retrieved": referenced})
    return documents, rows


def unpack(documents: list[dict[str, Any]], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rebuild drill rows with their documents inlined -- the inverse of :func:`pack`.

    Raises:
        CheckpointError: If a row references a document the table does not hold. Refused rather
            than defaulted to empty: judging a query against silently-missing context produces a
            complete, plausible, wrong verdict, which is the failure this project exists to catch.
    """
    restored: list[dict[str, Any]] = []
    for row in rows:
        inlined = []
        for doc in row.get("retrieved") or []:
            position = doc.get(_REF)
            if not isinstance(position, int) or not 0 <= position < len(documents):
                raise CheckpointError(
                    f"query {row.get('query_id')!r} references document {position!r}, which the "
                    f"checkpoint does not contain. Refusing to judge against missing context.")
            stored = documents[position]
            extras = {k: v for k, v in doc.items() if k != _REF}
            inlined.append({"id": stored["id"], "content": stored["content"], **extras})
        restored.append({**row, "retrieved": inlined})
    return restored


def write(path: Path, *, cell: dict[str, Any], budget_mode: str) -> None:
    """Write the pre-judgment cell, with its documents deduplicated.

    ``cell`` is the aggregate as it stands before judging -- the same shape the artifact has, minus
    the verdicts. Carrying the whole cell rather than the rows alone is what lets a resumed run
    produce the artifact the run WOULD have produced, instead of a second-class metrics dict that
    nobody trusts.

    ``budget_mode`` is the EFFECTIVE reader-context mode -- the target's ``context_budget``
    promoted by the benchmark's ``context_policy`` (runner._effective_budget_mode) -- and is the
    one thing the judge needs that no receipt field records. Without it a resumed uncapped arm
    would be re-judged as matched and silently report the fairness cap instead of what it
    actually carried.
    """
    documents, rows = pack(cell.get("per_query") or [])
    # `indent=None`, unlike every other JSON this project writes. Those are read by people and
    # diffed; this one is read by `rejudge` and thrown away. Full LoCoMo has ~31,000 retrieved
    # entries, and pretty-printing them spends more bytes on whitespace than on the run.
    write_json(path, {
        "schema_version": SCHEMA_VERSION,
        "budget_mode": budget_mode,
        "documents": documents,
        "cell": {**cell, "per_query": rows},
    }, indent=None)


def read(path: Path) -> tuple[dict[str, Any], str]:
    """Load a checkpoint; return ``(cell with content inlined, budget_mode)``."""
    import json

    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CheckpointError(f"could not read checkpoint {path}: {exc}") from exc
    version = data.get("schema_version")
    if version != SCHEMA_VERSION:
        raise CheckpointError(
            f"checkpoint {path} is schema {version!r}, this build reads {SCHEMA_VERSION!r}. "
            f"Resuming across a format change would grade a run under assumptions it was not "
            f"written with.")
    cell = dict(data["cell"])
    # `or []`: the document table is a LIST (`pack` builds one, and `unpack` indexes into it).
    # `or {}` was harmless only because an empty table means no row references anything.
    cell["per_query"] = unpack(data.get("documents") or [], cell.get("per_query") or [])
    return cell, data["budget_mode"]
