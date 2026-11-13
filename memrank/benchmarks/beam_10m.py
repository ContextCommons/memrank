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
"""BEAM's 10M bucket: a separate HuggingFace dataset with its own chat shape.

`Mohammadta/BEAM-10M` is not a fourth split of `Mohammadta/BEAM`. It is one split (``10M``)
of ten conversations, and its ``chat`` is nested one level deeper than the other tiers':

- other tiers: ``chat`` = [batch, ...], each batch a flat list of turns.
- 10M: ``chat`` = [{"plan-1": [batch, ...], "plan-2": None, ...}, ...] -- one element per plan,
  each a struct carrying every ``plan-N`` key with exactly one non-null; each batch is
  ``{"batch_number", "time_anchor", "turns"}`` and ``turns`` is a list of EXCHANGES (a user
  message, the reply, any follow-ups), each a flat list of turns.

:func:`sessions_of` maps it onto the other tiers' shape: one session per batch, in plan order
then batch order, holding the batch's exchanges concatenated. That is the batch BEAM's own
reference harness iterates (``create_chunking`` walks plan -> batch -> turns for 10M), and
turn ids run globally 0..N across it, so nothing is reordered. The row's ``plans`` column
repeats the same chat per plan with ids restarting at 0; it is neither read nor cached.

A 10M conversation is about 49M characters, and the whole split ~1 GB, so it is cached one
conversation per file with a manifest written LAST: a smoke or mini slice reads one or five
files rather than the whole tier, and an interrupted download is never mistaken for a
complete one.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from memrank.errors import MemrankError

HF_DATASET = "Mohammadta/BEAM-10M"
HF_SPLIT = "10M"

#: What a 10M row contributes. `plans` (a per-plan copy of the chat), `narratives`,
#: `conversation_plan`, `conversation_seed` and `user_profile` are generation material BEAM's
#: own baselines never see -- the same reason `user_profile` is not ingested at other tiers.
_COLUMNS = ("conversation_id", "chat", "probing_questions")

_MANIFEST = "manifest.json"


class Beam10MShapeError(MemrankError):
    """A BEAM-10M row whose chat is not the shape this loader maps."""


def _plan_batches(index: int, element: Any) -> list[Any]:
    """The batches of plan ``index`` (0-based), refusing any other layout."""
    if not isinstance(element, dict):
        raise Beam10MShapeError(f"BEAM-10M chat[{index}] is {type(element).__name__}, "
                                "expected a struct of plan-N keys")
    present = [key for key, value in element.items() if value is not None]
    expected = f"plan-{index + 1}"
    if present != [expected]:
        raise Beam10MShapeError(f"BEAM-10M chat[{index}] carries {present}, expected exactly "
                                f"[{expected!r}]; plans out of order or merged")
    batches = element[expected]
    if not isinstance(batches, list):
        raise Beam10MShapeError(f"BEAM-10M {expected} is not a list of batches")
    return batches


def _batch_turns(plan: int, position: int, batch: Any) -> list[dict[str, Any]]:
    """One batch's exchanges, concatenated into the flat turn list a session holds."""
    if not isinstance(batch, dict) or batch.get("batch_number") != position + 1:
        number = batch.get("batch_number") if isinstance(batch, dict) else None
        raise Beam10MShapeError(f"BEAM-10M plan-{plan + 1} batch at position {position} has "
                                f"batch_number {number!r}, expected {position + 1}")
    exchanges = batch.get("turns")
    if not isinstance(exchanges, list) or not all(isinstance(e, list) for e in exchanges):
        raise Beam10MShapeError(f"BEAM-10M plan-{plan + 1} batch {position + 1}: `turns` is "
                                "not a list of exchanges")
    return [turn for exchange in exchanges for turn in exchange]


def sessions_of(chat: Any) -> list[list[dict[str, Any]]]:
    """A 10M ``chat`` as the other tiers' shape: one flat turn list per batch, in order."""
    if not isinstance(chat, list) or not chat:
        raise Beam10MShapeError("BEAM-10M row has no `chat` list of plans")
    return [_batch_turns(plan, position, batch)
            for plan, element in enumerate(chat)
            for position, batch in enumerate(_plan_batches(plan, element))]


def _download(cache_dir: Path) -> None:
    """Fetch the split and write it one conversation per file, manifest last."""
    from memrank.errors import optional_import

    load_dataset = optional_import("datasets", "benchmarks").load_dataset
    # Arrow-backed and memory-mapped: iterating holds one row at a time, never the split.
    ds = load_dataset(HF_DATASET, split=HF_SPLIT).select_columns(list(_COLUMNS))
    cache_dir.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    for position, row in enumerate(ds):
        name = f"{position:02d}.json"
        with open(cache_dir / name, "w", encoding="utf-8") as f:
            json.dump(dict(row), f)
        names.append(name)
    with open(cache_dir / _MANIFEST, "w", encoding="utf-8") as f:
        json.dump(names, f)


def load_rows(cache_dir: Path, limit: int | None) -> list[dict[str, Any]]:
    """The first ``limit`` conversations (all when None), downloading once if needed."""
    manifest = cache_dir / _MANIFEST
    if not manifest.exists():
        from memrank.benchmarks import dataset_download_notice

        with dataset_download_notice("BEAM-10m", HF_DATASET, cache_dir):
            _download(cache_dir)
    with open(manifest, encoding="utf-8") as f:
        names = json.load(f)
    rows = []
    for name in names[:limit]:
        with open(cache_dir / name, encoding="utf-8") as f:
            rows.append(json.load(f))
    return rows
