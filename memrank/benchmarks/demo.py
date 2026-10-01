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
"""Hand-crafted demo benchmark.

A small, synthetic, multi-session conversation with durable facts, distractors,
and a negative/abstention query -- authored so engines genuinely differ and a
lazy "non-empty retrieval" cannot win. Scored by the evidence-based scorer.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from memrank.core import AdapterResponse, Benchmark, BenchmarkUnit, Document, EvalInfo
from memrank.judging.prompts import MEMRANK_READER
from memrank.metrics.scoring import SpanRecall

#: The bundled scenario, resolved beside this module inside the package.
#:
#: It used to be found by walking two directories up to `examples/data/`, a path that exists
#: only in a checkout: from a `pip install memrank` the demo failed on a missing file, and the
#: demo is the one benchmark that needs no engine, no credential and no download. The file now
#: ships as package data (`[tool.setuptools.package-data]`), so the package carries its own
#: scenario wherever it is installed.
#:
#: `Path(__file__).parent` rather than `importlib.resources.files()` for the same reason
#: `memrank/targets/catalog.py` resolves `builtin/` this way: `files()` returns a `Traversable`,
#: whose only importable name across our 3.10 floor and 3.12 is `importlib.abc.Traversable`,
#: which 3.12 deprecates and 3.14 removes. Both forms ask the package where it is; this one
#: stays a `Path`, which is also what a `DEMO_DATA_PATH` override is. ATO-2059.
_BUNDLED_SCENARIO = Path(__file__).resolve().parent / "data" / "demo_scenario.json"


class DemoBenchmark(Benchmark):
    """Single-unit synthetic benchmark loaded from a JSON scenario file."""

    name = "demo"
    dataset_version = "memrank-demo@v1"
    #: The scorer, under its public name. It used to be spelled out inline here, which is why
    #: it had no name for anyone else to reach (ATO-2135); the arithmetic is unchanged.
    scorer = SpanRecall()
    # No official reader prompt exists for this synthetic evaluation: memrank's own, declared.
    answer_prompt = MEMRANK_READER
    info = EvalInfo(unit="scenario",
                    units_declared="1 hand-crafted multi-session scenario (bundled JSON)",
                    slices=())

    def __init__(self, slice: str | None = None, k: int = 10,
                 data_path: str | None = None) -> None:
        self.k = k
        env = os.environ.get("DEMO_DATA_PATH") or data_path
        self._path = Path(env) if env else _BUNDLED_SCENARIO
        # Only the bundled scenario is synthetic; a DEMO_DATA_PATH/data_path
        # override may point at real data, so it is NOT egress-safe by default.
        self.is_synthetic = env is None
        # Question text is public only when using the bundled synthetic scenario.
        self.question_text_public = self.is_synthetic

    def load(self) -> list[BenchmarkUnit]:
        raw = json.loads(self._path.read_text(encoding="utf-8"))
        documents = [
            Document(id=d["id"], content=d["content"], user_id=raw["isolation_id"],
                     metadata={"doc_id": d.get("doc_id", d["id"])})
            for d in raw["documents"]
        ]
        return [BenchmarkUnit(
            unit_id=raw["unit_id"], isolation_id=raw["isolation_id"],
            documents=documents, queries=raw["queries"])]

    def score(self, unit: BenchmarkUnit,
              responses: list[AdapterResponse]) -> dict[str, Any]:
        return self.scorer.score(unit, responses)

    def report_template(self) -> str:
        return (
            "# Demo report -- {adapter}\n\n"
            "> Quality = retrieval recall proxy, NOT answer correctness.\n\n"
            "- Composite: **{composite}**\n"
            "- Dataset: {dataset_version}\n- k: {k}\n\n"
            "## Per-category recall\n{per_category_md}\n"
        )
