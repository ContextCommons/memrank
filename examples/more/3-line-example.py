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
"""The canonical Memrank "hello world" -- three meaningful lines.

Run from the repo root:

    python examples/3-line-example.py

Requires a reachable AtomicMemory backend (defaults to http://localhost:3070).
"""

from memrank.adapters import get_adapter
from memrank.benchmarks import get_benchmark
from memrank.core import AdapterResponse


def main() -> None:
    # Line 1: pick an adapter.
    adapter = get_adapter("atomicmemory")
    # Line 2: pick a benchmark + slice.
    benchmark = get_benchmark("locomo", slice="smoke", k=10)
    # Line 3: run units, score, print composite.
    units = benchmark.load()
    composites = []
    for unit in units:
        adapter.prepare(unit.isolation_id)
        adapter.ingest(unit.documents)
        responses: list[AdapterResponse] = []
        for query in unit.queries:
            recall = adapter.retrieve(query["text"], 10, query["user_id"])
            responses.append(AdapterResponse(query_id=query["id"],
                                             documents=recall.documents,
                                             raw=recall.declared))
        adapter.cleanup()
        composites.append(benchmark.score(unit, responses)["composite"])
    print(f"composite={sum(composites) / len(composites):.3f}")
    print(f"latency={adapter.latency_metrics()}")
    print(f"tokens={adapter.token_metrics()}")


if __name__ == "__main__":
    main()
