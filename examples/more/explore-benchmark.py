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
"""Load ONE benchmark and look at it -- no runs, no targets, nothing written.

Run from the repo root:

    python examples/explore-benchmark.py

Uses `demo` so it runs offline and instantly. For the real thing, swap the ref for
e.g. "beam:100k-smoke" or "locomo:smoke" -- the first `load()`/`raw()` then downloads
the dataset into ~/.memrank/datasets (or $MEMRANK_CACHE_DIR) and is cached afterwards.
"""

import memrank


def main() -> None:
    # One string, same grammar as the CLI's EVAL argument and `memrank.evaluation(ref)`.
    bench = memrank.benchmark("demo")
    # Equivalent class form, when you'd rather name the knobs than the ref:
    #   from memrank.benchmarks import BEAMBenchmark
    #   bench = BEAMBenchmark(tier="100k", slice="smoke")

    print(repr(bench))          # class, registry name, declared knobs -- no I/O
    print(bench.info)           # what a unit is, declared sizes, slices, tiers -- no I/O
    print()

    units = bench.load()        # the harness's shape: BenchmarkUnits
    print(f"{len(units)} unit(s); the first:")
    unit = units[0]
    print(" ", unit)
    for doc in unit.documents:
        print("   ", doc)
    for query in unit.queries[:3]:
        print("   query:", query["id"], "--", query["text"])
    print()

    records = bench.raw()       # what the benchmark's authors published, pre-normalisation
    print(f"raw upstream records: {len(records)}; keys of the first:")
    print(" ", sorted(records[0]))


if __name__ == "__main__":
    main()
