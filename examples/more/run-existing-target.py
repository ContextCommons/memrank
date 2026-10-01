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
"""Evaluate an EXISTING catalog target from a script -- no CLI, nothing saved.

Run from the repo root:

    python examples/run-existing-target.py

The target ref resolves exactly as `memrank submit` resolves it. `word-overlap` is a
built-in control arm, so this runs offline; a real engine ref ("mem0",
"hindsight:matched", or your own manifest on targets.path) works the same way BUT its
engine must already be reachable -- `memrank.run` provisions nothing. Provisioning,
run records, heartbeats and sync are what the CLI's `submit` adds on top.
"""

# The cell run, imported from its own module: `memrank.run` is now the typed run over
# the seven, and this script is about the artifact the cell run returns.
import memrank
from memrank.evaluation.api import run as run_cell


class Progress(memrank.EvalObserver):
    """A library run is silent by default; subclass what you want to hear.

    (This is the same hook the CLI itself is built on -- every method may be called
    from worker threads, so keep implementations thread-safe.)
    """

    def planned(self, plan):
        print(f"[plan] {plan.units} unit(s), {plan.documents} docs, "
              f"{plan.retrievals} retrievals")

    def item_done(self, stage, *, seconds, done=0, total=0, **kwargs):
        if done == total:
            print(f"[{stage}] {done}/{total} done")


def main() -> None:
    result = run_cell("word-overlap", "demo", repeats=1, observer=Progress())

    # The typed result is the whole answer: nothing was written anywhere.
    print(f"\ncomposite: {result.composite:.3f}  ({result.engine_ref} × {result.evaluation})")
    print("judged:", result.judged_metrics is not None,
          "(demo scores itself; judge-protocol evals like locomo default to judged)")
    misses = [row["query_id"] for row in result.per_query if not row["hit"]]
    print("missed queries:", misses or "none")
    # result.to_dict() is the exact artifact schema the CLI persists -- store it
    # wherever you like, or nowhere.
    print("artifact keys:", len(result.to_dict()))


if __name__ == "__main__":
    main()
