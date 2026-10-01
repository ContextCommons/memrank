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
"""Deprecated: the shipped evaluations as classes for the in-process Python route
(`Demo().run(system=...)`). Evaluate an agent with `memrank run` instead; see the README.

The evaluations memrank ships, as classes -- `from memrank.evaluations import Demo`.

`memrank.evaluation("demo")` needs the string first, and a string is not navigable: nothing in
an editor follows it, nothing states what a tier or a slice may be, and nothing tells a reader
what else is out there. Each class here is a subclass of `memrank.Evaluation` whose constructor
takes the keywords its benchmark takes, typed, and loads it -- so `Demo()` beside
`TFIDF()` reads as two nouns of one shape, and "go to definition" on either lands on a
class whose docstring says what it is.

The construction goes through `memrank.evaluation(...)` underneath, so there is one conversion
and not two, and an instance carries exactly the fields the string form produces.

================  ==================================================================
name              what it needs
================  ==================================================================
Demo()            nothing -- a bundled synthetic scenario
RelationGraph()   nothing -- in-repo fixtures
LoCoMo()          a one-time download, or LOCOMO_DATA_PATH; judging for any quality
LongMemEval()     a one-time download, or LONGMEMEVAL_DATA_PATH; judging likewise
BEAM()            the benchmarks extra's download, or BEAM_DATA_PATH; judging too
================  ==================================================================

Each benchmark class is imported INSIDE its constructor. `import memrank` reaches this module,
and a dataset loader's module graph is not something a caller who asked for `Demo()` should pay
for. `memrank.catalog()` prints this table at runtime.
"""

from __future__ import annotations

from memrank.instrument.catalog import ShippedEvaluation, evaluation
from memrank.instrument.evaluation import Evaluation


class _Shipped(Evaluation):
    """What the shipped classes share: taking on the fields of an evaluation already built.

    Each subclass differs only in which benchmark it loads. Spelling the six field names out in
    each constructor would make a new field require repeated edits.
    """

    def _adopt(self, built: Evaluation) -> None:
        """Become the evaluation `built` is. Called by a subclass constructor, once."""
        Evaluation.__init__(self, name=built.name, version=built.version, tasks=built.tasks,
                            measures=built.measures, clearing=built.clearing,
                            metadata=built.metadata)


class Demo(_Shipped):
    """One hand-crafted multi-session scenario: 5 questions about what it was told earlier.

    Needs nothing -- the scenario is bundled, synthetic and offline. This dependency-free smoke
    evaluation measures a substring retrieval proxy (`word-match`) plus the
    benchmark's own evidence recall, latency and failure rate; no judge, no key.

    `data_path` (or DEMO_DATA_PATH) points the loader at a scenario of your own, which is then
    no longer synthetic and no longer egress-safe.
    """

    def __init__(self, slice: str | None = None, k: int = 10,
                 data_path: str | None = None) -> None:
        from memrank.benchmarks.demo import DemoBenchmark

        built = evaluation(DemoBenchmark(slice=slice, k=k, data_path=data_path))
        self._adopt(built)


class RelationGraph(_Shipped):
    """Four synthetic fixtures asking whether a memory built the right relations between facts.

    Needs nothing -- the fixtures are in the repository -- but the system must be graph-capable:
    it has to supply the normalized graph snapshot the fixtures are scored against. Measures the
    benchmark's own structural graph score, which is self-contained and needs no judge, plus
    latency and failure rate.
    """

    def __init__(self, slice: str | None = None, k: int = 10) -> None:
        from memrank.benchmarks.relation_graph import RelationGraphBenchmark

        built = evaluation(RelationGraphBenchmark(slice=slice, k=k))
        self._adopt(built)


class LoCoMo(_Shipped):
    """Ten multi-session conversations, ~199 QA pairs each, about what was said across sessions.

    Needs a one-time download (or LOCOMO_DATA_PATH pointing at the dataset), and a judge for any
    quality number at all: LoCoMo's published protocol grades the generated answer, and its
    derived answers ("3 times", "2022") can never appear verbatim in retrieved context, so no
    substring proxy is computed. Unjudged, this measures latency and failure rate and nothing
    else. `slice` is "smoke", "mini" or None for the full set.
    """

    def __init__(self, slice: str | None = None, k: int = 10) -> None:
        from memrank.benchmarks.locomo import LoCoMoBenchmark

        built = evaluation(LoCoMoBenchmark(slice=slice, k=k))
        self._adopt(built)


class LongMemEval(_Shipped):
    """~500 questions across six question types, each with its own haystack of sessions.

    Needs a one-time download (or LONGMEMEVAL_DATA_PATH), and a judge for any quality number:
    the published metric is judged QA accuracy and the paper rejects exact matching, so no
    substring proxy is computed. Unjudged, this measures latency and failure rate. `slice` is
    "smoke", "mini" or None for the full set.
    """

    def __init__(self, slice: str | None = None, k: int = 10) -> None:
        from memrank.benchmarks.longmemeval import LongMemEvalBenchmark

        built = evaluation(LongMemEvalBenchmark(slice=slice, k=k))
        self._adopt(built)


class BEAM(_Shipped):
    """20 conversations / 400 questions at 100k, 35 / 700 at 500k and 1m; 2 questions per ability.

    Needs a one-time download from HuggingFace, through the `benchmarks` extra
    (`pip install 'memrank[benchmarks]'`), or BEAM_DATA_PATH; and a judge for any quality
    number: BEAM's gold answers are prose rubrics rather than verbatim spans, so no substring
    proxy is computed and the raw composite is withheld from ranking until judged. Unjudged,
    this measures latency and failure rate. `tier` is "100k", "500k" or "1m", and an unknown one
    is refused here, before anything is fetched.
    """

    def __init__(self, tier: str = "100k", slice: str | None = None, k: int = 10) -> None:
        from memrank.benchmarks.beam import BEAMBenchmark

        built = evaluation(BEAMBenchmark(tier=tier, slice=slice, k=k))
        self._adopt(built)


#: The same table the module docstring carries, in the form `memrank.catalog()` prints. One
#: entry per name in `memrank.benchmarks.REGISTRY`, which `tests/instrument` holds it to.
SHIPPED: tuple[ShippedEvaluation, ...] = (
    ShippedEvaluation("Demo", "demo", "a substring retrieval proxy and evidence recall",
                      "nothing -- a bundled synthetic scenario"),
    ShippedEvaluation("RelationGraph", "relation_graph", "a structural graph score",
                      "nothing -- in-repo fixtures, and a graph-capable system"),
    ShippedEvaluation("LoCoMo", "locomo", "latency and failures; quality needs a judge",
                      "a one-time download, or LOCOMO_DATA_PATH"),
    ShippedEvaluation("LongMemEval", "longmemeval", "latency and failures; quality needs a judge",
                      "a one-time download, or LONGMEMEVAL_DATA_PATH"),
    ShippedEvaluation("BEAM", "beam", "latency and failures; quality needs a judge",
                      "a one-time download (the benchmarks extra), or BEAM_DATA_PATH"),
)

__all__ = ["BEAM", "SHIPPED", "Demo", "LoCoMo", "LongMemEval", "RelationGraph"]
