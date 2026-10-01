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
"""Core abstractions for Memrank.

Shared by the current path. Deprecated here: the ``MemoryAdapter``/``MemoryEngine`` contract
and the in-process route it serves; evaluate an agent with `memrank run` instead (see the
README). ``Benchmark`` and the value types are not deprecated.

Defines ``Benchmark``, and re-exports the memory contract that ``Memory`` defines in
:mod:`memrank.instrument.kinds` -- under that name and under its deprecated spellings,
``MemoryAdapter`` and ``MemoryEngine``. The value types the two contracts exchange --
``Document``, ``Recall``, ``AdapterResponse``, ``BenchmarkUnit``, ``EvalInfo`` and the
required metric keys -- live in :mod:`memrank.contract` and are re-exported here too, which
is where every caller has always imported them from.

The ABCs are intentionally minimal. A memory wraps a memory engine; benchmarks
wrap a dataset + scorer. Everything else (latency capture, token capture,
receipt generation) is composed around them by the runner.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

# Re-exported, not merely used: `memrank.core` is the import path every adapter, benchmark and
# caller in the tree already names for these, and moving the definitions must not move that.
# `Document`, `Recall` and the two key tuples are named by the memory contract, which now lives
# in `memrank.instrument.kinds`; the `X as X` spelling is what says they are re-exported here
# deliberately rather than left behind by the move.
from memrank.contract import (
    REQUIRED_LATENCY_KEYS as REQUIRED_LATENCY_KEYS,  # noqa: PLC0414 - re-export only
)
from memrank.contract import (
    REQUIRED_TOKEN_KEYS as REQUIRED_TOKEN_KEYS,  # noqa: PLC0414 - re-export only
)
from memrank.contract import (
    AdapterResponse,
    BenchmarkUnit,
    EvalInfo,
)
from memrank.contract import (
    Document as Document,  # noqa: PLC0414 - re-export only
)
from memrank.contract import (
    Recall as Recall,  # noqa: PLC0414 - re-export only
)
from memrank.instrument.kinds import Memory
from memrank.judging.prompts import MEMRANK_READER, AnswerPrompt
from memrank.quality import DATASET_VERSION_UNSET, QUALITY_METRICS

if TYPE_CHECKING:  # `judge_shape` reaches `judge`; keep that out of `core`'s import graph.
    from memrank.judging.shape import JudgeShape


#: The memory contract, defined in :mod:`memrank.instrument.kinds` under the kind's own name
#: and re-exported here, which is the import path every adapter in the tree already names.
#: `MemoryAdapter` and `MemoryEngine` are deprecated aliases of that same class object, so
#: ``isinstance(x, MemoryAdapter)`` and every registered adapter keep holding; they go at
#: plan step 22 (ATO-2151).
MemoryAdapter = Memory
MemoryEngine = Memory


# Human-facing labels/descriptions for a benchmark's composite quality metric,
# keyed by Benchmark.quality_metric. Unknown kinds fall back to the raw name.
#
# DERIVED from `memrank.quality.QUALITY_METRICS`, which holds the same prose split into the
# parts a result can carry by name -- what the number is OF, and what it is NOT. These two
# dicts are byte-identical to the literals they replaced and every reader of them is
# unchanged; the split exists so the caveat is reachable from a result rather than only from
# the terminal. `tests/core/test_quality_declarations.py` pins the byte-identity.
QUALITY_METRIC_LABELS = {kind: d.label for kind, d in QUALITY_METRICS.items()}
QUALITY_METRIC_DESCRIPTIONS = {kind: d.description for kind, d in QUALITY_METRICS.items()}



class Benchmark(ABC):
    """The contract every benchmark dataset+scorer implements."""

    name: str = "abstract"
    #: The version of the material this benchmark asks its questions against.
    #: :data:`~memrank.quality.DATASET_VERSION_UNSET` -- the default -- means the author did
    #: not state one. :data:`~memrank.quality.UNFREEZABLE` means the material CANNOT be
    #: frozen, and a run over it makes no reproducibility claim in its receipt. The two are
    #: deliberately different values: one is a gap, the other is an honest statement.
    dataset_version: str = DATASET_VERSION_UNSET
    # Catalog metadata for `evals show`. Declared per benchmark with no default:
    # a benchmark without one fails loudly the first time the catalog asks. Annotated rather
    # than `ClassVar` so a benchmark ASSEMBLED at runtime -- `ComposedEvaluation` -- can declare
    # its own; a class-body assignment and `cls.info` both read exactly as they did.
    info: EvalInfo
    # Whether the deterministic substring-recall proxy is a meaningful quality
    # signal for this benchmark. False for benchmarks whose gold answers are
    # prose / not verbatim spans (e.g. BEAM ideal_responses), where substring
    # recall is structurally ~0 and must be withheld in favor of an LLM judge.
    substring_recall_supported: bool = True
    # Whether ``composite`` is a self-contained quality score to DISPLAY and RANK
    # as-is, without an LLM judge. DECOUPLED from substring_recall_supported: a
    # graph benchmark sets substring_recall_supported=False (substring recall is
    # N/A) yet composite_rankable=True (its composite IS a real graph score).
    # False for benchmarks whose raw composite is not rankable without a judge
    # (e.g. BEAM prose golds) -- withheld from rankings, shown as "judge required".
    composite_rankable: bool = True
    # Name/kind of the composite metric, driving the displayed column label and
    # the leaderboard quality kind (e.g. "substring_recall", "graph_score").
    quality_metric: str = "substring_recall"
    # Whether this benchmark's data is synthetic (safe to send to an LLM judge
    # without explicit egress consent). Real datasets must be False so judged
    # runs require --ack-egress. Keyed on the benchmark, never on its name.
    is_synthetic: bool = False
    # Whether this run's question text comes from a canonical PUBLIC dataset and
    # may appear in a public default drill-in. False for any local-path override
    # (which may point at private data). Persisted into artifacts at run time.
    question_text_public: bool = False
    # Whether this benchmark needs an adapter graph snapshot to score. When True,
    # the runner SKIPS cells whose adapter is not ``graph_capable`` with a
    # ``not_applicable`` result instead of running (and crashing in) the scorer.
    requires_graph: bool = False
    # The benchmark protocol's reader-context policy, in the adapter-side ``context_budget``
    # vocabulary. "matched" -- the default -- leaves the shared --token-budget cap in force (the
    # leaderboard's fairness control). "uncapped" declares that this benchmark's own protocol
    # hands the reader everything retrieval returned (BEAM: every published harness runs
    # uncapped, so a capped run is not that benchmark). The runner promotes matched arms
    # accordingly; the no-memory arm ("none") is never promoted.
    context_policy: str = "matched"
    # The reader prompt this evaluation's answers are generated with: the benchmark's OFFICIAL
    # one where its authors publish one (memrank/benchmarks/answer_prompts.py), so a score follows
    # that benchmark's protocol. The default is memrank's own reader, for an evaluation with no
    # official prompt; demo and relation_graph declare it explicitly. Every run records which.
    answer_prompt: AnswerPrompt = MEMRANK_READER
    # The scored keys this benchmark's questions EXPECT their scorer to produce -- the
    # declaration `memrank.composition.ComposedEvaluation` checks the scorer's against when the
    # two halves are supplied separately (ATO-2149). Empty -- the default, and what all five
    # registered benchmarks declare -- means "whatever the scorer produces", which is what keeps
    # somebody else's criteria acceptable over memrank's questions. Declare it only when this
    # benchmark's own aggregation names a criterion literally and so cannot take another's.
    criterion_names: ClassVar[tuple[str, ...]] = ()
    # Task/scoring definition version (lm-eval-harness pattern). Bump on any
    # BREAKING change to how this benchmark loads or scores, so version-vs-version
    # comparison can flag "not directly comparable" instead of silently comparing
    # scores computed under different definitions. Distinct from ``dataset_version``
    # (the data) -- this versions the code/definition.
    VERSION: int = 0

    def __repr__(self) -> str:
        """Class, registry name, and the declared knobs -- never the data.

        Built from :meth:`config_for_receipt` (the same knobs artifacts record), so
        what a notebook shows and what a receipt hashes cannot disagree. I/O-free by
        the same contract as construction: rendering this must not call ``load()``.
        """
        knobs = " ".join(f"{key}={value}" for key, value in self.config_for_receipt().items()
                         if value is not None)
        detail = f" {knobs}" if knobs else ""
        return f"<{type(self).__name__} {self.name!r}{detail}>"

    def raw(self) -> list[dict[str, Any]]:
        """This benchmark's records exactly as upstream ships them, before normalisation.

        ``load()`` returns ``BenchmarkUnit``s -- the harness's shape, which every
        benchmark's upstream records have already been flattened, chunked and relabelled
        into. This returns what came off HuggingFace or GitHub, so a reader can see what
        the benchmark's authors actually published, including the fields loading
        deliberately drops (BEAM's ``user_profile`` and ``time_anchor``, LoCoMo's
        adversarial category 5). Uses THIS instance's configuration -- a
        ``BEAMBenchmark(tier="500k")`` answers for its own tier.

        Downloads, caches and digest-verifies exactly as a real run does, because it
        goes through the benchmark's own loader rather than a second copy of the fetch
        logic. First call on a cold cache may take minutes.
        """
        # Dispatch on what the benchmark HAS, not on its name: a renamed entry in the
        # registry would otherwise fall through to the error below while still being
        # perfectly readable.
        if (fetch := getattr(self, "_load_raw", None)) is not None:
            return fetch()                                            # beam, locomo, longmemeval
        if (fixtures := getattr(self, "_fixtures", None)) is not None:
            return list(fixtures())                                   # relation_graph
        if (path := getattr(self, "_path", None)) is not None:
            import json
            from pathlib import Path

            return [json.loads(Path(path).read_text(encoding="utf-8"))]  # demo, one scenario
        raise NotImplementedError(
            f"{self.name!r} exposes no raw records: it defines none of "
            f"_load_raw, _fixtures, _path")

    def config_for_receipt(self) -> dict[str, Any]:
        """Benchmark-specific reproducibility knobs to record in artifacts AND
        the receipt hash, so runs at different settings don't collide. The base
        captures the common ``tier``/``slice`` dimensions plus the task ``VERSION``;
        override to add more (it's part of the contract -- new dimensions must be
        exposed here)."""
        return {"tier": getattr(self, "tier", None),
                "slice": getattr(self, "slice", None),
                "task_version": self.VERSION}

    def rollup(self, per_unit_scores: list[dict[str, Any]], *,
               ranked: bool = True) -> dict[str, Any]:
        """Run-level metrics reduced from every unit's ``score()`` dict. Empty by default.

        The counterpart of :meth:`JudgeShape.aggregates` on the unjudged side. It exists because
        a per-unit metric nobody reduces is a metric nobody can read: LongMemEval's
        ``recall_all@k`` and LoCoMo's ``evidence_recall`` were both computed per unit, stored
        under ``per_unit``, and never turned into a number -- while ``composite`` (the only thing
        that WAS reduced) is None for both by design.

        Implementations must weight by each unit's own denominator rather than averaging unit
        means, so a unit covering three scoreable queries outweighs one covering a single query,
        and so units excluded from a metric contribute nothing instead of contributing a zero.

        ``ranked`` is the adapter's :attr:`Memory.ranks_results`. A rank-cut metric is
        undefined for an arm that returns the whole store unordered, and a benchmark declaring
        one must report None rather than a 0.0 that reads as retrieval failure.
        """
        return {}

    def judge_shape(self) -> JudgeShape:
        """How this benchmark's answers are graded under ``--judge``.

        The default is one answer, one verdict, one boolean -- right for any benchmark whose
        questions have a single gold answer, which is every one of them except BEAM. BEAM scores
        against a rubric of atomic nuggets, one judge call each, averaged within the question;
        that is a different SHAPE of judging, and overriding this is how a benchmark says so
        without the runner having to know which benchmark it is holding.

        The binary default carries ``GENERIC_BINARY_CATEGORIES`` -- the vocabulary of the demo
        benchmark and ad-hoc units only. A benchmark with a real category taxonomy overrides
        this and declares its own set (LoCoMo, LongMemEval), so its loader's labels and its
        judge gate cannot drift apart.

        Imported lazily: ``memrank.judging.shape`` imports ``memrank.judging.judge``, and a
        module-level import here would drag the judge into every consumer of ``core``.
        """
        from memrank.judging.shape import GENERIC_BINARY_CATEGORIES, BinaryJudgeShape
        return BinaryJudgeShape(GENERIC_BINARY_CATEGORIES)

    @abstractmethod
    def load(self) -> list[BenchmarkUnit]:
        """Return all benchmark units to score."""

    @abstractmethod
    def score(
        self,
        unit: BenchmarkUnit,
        responses: list[AdapterResponse],
    ) -> dict[str, Any]:
        """Score adapter responses for a single unit.

        Returns a dict with at minimum a ``composite`` float in [0, 1] and a
        ``per_category`` mapping when the benchmark has sub-categories.
        """

    @abstractmethod
    def report_template(self) -> str:
        """Markdown template used to render the per-benchmark report."""


#: A second name for `Benchmark`, left from the four-noun step, and NOT the public
#: `memrank.Evaluation`, which is a different class (`memrank/instrument/evaluation.py` --
#: tasks, measures, a clearing rule). The two contracts cannot share a name, so `Benchmark`
#: keeps its own: it is demoted rather than deprecated (ATO-2210), reached through
#: `memrank.evaluation(ref)` and taught only where the command line resolves an eval by name.
#: Kept because it is importable and nothing is being taken away; write `Benchmark` in new code.
Evaluation = Benchmark
