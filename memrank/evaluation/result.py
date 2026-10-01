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
#
# Used by the current path (runs/registry): it sits in a deprecated package but is not
# deprecated itself, and it must not warn.
"""The typed result of one evaluated cell -- and the artifact schema's single owner.

``EvalResult.to_dict()`` emits the per-cell artifact dict byte-for-byte as the pre-split
``_aggregate_cell`` built it (``aggregate.build_result`` constructs the instance, and the
dict form is consumed as-is by ten downstream modules: registry, record, leaderboard,
API projection, the uploader). Two rules keep that contract honest, both pinned by
``tests/orchestration/test_golden_run.py``:

- ``judged_metrics`` is ABSENT when no judge ran, never null -- absent means "no judge
  ran"; null would claim a judge ran and produced nothing.
- ``rollup`` and ``benchmark_config`` are spread LAST, so a benchmark-declared key
  shadows a closed one exactly as the two trailing ``**`` spreads always have. The spread
  position is unchanged and still pinned there -- what changed is that a key which WOULD
  shadow a closed field is now refused at assembly (:class:`ShadowedResultField`) instead of
  replacing it silently.

**THE KEYS ARE NAMED HERE AND NOWHERE ELSE.** A result dict's identity keys --
``adapter`` and ``benchmark`` -- were read by literal subscript in nine modules, so every
one of them was a second place the artifact vocabulary was interpreted and a tenth edit
whenever it changes. The four accessors below are the only place those two strings appear,
and ``tests/repo/test_result_vocabulary.py`` enumerates every module to keep it that way.
Translating them into the OUTWARD vocabulary is a separate, single job, and
:func:`memrank.api.run_projection.identity_of` is the one place it happens.

Strict and tolerant are separate functions rather than one with a default, because which of
the two a caller wants is a real distinction here: a curator pairing battles must fail on a
row with no engine, while a listing built from artifacts on disk must render a half-written
cell rather than raise out of the whole listing. A single accessor with a default would make
every caller look tolerant and hide which ones actually are.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Any

from memrank.errors import MemrankError
from memrank.evaluation.case import CaseRow, rows_of
from memrank.evaluation.constants import DEFAULT_JUDGE_WORKERS
from memrank.evaluation.score import Score, score_of

#: The artifact's two identity keys. Referenced by the accessors below rather than written at
#: each call site, so renaming the artifact's own vocabulary is an edit to these two lines.
_ADAPTER_KEY = "adapter"
_BENCHMARK_KEY = "benchmark"


def adapter_of(result: Mapping[str, Any]) -> Any:
    """The engine this result measured. Raises ``KeyError`` when the result names none."""
    return result[_ADAPTER_KEY]


def benchmark_of(result: Mapping[str, Any]) -> Any:
    """The evaluation this result ran. Raises ``KeyError`` when the result names none."""
    return result[_BENCHMARK_KEY]


def adapter_if_present(result: Mapping[str, Any]) -> Any:
    """The engine this result measured, or ``None`` on a result that names none."""
    return result.get(_ADAPTER_KEY)


def benchmark_if_present(result: Mapping[str, Any]) -> Any:
    """The evaluation this result ran, or ``None`` on a result that names none."""
    return result.get(_BENCHMARK_KEY)


class ShadowedResultField(MemrankError):
    """A benchmark declared a run-level key that a fixed result field already owns.

    The two open surfaces -- ``rollup()`` and ``config_for_receipt()`` -- are spread LAST into
    the artifact dict, so before this existed a benchmark that happened to name one of its
    keys ``composite`` (or ``k``, or ``receipt``) replaced the result's own value and nothing
    said so: the artifact stayed well-formed, the run reported success, and the number a
    reader ranked on was the benchmark's rather than the measurement's. Every fixed field
    added to the result widens that surface, which is why this is loud rather than a note.
    """


@dataclass(frozen=True)
class EvalResult:
    """One (adapter x benchmark) cell's complete result.

    Frozen: a result is a record of a measurement, and nothing downstream may edit one in
    place. The one post-hoc addition the run lifecycle makes -- the target ref -- goes
    through ``dataclasses.replace``.
    """

    adapter: str
    benchmark: str
    composite: float | None
    #: ``"not_applicable"`` for the cell a capability gate skipped; ``None`` for a cell
    #: that ran. The skipped shell serializes as exactly five keys.
    status: str | None = None
    reason: str | None = None
    per_unit: list[dict[str, Any]] = field(default_factory=list)
    per_query: list[dict[str, Any]] = field(default_factory=list)
    ingested_documents: list[dict[str, Any]] = field(default_factory=list)
    latency_metrics: dict[str, Any] = field(default_factory=dict)
    retrieve_latency_summary: dict[str, Any] | None = None
    ingest_latency_summary: dict[str, Any] | None = None
    ingest_throughput: dict[str, float] | None = None
    token_metrics: dict[str, Any] = field(default_factory=dict)
    corpus_documents: int = 0
    corpus_bytes: int = 0
    corpus_tokens: int = 0
    workers: int = 1
    latency_contended: bool = False
    judge_workers: int = DEFAULT_JUDGE_WORKERS
    context_tokens_mean: float = 0.0
    est_dollars_per_query: float = 0.0
    est_dollars_per_cell: float = 0.0
    cost_basis: dict[str, Any] = field(default_factory=dict)
    cleanup_is_destructive: bool | None = None
    substring_recall_supported: bool = True
    composite_rankable: bool = True
    quality_metric: str = "substring_recall"
    question_text_public: bool = False
    receipt: dict[str, Any] = field(default_factory=dict)
    n_units: int = 0
    k: int = 0
    repeats: int = 0
    #: One record per unit ATTEMPTED, in unit order: ``{"unit_id", "outcome"}`` for a unit that
    #: ran, plus ``stage``/``error``/``message`` for one that did not. Empty for a cell built
    #: without them (``_aggregate_cell``'s legacy callers), which is why the three counts below
    #: are read from this list rather than from ``n_units``.
    unit_outcomes: list[dict[str, Any]] = field(default_factory=list)
    #: Units attempted, units lost, and the ratio -- beside the composite because the composite
    #: is a mean over the units that RAN. 50 of 385 units scored is not a score over 385, and
    #: without these nothing in the artifact says which it is. ``unit_failure_rate`` is None for
    #: a cell with no units, on the same None-vs-0.0 rule the composite follows.
    units_total: int = 0
    units_failed: int = 0
    unit_failure_rate: float | None = None
    #: ``benchmark.rollup(...)`` -- run-level metrics the benchmark reduces itself.
    #: Open keys: the benchmark names them, the schema does not.
    rollup: dict[str, Any] = field(default_factory=dict)
    #: ``benchmark.config_for_receipt()`` -- the benchmark-specific knobs (tier, slice,
    #: task_version, ...). Open keys, same contract as ``rollup``.
    benchmark_config: dict[str, Any] = field(default_factory=dict)
    judged_metrics: dict[str, Any] | None = None
    #: The target REF this cell ran as, injected by the run lifecycle after the fact --
    #: the eval loop measures an adapter and does not know its catalog name.
    target: str | None = None

    @property
    def engine(self) -> str:
        """The engine this result measured, under the noun the four-noun model uses.

        ``adapter`` is the field, because it is also the artifact's key and that key is
        renamed behind a schema version by a later step. This is the name a person reads
        and writes in Python, available now and unaffected by that rename.
        """
        return self.adapter

    @property
    def evaluation(self) -> str:
        """The evaluation this result applied, under the noun the four-noun model uses."""
        return self.benchmark

    @property
    def engine_ref(self) -> str | None:
        """The catalog name this engine ran as, or ``None`` when an instance was passed.

        The same value as the ``target`` field, which is the command line's noun for it --
        in machine learning a *target* is the expected answer, and in operations it is a
        deployment destination, so neither reading is what this holds.
        """
        return self.target

    @property
    def is_applicable(self) -> bool:
        return self.status != "not_applicable"

    @property
    def cases(self) -> tuple[CaseRow, ...]:
        """Every case measured, typed -- what was asked, recalled, answered and decided.

        A view over ``per_query``, derived on each read and stored nowhere: the artifact
        dict is unchanged, and so is every allowlist that selects keys out of it. Use this
        to read a run; use ``per_query`` only where the artifact's own shape is the point.
        """
        return rows_of(self.per_query)

    @property
    def score(self) -> Score:
        """The headline number and the honest declarations that travel with it.

        What the number is OF and what it is NOT lived in
        ``core.QUALITY_METRIC_DESCRIPTIONS``, keyed by ``quality_metric`` and reachable from
        the terminal only, so ``quality_metric: "substring_recall"`` had to speak for itself.
        It does not, which is why that constant exists. This carries it.
        """
        return score_of(self.to_dict())

    @property
    def reproducible(self) -> bool:
        """Whether this run can claim reproducibility at all.

        ``False`` for a benchmark whose material CANNOT be frozen
        (:data:`~memrank.quality.UNFREEZABLE`), which is a different statement from a
        benchmark author who never set a dataset version. The receipt says the same thing;
        this is the same fact reachable from the result.
        """
        return bool(self.receipt.get("reproducible", True))

    @property
    def headline(self):
        """The one summary figure readers rank on -- `metrics.headline` owns the rule."""
        from memrank.metrics import headline

        return headline.cell_headline(self.to_dict())

    @classmethod
    def not_applicable(cls, adapter_name: str, benchmark_name: str, *,
                       reason: str) -> EvalResult:
        return cls(adapter=adapter_name, benchmark=benchmark_name, composite=None,
                   status="not_applicable", reason=reason)

    def __post_init__(self) -> None:
        """Refuse a benchmark-declared key that a fixed field already owns.

        At assembly rather than in ``to_dict``, because assembly is where the benchmark that
        declared the key is still identifiable and where nothing has consumed the result yet.
        ``to_dict`` is called repeatedly by several readers; a refusal belongs at the one
        point the result comes into being.
        """
        self._refuse_shadowing("rollup", self.rollup)
        self._refuse_shadowing("config_for_receipt", self.benchmark_config)

    def _refuse_shadowing(self, surface: str, declared: dict[str, Any]) -> None:
        """Raise when ``declared`` names a fixed field, saying which benchmark and which key.

        ``surface`` is the benchmark METHOD that produced the keys, because that is what the
        author has to edit -- "a rollup key collides" sends someone to the wrong method half
        the time.
        """
        shadowed = sorted(set(declared) & _CONTRACT_FIELDS)
        if not shadowed:
            return
        keys = ", ".join(repr(key) for key in shadowed)
        raise ShadowedResultField(
            f"benchmark {self.benchmark!r} declared {surface}() key(s) {keys}, which the "
            f"result already owns as fixed field(s) of its own. Those keys are spread into "
            f"the artifact last, so each would have replaced a measured value silently. "
            f"Rename them in {self.benchmark}'s {surface}(); the names that are taken are "
            f"the fields of memrank.evaluation.result.EvalResult.")

    def to_dict(self) -> dict[str, Any]:
        """The exact artifact dict, in the construction order the schema has always had."""
        if self.status == "not_applicable":
            return {
                _ADAPTER_KEY: self.adapter,
                _BENCHMARK_KEY: self.benchmark,
                "status": "not_applicable",
                "composite": None,
                "reason": self.reason,
            }
        out: dict[str, Any] = {
            _ADAPTER_KEY: self.adapter,
            _BENCHMARK_KEY: self.benchmark,
            "composite": self.composite,
            "per_unit": self.per_unit,
            "per_query": self.per_query,
            "ingested_documents": self.ingested_documents,
            "latency_metrics": self.latency_metrics,
            "retrieve_latency_summary": self.retrieve_latency_summary,
            "ingest_latency_summary": self.ingest_latency_summary,
            "ingest_throughput": self.ingest_throughput,
            "token_metrics": self.token_metrics,
            "corpus_documents": self.corpus_documents,
            "corpus_bytes": self.corpus_bytes,
            "corpus_tokens": self.corpus_tokens,
            "workers": self.workers,
            "latency_contended": self.latency_contended,
            "judge_workers": self.judge_workers,
            "context_tokens_mean": self.context_tokens_mean,
            "est_dollars_per_query": self.est_dollars_per_query,
            "est_dollars_per_cell": self.est_dollars_per_cell,
            "cost_basis": self.cost_basis,
            "cleanup_is_destructive": self.cleanup_is_destructive,
            "substring_recall_supported": self.substring_recall_supported,
            "composite_rankable": self.composite_rankable,
            "quality_metric": self.quality_metric,
            "question_text_public": self.question_text_public,
            "receipt": self.receipt,
            "n_units": self.n_units,
            "k": self.k,
            "repeats": self.repeats,
            "unit_outcomes": self.unit_outcomes,
            "units_total": self.units_total,
            "units_failed": self.units_failed,
            "unit_failure_rate": self.unit_failure_rate,
            **self.rollup,
            **self.benchmark_config,
        }
        if self.judged_metrics is not None:
            out["judged_metrics"] = self.judged_metrics
        if self.target is not None:
            out["target"] = self.target
        return out


#: Every field the result owns itself -- the closed half of the schema, against which the two
#: open surfaces are checked. Derived from the dataclass rather than written out again, so a
#: field added to the class above is protected by the guard without a second edit; that is the
#: whole point, since it is added fields that enlarge the hazard. The two open surfaces are
#: excluded because they are the containers being checked, not fields a key can shadow.
_CONTRACT_FIELDS: frozenset[str] = frozenset(
    f.name for f in fields(EvalResult) if f.name not in ("rollup", "benchmark_config"))
