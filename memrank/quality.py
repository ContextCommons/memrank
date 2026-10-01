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
"""What a quality number IS, what it is NOT, and when material cannot be frozen.

The honest caveat about memrank's cheapest number -- *retrieval recall (a substring proxy),
not end-to-end answer correctness* -- was one string in a dict keyed by metric name, read by
the terminal and by nothing else. A person holding a result could not reach it, so the
result's ``quality_metric: "substring_recall"`` had to be self-explanatory, which is the
thing that very constant exists to deny.

:class:`MetricDeclaration` splits that prose into its parts, so a result can carry them by
name. ``core.QUALITY_METRIC_LABELS`` and ``core.QUALITY_METRIC_DESCRIPTIONS`` are DERIVED
from the table below and are byte-identical to the literals they replaced --
``tests/core/test_quality_declarations.py`` pins that -- so every existing reader is
untouched.

A leaf module on purpose: it imports nothing from memrank, so ``core`` (which owns the
benchmark contract) and ``evaluation.result`` (which owns the artifact) can both read it
without either importing the other.
"""
from __future__ import annotations

from dataclasses import dataclass

#: The ``Benchmark.dataset_version`` a benchmark sets when its material CANNOT be frozen --
#: a live corpus, a generated one, anything whose content is not fixed at a version. Distinct
#: from :data:`DATASET_VERSION_UNSET`, which means only that nobody said. The difference
#: matters because it is the difference between a reproducibility claim nobody made and one
#: that cannot honestly be made at all.
UNFREEZABLE = "unfreezable"

#: The default ``Benchmark.dataset_version``: the author did not state one.
DATASET_VERSION_UNSET = "unknown"


@dataclass(frozen=True)
class MetricDeclaration:
    """What one kind of quality number measures, and what it does not establish.

    ``of`` and ``not_of`` are the two halves of the caveat prose. ``withheld`` carries the
    other kind of caveat -- a number this benchmark only has when a judge produced it -- and
    the two are mutually exclusive in practice: a metric that IS the judge's verdict has
    nothing to disclaim, and one that is a proxy is always available.
    """

    label: str
    of: str
    not_of: str | None = None
    withheld: str | None = None

    @property
    def description(self) -> str:
        """The one-line caveat, exactly as ``QUALITY_METRIC_DESCRIPTIONS`` has always read."""
        if self.not_of is not None:
            return f"{self.of}, not {self.not_of}"
        if self.withheld is not None:
            return f"{self.of} ({self.withheld})"
        return self.of


#: Caveat for a run whose headline is withheld. Not a metric a benchmark declares: it is what
#: :func:`declaration_for` answers with when the rule in ``metrics.headline`` produced no
#: number, so a reader is never handed a declaration about a number that does not exist.
WITHHELD_DECLARATION = MetricDeclaration(
    label="withheld",
    of="no quality number",
    withheld="this run produced none that may be shown; a judged run is what produces one here",
)

#: The judge's verdict, for a benchmark that produced one. ``metrics.headline`` reports this
#: kind for EVERY judged run whatever benchmark ran, because the runner writes rubric averages
#: and binary verdicts alike into ``answer_correctness``.
_JUDGED_BY_DEFAULT = "judged by default; `--no-judge` reports no quality score"

QUALITY_METRICS: dict[str, MetricDeclaration] = {
    "substring_recall": MetricDeclaration(
        label="recall",
        of="retrieval recall (a substring proxy)",
        not_of="end-to-end answer correctness"),
    # Unlike the answer substring proxy, this checks only the known source passage;
    # the scorer never inspects an answer.
    "passage_recall": MetricDeclaration(
        label="passage recall",
        of="full-passage retrieval recall",
        not_of="answer-span or end-to-end answer correctness"),
    "graph_score": MetricDeclaration(
        label="graph score",
        of="a relation-graph structural score (graph correctness)",
        not_of="retrieval coverage or answer correctness"),
    # The external benchmarks' own protocols. They ship no self-contained number, which is why
    # these evals judge by default rather than asking -- and why `--no-judge` yields a run with
    # no quality score at all rather than a lesser one.
    "judged_answer_correctness": MetricDeclaration(
        label="judged correctness",
        of="LLM-judged answer correctness against the dataset's reference answer",
        withheld=_JUDGED_BY_DEFAULT),
    "judged_nugget_rubric": MetricDeclaration(
        label="judged rubric",
        of="LLM-judged rubric coverage, one call per atomic nugget",
        withheld=_JUDGED_BY_DEFAULT),
}


def declaration_for(kind: str) -> MetricDeclaration:
    """The declaration for a metric kind, or one saying the number is unnamed.

    Never raises and never falls back to a WRONG declaration: a kind this memrank does not
    know -- a benchmark from a newer version, or a plugin's own -- gets a declaration that
    names it and claims nothing about it, which is the honest answer and not a degraded one.
    """
    known = QUALITY_METRICS.get(kind)
    if known is not None:
        return known
    return MetricDeclaration(
        label=kind,
        of=f"{kind}, a metric this memrank carries no declaration for",
        not_of="anything this result can state on its behalf")


def is_reproducible(dataset_version: str) -> bool:
    """Whether a run over this material can claim reproducibility at all."""
    return dataset_version != UNFREEZABLE
