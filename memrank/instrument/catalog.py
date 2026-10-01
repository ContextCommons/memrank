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
"""`memrank.evaluation("demo")` and `memrank.system("tfidf")` -- the shipped names.

Two constructors, one idea: the things memrank ships are reached by the catalog name they are
printed under, so the first line a person copies stays inside the seven words. `system` hands
back a constructed shipped system; `evaluation` converts an in-tree benchmark into an
`Evaluation`.

memrank's own evaluations and a person's are the same kind of thing, so this produces exactly
what a person writes by hand: tasks, measures, a clearing rule. One group per unit, the unit's
documents as the group's context, and the benchmark's own `score()` wrapped as one measure
beside the measures that need no answer writer.

What does not convert is carried honestly rather than approximated: a benchmark whose span
proxy is meaningless (`substring_recall_supported = False`) ships no `WordMatch`, and nothing
here judges, because judging needs an answer and an answer needs a writer and a key.

`catalog()` is the third door and the one that presupposes nothing: it prints what ships, under
both the string names these two take and the Python names `memrank.systems` and
`memrank.evaluations` carry, which is where an editor can follow them.

`QUICK_START` is the one place the pair a first result is made of is declared. The table
`catalog()` prints opens with it, and the published pages that name the pair are held to it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from memrank.core import Benchmark, BenchmarkUnit
from memrank.instrument.evaluation import UNFREEZABLE, Clearing, Evaluation
from memrank.instrument.measure import Measure
from memrank.instrument.measures import BenchmarkScore, FailureRate, Latency, WordMatch
from memrank.instrument.system import System
from memrank.instrument.task import Expected, Task

#: Query keys the five registered loaders fill that are NOT part of `Task`'s own fields, and
#: that something downstream reads. Anything else a loader puts in a query dict is dropped.
CARRIED_METADATA = ("judge_prompt_key", "retrieval_scoreable", "ordering_tested",
                    "query_timestamp", "gold_ids", "expected_retrieval")


def _expected(query: dict[str, Any]) -> Expected:
    """What a correct outcome looks like, from the query dict the loader produced."""
    return Expected(
        answers=tuple(query.get("gold_answers") or ()),
        required_spans=tuple(query.get("required_spans") or ()),
        forbidden_spans=tuple(query.get("forbidden_spans") or ()),
        evidence_doc_ids=tuple(query.get("evidence_doc_ids") or ()),
        rubric=tuple(query.get("rubric") or ()),
        polarity="negative" if query.get("kind") == "negative" else "positive")


def task_from_query(query: dict[str, Any], unit: BenchmarkUnit) -> Task:
    """One query of one unit, as a task. The unit is the group, so the unit is the state."""
    return Task(
        id=str(query["id"]),
        prompt=str(query.get("text") or ""),
        expected=_expected(query),
        group=unit.isolation_id,
        context=tuple(unit.documents),
        category=query.get("category"),
        metadata={key: query[key] for key in CARRIED_METADATA if key in query})


def _version(benchmark: Benchmark) -> str:
    """The version of what this evaluation asks. A gap and a statement stay different values."""
    from memrank.quality import UNFREEZABLE as BENCHMARK_UNFREEZABLE

    declared = benchmark.dataset_version
    if declared == BENCHMARK_UNFREEZABLE:
        return UNFREEZABLE
    return f"{declared}+def{benchmark.VERSION}"


def _measures(benchmark: Benchmark, units: Sequence[BenchmarkUnit]) -> tuple[Measure, ...]:
    """The benchmark's own score, plus every measure that needs no answer writer."""
    shipped: list[Measure] = [BenchmarkScore(benchmark, units)]
    if benchmark.substring_recall_supported:
        shipped.append(WordMatch())
    shipped.extend((Latency(), FailureRate()))
    return tuple(shipped)


def from_benchmark(benchmark: Benchmark, units: Sequence[BenchmarkUnit] | None = None,
                   *, name: str | None = None) -> Evaluation:
    """Convert an in-tree benchmark into an evaluation. `units` defaults to `benchmark.load()`.

    Passing `units` is how a caller converts without loading -- the loaders that download do
    it in `load()`, and a conversion is not a reason to fetch a dataset.
    """
    loaded = list(units if units is not None else benchmark.load())
    tasks = tuple(task_from_query(query, unit) for unit in loaded for query in unit.queries)
    return Evaluation(
        name=name or benchmark.name,
        version=_version(benchmark),
        tasks=tasks,
        measures=_measures(benchmark, loaded),
        clearing=Clearing.PER_GROUP,
        metadata={"benchmark": benchmark.name,
                  "quality_metric": benchmark.quality_metric,
                  "task_version": benchmark.VERSION,
                  "units": len(loaded)})


def evaluation(ref: str | Benchmark, **overrides: Any) -> Evaluation:
    """The evaluation an in-tree ref names -- `memrank.evaluation("demo")`.

    ``ref`` takes the same grammar as the rest of memrank (`"beam:100k-smoke"`), or a
    `Benchmark` already built. Loading happens here, which is where a dataset may be fetched.

    An unknown ref names the known ones and points at `memrank.evaluations`, where the same
    evaluations are classes an editor can follow -- a reader who had to guess the string is the
    reader who did not know there was a catalog.
    """
    if isinstance(ref, Benchmark):
        return from_benchmark(ref)
    from memrank.benchmarks import from_ref
    from memrank.targets.resolve import RefError

    try:
        benchmark = from_ref(ref, **overrides)
    except RefError as exc:
        raise RefError(f"{exc}. Or import one from `memrank.evaluations`; "
                       f"`memrank.catalog()` prints them all.") from None
    return from_benchmark(benchmark)


def system(name: str, **options: Any) -> System:
    """The shipped system that catalog name names -- `memrank.system("tfidf")`.

    The names are the ones `memrank.catalog()` prints, and `options` goes straight to the
    constructor, which is how a client for a running service is given its `base_url=` and its
    `api_key=`. An unknown name is refused with the known ones named, because the alternative
    is a reader guessing at a spelling memrank could simply have told them.
    """
    from memrank.adapters import REGISTRY

    if name not in REGISTRY:
        known = ", ".join(sorted(REGISTRY))
        raise ValueError(f"Unknown system: {name!r}. memrank ships: {known}. Or import one "
                         f"from `memrank.systems`; `memrank.catalog()` prints them all.")
    return REGISTRY[name](**options)


@dataclass(frozen=True)
class ShippedSystem:
    """One system memrank ships, as `memrank.catalog()` reports it."""

    #: What to type in Python: `memrank.systems.TFIDF`.
    python_name: str
    #: What to type as a string: `memrank.system("tfidf")`, and what the CLI prints.
    name: str
    #: `memory` for a system under test, `control` for a floor or a ceiling.
    kind: str
    #: What has to be there before it runs -- nothing, an address, or an SDK.
    needs: str


@dataclass(frozen=True)
class ShippedEvaluation:
    """One evaluation memrank ships, as `memrank.catalog()` reports it."""

    #: What to type in Python: `memrank.evaluations.Demo`.
    python_name: str
    #: What to type as a string: `memrank.evaluation("demo")`.
    name: str
    #: What comes out of a run of it without a judge.
    measures: str
    #: What has to be there before it runs -- a download, an environment variable, nothing.
    needs: str


@dataclass(frozen=True)
class QuickStart:
    """The pair a first result is made of: one shipped system on one shipped evaluation."""

    #: The Python name in `memrank.systems`.
    system: str
    #: The Python name in `memrank.evaluations`.
    evaluation: str


#: The quick start, declared once. `README.md` and `docs/install.md` run exactly this pair,
#: `catalog()` prints it as the example, and `tests/repo/test_the_quick_start_is_named_once.py`
#: holds every published page and every docstring that names the pair to it. A demotion -- the
#: way `WordOverlap` and `Demo` stopped being the pair in 0.4.4 -- is this one edit, and the
#: guard then names every sentence still giving the role to the old pair.
QUICK_START = QuickStart(system="TFIDF", evaluation="Demo")


def _named(entries: Sequence[Any], python_name: str) -> Any:
    """The entry with that Python name. A declared name nothing ships is refused, not skipped."""
    for entry in entries:
        if entry.python_name == python_name:
            return entry
    raise LookupError(f"QUICK_START names {python_name!r}, which memrank does not ship")


def _system_lines(shipped: Sequence[ShippedSystem]) -> list[str]:
    """One line per system: the Python name, the string name, the kind, what it needs."""
    width = max(len(entry.python_name) for entry in shipped)
    kind_width = max(len(entry.kind) for entry in shipped)
    return [f"  {e.python_name:<{width}}  {f'{e.name!r}':<16}  {e.kind:<{kind_width}}  {e.needs}"
            for e in shipped]


def _evaluation_lines(shipped: Sequence[ShippedEvaluation]) -> list[str]:
    """Two lines per evaluation: the names and what it needs, then what it measures."""
    width = max(len(entry.python_name) + 2 for entry in shipped)
    lines = []
    for entry in shipped:
        call = f"{entry.python_name}()"
        lines.append(f"  {call:<{width}}  {f'{entry.name!r}':<16}  {entry.needs}")
        lines.append(f"  {'':<{width}}  {'':<16}  measures {entry.measures}")
    return lines


@dataclass(frozen=True)
class Catalog:
    """Everything memrank ships, under the names a reader can command-click."""

    systems: tuple[ShippedSystem, ...]
    evaluations: tuple[ShippedEvaluation, ...]

    def quick_start(self) -> tuple[ShippedSystem, ShippedEvaluation]:
        """The entries `QUICK_START` names -- the pair a first result is made of."""
        return (_named(self.systems, QUICK_START.system),
                _named(self.evaluations, QUICK_START.evaluation))

    def __str__(self) -> str:
        first_system, first_evaluation = self.quick_start()
        return "\n".join([
            f"memrank ships {len(self.systems)} system(s) and "
            f"{len(self.evaluations)} evaluation(s).",
            "",
            f"systems -- `from memrank.systems import {first_system.python_name}`, "
            f"or `memrank.system(\"{first_system.name}\")`",
            "",
            *_system_lines(self.systems),
            "",
            f"evaluations -- `from memrank.evaluations import {first_evaluation.python_name}`, "
            f"or `memrank.evaluation(\"{first_evaluation.name}\")`",
            "",
            *_evaluation_lines(self.evaluations),
        ])


def catalog() -> Catalog:
    """Print what memrank ships, and return it -- `memrank.catalog()`.

    The one call a reader who knows no names can type. `system(name)` and `evaluation(name)`
    both presuppose the string; this is where the strings come from, beside the Python names
    that lead to the class and the function themselves.

    It prints AND returns, because the two readers are different: a person runs it for the
    table, and a program reads `catalog().systems` for the entries.
    """
    # `term.style` deferred, not module-level: it imports typer, and `tests/repo/
    # test_import_weight.py` holds `import memrank` to a graph with no terminal stack in it. A
    # person who calls `catalog()` has asked for the terminal; a library caller never reaches
    # this line.
    from memrank import evaluations as shipped_evaluations
    from memrank import systems as shipped_systems
    from memrank.term import style

    printed = Catalog(systems=shipped_systems.SHIPPED,
                      evaluations=shipped_evaluations.SHIPPED)
    # The table IS the answer here rather than narration about producing it, so `out` (stdout)
    # and not `say`. Through the chokepoint because `tests/term/test_output_streams.py` allows
    # no module but `term/style.py` to name a stream.
    style.out(str(printed))
    return printed
