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
"""The seven are the Python surface, and the names they replaced cannot creep back.

`tests/repo/test_docs_teach_the_current_cli.py` records why this is enumerated rather than
left to per-file vigilance: the CLI's cutover was done in three passes, each one missed what
the last had touched, and sixteen call sites were still printing refused commands months
later. This is the same control for the Python surface, and it has the same two halves.

**The surface.** `memrank.__all__` is pinned to the seven, the kinds a person subclasses, the
measures memrank ships and the two readings above the run -- so a name cannot join the first
thing a person reads without a deliberate edit here -- and the run is a verb on the evaluation,
`Evaluation.run(system)`, whose first parameter is checked by position. The previous run keeps
its own contract, checked here too: it is `memrank.evaluation.api.run`, its first two parameters
are `engine` and `evaluation`, and every deprecated spelling is keyword-only and named in
:data:`memrank.evaluation.api.DEPRECATED_PARAMETERS`. A third spelling fails here.

**The corpus.** Every live document, example and script is scanned for the spellings this
step retired. A retired spelling in an instruction is a person typing it tomorrow.

*Retired* here is about what a document may teach, not about what still imports -- except for
`memrank.run(...)`, which is retired in both senses (ATO-2247): decision 0008 chose
`evaluation.run(system)`, the 0.4.x build shipped the free function anyway, and this is
greenfield, so the free function left the package surface rather than staying as an alias. The
nine shipped system classes are the other shape: they dropped their `Adapter` suffix
(ATO-2220) and the old spellings are kept as aliases for code outside this repository;
:data:`SUFFIXED_ALIASES` names all nine, which both feeds the corpus scan and asserts each one
still resolves to the same class object. So a document that teaches `WordOverlapAdapter` fails,
and a checkout that has quietly deleted it fails too -- the second is what tells a retired
spelling apart from a removed name.

**The operator layer.** A third table names what is importable from `memrank` and deliberately
off `__all__`, and asserts the entry path never makes a reader meet it. `Benchmark` is the case
that table exists for (ATO-2210): it could not be renamed to `Evaluation` -- a different class
holds that name -- so it was demoted instead, and demotion, unlike a rename, leaves nothing
behind that fails when it is undone. `memrank.evaluation(ref)` is the route to a named
evaluation; `Benchmark` is what that route resolves one to.

**What is deliberately not here.** `MemoryAdapter`, `MemoryEngine` and `EvalResult` are
*deprecated*, not retired: they still work, they may still be named in a document that is
explaining the deprecation, and ATO-2151 is what removes them and extends the table below.
`Benchmark` is neither -- it is demoted and it stays. And `memrank submit TARGET EVAL` keeps
its own nouns -- *target* is the command line's word for a catalog entry, this step did not
change it. Only Python spellings are
scanned, never a file name.
"""
from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest

import memrank
import memrank.adapters
import memrank.core
from memrank.evaluation.api import DEPRECATED_PARAMETERS
from memrank.evaluation.api import run as cell_run
from memrank.evaluation.result import EvalResult

ROOT = Path(__file__).resolve().parents[2]

#: The whole of what `import memrank` advertises: the seven, the kinds a person subclasses,
#: the measures memrank ships, the two readings above the run, the three value types a task or
#: a recall is written with (`Document`, `Expected`, `Recall`), the version, and the three
#: doors onto what ships -- `catalog()` and the `systems` and `evaluations` modules, which are
#: the same things under Python names an editor can follow. Spelled out rather than counted, so
#: that adding a name is a visible edit to this list and a reader can see what the surface is
#: without running anything.
SURFACE = ("Assistant", "Clearing", "Decider", "Document", "Evaluation", "Expected",
           "FailureRate", "Judge", "Latency", "Measure", "Memory", "Model", "Paired", "Recall",
           "Result", "Retriever", "Scope", "System", "Task", "Trace", "Value", "WordMatch",
           "__version__", "catalog", "evaluation", "evaluations", "measure", "paired",
           "system", "systems")

#: The positional parameter of the run verb. The evaluation is `self`, which is the whole point
#: of the verb living on it: the one thing left to name is the system.
POSITIONAL = ("system",)
#: And of the previous run, which the command line and the cloud still call.
CELL_POSITIONAL = ("engine", "evaluation")

#: ``{alias: the name it is an alias of}``. The nine shipped system classes dropped the
#: `Adapter` suffix (ATO-2220). Each old spelling is still importable from
#: `memrank.adapters` -- out-of-tree code that names one keeps working, and removing them is
#: ATO-2151 -- but no document or script may teach one, so each also appears in `RETIRED`
#: below. Naming them here is what keeps *retired spelling* from drifting into *removed name*:
#: a deleted alias fails here rather than in somebody else's checkout.
SUFFIXED_ALIASES = {
    "AtomicMemoryAdapter": "AtomicMemory",
    "FixedContextAdapter": "FixedContext",
    "FullContextAdapter": "FullContext",
    "HindsightAdapter": "Hindsight",
    "Mem0Adapter": "Mem0",
    "NativeAdapter": "Native",
    "NoContextAdapter": "NoContext",
    "SupermemoryAdapter": "Supermemory",
    "WordOverlapAdapter": "WordOverlap",
}

#: ``{regex: what to write instead}``. Each pattern is a PYTHON spelling this step retired.
RETIRED = {
    r"\bmemrank\.run\(": "evaluation.run(system=...)",
    r"\brun\(\s*target\s*=": "run(engine=...)",
    r"\brun\([^)]*\beval\s*=": "run(evaluation=...)",
    r"\bresult\.target\b": "result.engine_ref",
    r"\bresult\.adapter\b": "result.engine",
    r"\bresult\.benchmark\b": "result.evaluation",
    r"\bmemrank\.EvalResult\b": "memrank.Result",
    **{rf"\b{alias}\b": current for alias, current in SUFFIXED_ALIASES.items()},
}
RETIRED_PATTERNS = {re.compile(pattern): instead for pattern, instead in RETIRED.items()}

#: Importable from `memrank` and deliberately absent from `SURFACE`: the previous surface, which
#: the command line and the cloud call and a reader of the entry path never has to meet. Named
#: here so that putting one back on the front page is an edit to this list rather than a thing
#: nobody notices. `Benchmark` is demoted, not deprecated -- it keeps its name and still works.
OPERATOR_LAYER = ("AdapterResponse", "Benchmark", "BenchmarkUnit", "ComposedEvaluation",
                  "EvalInfo", "EvalResult", "MemoryAdapter", "MemoryEngine", "Scorer",
                  "SpanRecall")

#: The same layer, in names that are also ordinary English words. Their import is checked with
#: the rest; the prose scan cannot see them, because `benchmark` in a sentence about what `demo`
#: is means the word and not `memrank.benchmark`, and a scan that cannot tell those apart fails
#: on prose nobody should change.
OPERATOR_LAYER_UNSCANNABLE = ("benchmark",)

#: Layer 1 of the information architecture: what a person reads before they have chosen to go
#: deeper. `docs/misc/` -- where the command line and the catalog are documented -- is not here,
#: because that is exactly where the operator layer is allowed to be taught.
ENTRY_PATH = ("README.md", "docs/install.md", "docs/reference/*.md")

#: Where instructions live -- prose a reader may copy from, and scripts they may run.
LIVE_GLOBS = ("*.md", "docs/**/*.md", "examples/**/*.md", "notebooks/**/*.md",
              "examples/**/*.py", "notebooks/**/*.py", "scripts/*.py")

#: Records of what was typed at the time, not instructions. Rewriting one would falsify the
#: record it exists to preserve -- the same exclusion, for the same reason, as the CLI test's.
HISTORICAL: tuple[str, ...] = ("tech-debt.md", "scripts/internal/one-offs/")
DATED = re.compile(r"/\d{4}-\d{2}-\d{2}-")


def _live_files() -> list[Path]:
    found: set[Path] = set()
    for glob in LIVE_GLOBS:
        found.update(p for p in ROOT.glob(glob) if p.is_file())
    live = []
    for path in found:
        relative = path.relative_to(ROOT).as_posix()
        if any(relative.startswith(h) for h in HISTORICAL) or DATED.search("/" + relative):
            continue
        live.append(path)
    return sorted(live)


def _offences(path: Path) -> list[str]:
    """Every retired spelling this file would have a reader write."""
    found = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        for pattern, instead in RETIRED_PATTERNS.items():
            if pattern.search(line):
                found.append(
                    f"{path.relative_to(ROOT)}:{number} teaches `{pattern.pattern}` -- "
                    f"write {instead}")
    return found


@pytest.mark.parametrize("path", _live_files(), ids=lambda p: p.relative_to(ROOT).as_posix())
def test_no_live_file_teaches_a_retired_python_name(path: Path):
    """A document that teaches the old vocabulary is what puts it back in someone's code."""
    offences = _offences(path)
    assert not offences, "\n".join(offences)


#: Files the scan must reach, or every case above passes vacuously.
MUST_COVER = ("README.md", "examples/02-your-own-system/run.py", "docs/systems.md")


def test_the_scan_reaches_the_files_that_teach_the_python_surface():
    """Guards the guard: an over-eager exclusion would empty the corpus silently."""
    scanned = {p.relative_to(ROOT).as_posix() for p in _live_files()}
    for must_cover in MUST_COVER:
        assert must_cover in scanned, f"{must_cover} fell out of the live corpus"


def test_the_package_surface_is_the_seven():
    """What a person meets on `import memrank`, pinned by name."""
    assert tuple(memrank.__all__) == SURFACE


def _positional(function) -> tuple[str, ...]:
    return tuple(p.name for p in inspect.signature(function).parameters.values()
                 if p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD)


def test_the_run_is_a_verb_on_the_evaluation_taking_the_system():
    """`evaluation.run(system=...)`: the evaluation is the subject, the system the argument.

    Checked on a BOUND method, which is what a caller holds: `self` is already the evaluation,
    so the signature a person reads has exactly one positional name left, and it is the system.
    """
    bound = memrank.Evaluation(name="any", version="1").run
    assert _positional(bound) == POSITIONAL
    assert memrank.Evaluation.run.__module__ == "memrank.instrument.evaluation"


def test_the_free_run_function_is_off_the_package_surface():
    """Greenfield, not deprecated: `memrank.run` is gone, and nothing answers to it."""
    assert not hasattr(memrank, "run"), (
        "memrank.run is the free function decision 0008 rejected; the verb is Evaluation.run")


def test_the_previous_run_keeps_its_own_nouns_where_the_cli_and_the_cloud_call_it():
    assert _positional(cell_run) == CELL_POSITIONAL


def test_every_deprecated_parameter_is_keyword_only_and_declared():
    """A deprecated spelling may be accepted, never offered a position.

    Both halves matter. Keyword-only is what stops `run(x, y)` meaning the old thing, and the
    declaration is what stops a third spelling arriving without an edit to the table step 22
    removes.
    """
    parameters = inspect.signature(cell_run).parameters
    for old_name, current in DEPRECATED_PARAMETERS.items():
        assert old_name in parameters, f"{old_name}= must keep working until step 22"
        assert parameters[old_name].kind is inspect.Parameter.KEYWORD_ONLY
        assert current in parameters, f"{old_name} is deprecated in favour of {current}"
    undeclared = {name for name in parameters
                  if name in RETIRED and name not in DEPRECATED_PARAMETERS}
    assert undeclared == set(), f"retired parameter names not declared deprecated: {undeclared}"


def test_the_result_answers_to_the_nouns():
    """A reader asks a result which engine and which evaluation, in those words."""
    result = EvalResult(adapter="word-overlap", benchmark="demo", composite=1.0, target="ref")
    assert (result.engine, result.evaluation, result.engine_ref) == ("word-overlap", "demo", "ref")


def test_every_retired_suffixed_spelling_still_imports_as_the_same_class():
    """Retired from the documents, not from the package: out-of-tree code keeps working.

    Deleting an alias is ATO-2151 and is a release-note change; doing it by accident here
    would break every caller the rename pushed onto the new name, silently.
    """
    for alias, current in SUFFIXED_ALIASES.items():
        old = getattr(memrank.adapters, alias, None)
        assert old is not None, f"memrank.adapters.{alias} stopped importing"
        assert old is getattr(memrank.adapters, current), (
            f"{alias} must be the same class object as {current}, not a second class beside it")


def test_the_deprecated_exports_are_the_same_objects():
    """Not near-copies: the same class, so a subclass written against either name is one type.

    A parallel hierarchy would pass a name check and fail every `isinstance` in the registry.
    """
    assert memrank.MemoryEngine is memrank.MemoryAdapter
    assert memrank.Memory is memrank.MemoryAdapter, (
        "the memory kind is the memory contract, not a second class beside it")
    assert memrank.Benchmark is memrank.core.Evaluation
    assert memrank.Result is not memrank.EvalResult, (
        "the new Result is the read form of the new run; the stored artifact is EvalResult")


def _entry_path_files() -> list[Path]:
    found: set[Path] = set()
    for glob in ENTRY_PATH:
        found.update(p for p in ROOT.glob(glob) if p.is_file())
    return sorted(found)


def test_the_operator_layer_is_importable_and_off_the_surface():
    """Demoted means both halves: still there, and not on the first page a person reads."""
    for name in OPERATOR_LAYER + OPERATOR_LAYER_UNSCANNABLE:
        assert getattr(memrank, name, None) is not None, f"memrank.{name} stopped importing"
        assert name not in SURFACE, f"{name} is the operator layer; it cannot be on the surface"


@pytest.mark.parametrize("path", _entry_path_files(),
                         ids=lambda p: p.relative_to(ROOT).as_posix())
def test_the_entry_path_never_makes_a_reader_meet_the_operator_layer(path: Path):
    """A first result must be reachable without learning the vocabulary underneath it."""
    text = path.read_text(encoding="utf-8")
    met = [name for name in OPERATOR_LAYER
           if re.search(rf"(?<![\w.]){re.escape(name)}\b", text)]
    assert not met, (f"{path.relative_to(ROOT)} teaches {', '.join(met)} -- the entry path "
                     "reaches a named evaluation with memrank.evaluation(ref); the operator "
                     "layer is taught in docs/misc/")


def test_the_entry_path_scan_reaches_the_front_page_and_the_reference():
    """Guards the guard: a moved file would empty this scan and pass every case above."""
    scanned = {p.relative_to(ROOT).as_posix() for p in _entry_path_files()}
    assert "README.md" in scanned
    assert "docs/install.md" in scanned
    assert any(p.startswith("docs/reference/") for p in scanned), "the reference folder is layer 1"
