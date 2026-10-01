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
"""Memrank -- Evaluate your agent on your tasks.

You supply the system. Memrank puts the evaluation's tasks to it, records what happened,
and applies the measures that turn those records into named values. (A memory is one kind
of system it evaluates; `System` below names the others.)

    import memrank
    from memrank.evaluations import Demo

    class MyMemory(memrank.Memory):
        ...                                        # the four verbs its kind requires

    result = Demo().run(system=MyMemory())
    result.values                                  # named values, each with its decider
    result.traces                                  # one per task per attempt, always

Seven things, and nothing else has to be learned:

- **system** -- the thing under test. Subclass the kind it is: ``Memory`` (tell it things, ask
  it for what is relevant, clear on request), ``Model``, ``Retriever`` or ``Assistant``. The
  required verbs are abstract; the optional declarations return ``None`` until you say.
  ``memrank.systems.TFIDF()`` is one memrank ships, and
  ``memrank.system("tfidf")`` is the same class by its catalog name.
- **evaluation** -- a named, versioned bundle: its tasks, the measures it ships with, and the
  rule for when state is cleared. ``memrank.evaluations.Demo()`` builds one from an in-tree
  benchmark, ``memrank.evaluation("demo")`` does the same by name, and
  ``memrank.Evaluation(...)`` is the same object, written by hand.
- **task** -- one thing to put to the system: the context, the prompt, what is expected.
- **trace** -- everything observed while one task ran. Exactly one per task per attempt,
  including on failure.
- **measure** -- a named rule from traces to values, declaring what it reads and who decides.
  ``WordMatch``, ``Judge``, ``Latency``, ``FailureRate``.
- **run** -- the act, and a verb on the evaluation: ``evaluation.run(system=...)``. It refuses
  BEFORE touching the system when the run cannot be set up, and returns a result with no traces
  and a stated reason. The evaluation owns the verb so that the objects say which is which.
- **result** -- the traces and the values, each value carrying its measure's name and its
  decider. Never a bare number, and never a verdict.

Two things sit above the seven: ``memrank.measure(result, MyMeasure())`` applies a measure
thought of later to stored traces, and ``memrank.paired(a, b)`` reads two results of the same
evaluation side by side. Neither says "better".

What memrank ships is reachable without knowing a single string. ``memrank.catalog()`` prints
it all; ``memrank.systems`` holds the shipped systems as classes and ``memrank.evaluations``
the shipped evaluations as classes too, so "go to definition" lands on the real thing where a
string literal led nowhere. ``memrank.system(name)`` and ``memrank.evaluation(name)`` remain
for choosing by string, which is what a config file and the command line have.

The previous Python surface has not been taken away, only moved off the front page:
``memrank.evaluation.api.run`` still produces the ``EvalResult`` the command line and the
cloud read, and ``MemoryAdapter``, ``Benchmark``, ``EvalResult``, ``SpanRecall``,
``ComposedEvaluation`` and the sub-nouns are still importable from here and from their own
modules.
"""

# Every name below is a plain import, so "go to definition" lands on the class or the function
# and a type checker sees the real signature. They were resolved through a module-level
# `__getattr__` until this commit, which is invisible to every editor: command-click on
# `memrank.measure` went nowhere and hover showed nothing.
#
# The nine spelled `X as X` are not in `__all__`, and the redundant alias is what says
# "deliberately re-exported" to the linter and to a reader -- these are the operator layer and
# the sub-nouns, not names this module forgot to use. `tests/repo/test_python_vocabulary.py`
# holds which is which: deprecated (`MemoryAdapter`, `EvalResult`) and merely demoted
# (`Benchmark`, which keeps its name and is reached through `memrank.evaluation(ref)`).
import sys as _sys
from types import ModuleType as _ModuleType
from typing import Any

from memrank import evaluations, systems
from memrank.contract import Document, Recall
from memrank.core import AdapterResponse as AdapterResponse
from memrank.core import Benchmark as Benchmark
from memrank.core import BenchmarkUnit as BenchmarkUnit
from memrank.core import EvalInfo as EvalInfo
from memrank.core import MemoryAdapter as MemoryAdapter
from memrank.core import MemoryEngine as MemoryEngine
from memrank.instrument.catalog import catalog, evaluation, system
from memrank.instrument.evaluation import Clearing, Evaluation
from memrank.instrument.kinds import Memory
from memrank.instrument.measure import Decider, Measure, Scope, Value
from memrank.instrument.measures import BenchmarkScore as BenchmarkScore
from memrank.instrument.measures import FailureRate, Judge, Latency, WordMatch
from memrank.instrument.paired import Paired, paired
from memrank.instrument.paired import PairingRefused as PairingRefused
from memrank.instrument.result import Result
from memrank.instrument.run import Answerer as Answerer
from memrank.instrument.run import measure
from memrank.instrument.system import Assistant, Model, Retriever, System
from memrank.instrument.task import Expected, Task
from memrank.instrument.trace import Trace


def _installed_version() -> str:
    """The version of the installed distribution, so `memrank version` cannot disagree with it.

    A hardcoded constant beside pyproject's is two sources of truth, and the one that goes stale
    is the one users quote in bug reports -- this printed 0.1.0 from a 0.2.0 install. Falls back
    to the literal only when the package is not installed at all (a bare source checkout on
    sys.path), which is the one case metadata genuinely cannot answer.
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("memrank")
    except PackageNotFoundError:
        return "0.0.0+unknown"


__version__ = _installed_version()

#: The previous surface, and the only names still resolved lazily (PEP 562). Not because they
#: are secondary -- because importing them costs what the seven do not: the cell run reaches
#: `memrank.metrics.cost` and so `tiktoken`, and the judge reaches `anthropic`. Measured at the
#: time of writing, `import memrank` is ~20ms with these lazy and ~340ms with them static, and
#: none of it is on the path of a caller who asked for `evaluation.run`.
#:
#: They are off the front page rather than deprecated-and-hidden: each is reachable by its own
#: full module path, which is where "go to definition" finds it. `tests/repo/test_import_weight.py`
#: holds the boundary.
_PREVIOUS_SURFACE = {
    "benchmark": ("memrank.benchmarks", "from_ref"),
    "EvalResult": ("memrank.evaluation.result", "EvalResult"),
    "EvalObserver": ("memrank.evaluation.observer", "EvalObserver"),
    "EvalPlan": ("memrank.evaluation.observer", "EvalPlan"),
    "JudgeConfig": ("memrank.judging.judge", "JudgeConfig"),
    "SpanRecall": ("memrank.metrics.scoring", "SpanRecall"),
    "ComposedEvaluation": ("memrank.composition", "ComposedEvaluation"),
    "Scorer": ("memrank.composition", "Scorer"),
    "CriteriaMismatch": ("memrank.composition", "CriteriaMismatch"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _PREVIOUS_SURFACE[name]
    except KeyError:
        raise AttributeError(f"module 'memrank' has no attribute {name!r}") from None
    from importlib import import_module

    return getattr(import_module(module_name), attribute)


#: `memrank.evaluation` is BOTH the constructor of the seven and the name of the package the
#: cell run lives in, and an imported submodule wins: importing `memrank.evaluation.api` makes
#: the import machinery rebind `evaluation` in this module's `__dict__` to the package. A plain
#: import does not make that go away -- it is a write over the name the import above bound, and
#: it was measured here before this line was restored: `memrank.evaluation("demo")` raised
#: `'module' object is not callable`, and only once the import order happened to put the
#: package first. So the shadowing guard stays, and it holds the function in a private name the
#: import machinery has no reason to touch.
#:
#: It costs nothing an editor can see. The plain import above is what "go to definition", hover
#: and autocomplete read, and this only decides which object wins at RUNTIME. Both halves hold:
#: `memrank.evaluation` is the function, and the package is still importable by its full name,
#: `from memrank.evaluation.api import run`.
_evaluation = evaluation


class _Front(_ModuleType):
    """This package's own type, so a submodule cannot shadow one of the seven's verbs."""

    def __getattribute__(self, name: str) -> Any:
        if name == "evaluation":
            return super().__getattribute__("_evaluation")
        return super().__getattribute__(name)


_sys.modules[__name__].__class__ = _Front


def __dir__() -> list[str]:
    return sorted([*globals(), *_PREVIOUS_SURFACE])


#: The seven, the names a person subclasses to declare a kind, the measures memrank ships,
#: the two readings above the run, and the three value types a self-contained example cannot
#: avoid writing -- `Document` and `Expected` to state a task, `Recall` to return one.
#: Everything else this module exposes is the previous surface -- still importable, and none of
#: it something a first number requires. Held to this by `tests/repo/test_python_vocabulary.py`.
__all__ = [
    "Assistant",
    "Clearing",
    "Decider",
    "Document",
    "Evaluation",
    "Expected",
    "FailureRate",
    "Judge",
    "Latency",
    "Measure",
    "Memory",
    "Model",
    "Paired",
    "Recall",
    "Result",
    "Retriever",
    "Scope",
    "System",
    "Task",
    "Trace",
    "Value",
    "WordMatch",
    "__version__",
    "catalog",
    "evaluation",
    "evaluations",
    "measure",
    "paired",
    "system",
    "systems",
]
