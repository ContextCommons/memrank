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
"""Deprecated: the in-process Python route (`evaluation.run(system=...)`) and its seven nouns.
Evaluate an agent with `memrank run` instead; see the README. `kinds` (via `memrank.core`) and
`statistics` (via `service/scoring`) are used by the current path and do not warn.

The seven: system, evaluation, task, trace, measure, run, result.

One module per noun, and the names here are the words the conceptual model uses -- the
nouns are the vocabulary, so the code spells them the way the documentation does.

Nothing is imported eagerly. `memrank.core` imports `memrank.instrument.system`, so a package
body that reached back into `memrank.core` would close the loop -- and `import memrank` stays
as light as it was, with the loop loading only when a run is asked for.

`memrank/instrument/` rather than `memrank/model/`: "model" is already a kind of system here,
and in the field at large it means the thing being run, not the frame it is run in. memrank
calls itself an instrument, and these are the parts of the instrument.
"""

from __future__ import annotations

from typing import Any

#: The nouns, by name. The VERBS -- `run`, `measure`, `evaluation`, `paired` -- are not here:
#: each shares its name with the module it lives in, and an imported submodule shadows a
#: package attribute, so `memrank.instrument.run` is the module and always will be. A person
#: reaches the verbs at `memrank.run`, `memrank.measure`, `memrank.evaluation`,
#: `memrank.paired`, which resolve straight to the leaf modules.
_LAZY = {
    "System": ("memrank.instrument.system", "System"),
    "Model": ("memrank.instrument.system", "Model"),
    "Retriever": ("memrank.instrument.system", "Retriever"),
    "Assistant": ("memrank.instrument.system", "Assistant"),
    "Memory": ("memrank.instrument.kinds", "Memory"),
    "Evaluation": ("memrank.instrument.evaluation", "Evaluation"),
    "Clearing": ("memrank.instrument.evaluation", "Clearing"),
    "Task": ("memrank.instrument.task", "Task"),
    "Expected": ("memrank.instrument.task", "Expected"),
    "Trace": ("memrank.instrument.trace", "Trace"),
    #: The recalled half of a trace is `memrank.contract.Recall`, the type retrieve
    #: returns; it is re-exported from `memrank.core` and from `memrank`.
    "Measure": ("memrank.instrument.measure", "Measure"),
    "Value": ("memrank.instrument.measure", "Value"),
    "Scope": ("memrank.instrument.measure", "Scope"),
    "Decider": ("memrank.instrument.measure", "Decider"),
    "Answerer": ("memrank.instrument.run", "Answerer"),
    "Result": ("memrank.instrument.result", "Result"),
    "Paired": ("memrank.instrument.paired", "Paired"),
    "PairingRefused": ("memrank.instrument.paired", "PairingRefused"),
    "WordMatch": ("memrank.instrument.measures", "WordMatch"),
    "Judge": ("memrank.instrument.measures", "Judge"),
    "Latency": ("memrank.instrument.measures", "Latency"),
    "FailureRate": ("memrank.instrument.measures", "FailureRate"),
    "BenchmarkScore": ("memrank.instrument.measures", "BenchmarkScore"),
}

__all__ = sorted(_LAZY)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module 'memrank.instrument' has no attribute {name!r}") from None
    from importlib import import_module

    return getattr(import_module(module_name), attribute)


def __dir__() -> list[str]:
    return sorted([*globals(), *_LAZY])
