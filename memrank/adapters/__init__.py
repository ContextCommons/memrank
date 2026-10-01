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
"""Deprecated: memory adapters and `MemoryAdapter`. Evaluate an agent with `memrank run` instead;
see the README. `errors` is used by the current path (`connect/http`, `api_client`) and does
not warn.

Adapter registry.

Adapter classes register themselves here by short name. The CLI looks up by
this name; the conformance suite iterates over the registry to assert every
registered adapter satisfies the contract.
"""

from __future__ import annotations

from memrank.adapters.atomicmemory import AtomicMemory
from memrank.adapters.bm25 import BM25
from memrank.adapters.controls import (
    FixedContext,
    FullContext,
    NoContext,
)
from memrank.adapters.hindsight import Hindsight
from memrank.adapters.mem0 import Mem0
from memrank.adapters.native import Native
from memrank.adapters.supermemory import Supermemory
from memrank.adapters.tfidf import TFIDF
from memrank.adapters.word_overlap import WordOverlap
from memrank.core import Memory

REGISTRY: dict[str, type[Memory]] = {
    "atomicmemory": AtomicMemory,
    "word-overlap": WordOverlap,
    # The two named lexical methods, in plain Python: a reader recognises what is being run,
    # and neither costs a dependency (ATO-2259).
    "tfidf": TFIDF,
    "bm25": BM25,
    "mem0": Mem0,
    "hindsight": Hindsight,
    "supermemory": Supermemory,
    # Not an engine: the client for memrank's own translator contract
    # (docs/system-contract.md), which is how an engine memrank has never seen gets evaluated
    # without a fork. Every other entry here names one vendor; this one names none.
    "native": Native,
    # The mandatory baseline arms (PRD_Memrank_v2.md:86) -- not memory systems.
    "no-context": NoContext,
    "fixed-context": FixedContext,
    "full-context": FullContext,
}


def get_adapter(name: str, **kwargs) -> Memory:
    """Construct the adapter registered under ``name``.

    Raises ``ValueError`` with a helpful message when the name is unknown.
    """
    if name not in REGISTRY:
        available = ", ".join(sorted(REGISTRY))
        raise ValueError(f"Unknown adapter: {name!r}. Available: {available}")
    return REGISTRY[name](**kwargs)


def list_adapters() -> list[str]:
    """Return the registered adapter names in deterministic order."""
    return sorted(REGISTRY.keys())


#: The suffixed spellings these classes used to carry, re-exported from the import path callers
#: already name. Each is the same class object as the name above it, so `REGISTRY` and every
#: `isinstance` are indifferent to which spelling reached them. Removing these is plan step 22
#: (ATO-2151).
AtomicMemoryAdapter = AtomicMemory
HindsightAdapter = Hindsight
Mem0Adapter = Mem0
NativeAdapter = Native
SupermemoryAdapter = Supermemory
WordOverlapAdapter = WordOverlap
FixedContextAdapter = FixedContext
FullContextAdapter = FullContext
NoContextAdapter = NoContext


__all__ = [
    "BM25",
    "REGISTRY",
    "TFIDF",
    "AtomicMemory",
    "FixedContext",
    "FullContext",
    "Hindsight",
    "Mem0",
    "Native",
    "NoContext",
    "Supermemory",
    "WordOverlap",
    "get_adapter",
    "list_adapters",
    # Deprecated, and listed so `from memrank.adapters import *` keeps serving them.
    "AtomicMemoryAdapter",
    "FixedContextAdapter",
    "FullContextAdapter",
    "HindsightAdapter",
    "Mem0Adapter",
    "NativeAdapter",
    "NoContextAdapter",
    "SupermemoryAdapter",
    "WordOverlapAdapter",
]
