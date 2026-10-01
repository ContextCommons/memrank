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
"""Deprecated: the shipped in-process systems (`TFIDF`, `BM25`, the engine clients) and the
in-process Python route they serve. Evaluate an agent with `memrank run` instead; see the
README.

The systems memrank ships, as classes -- `from memrank.systems import TFIDF`.

`memrank.system("tfidf")` needs the string first, and a string is not navigable: nothing
in an editor follows it, and nothing tells a reader what else is out there. Here the same
systems are Python names, so "go to definition" lands on the class, hover shows its docstring,
and autocomplete lists the catalog.

Each name below is the class's own name: `memrank/adapters/` defines `TFIDF`, and this
module re-exports it. The suffixed spellings those classes used to carry -- `WordOverlapAdapter`
and its eight siblings -- remain importable from `memrank.adapters` as deprecated aliases of the
same class objects, so nothing is duplicated, every registered adapter keeps working, and the
command line keeps the older spellings it prints.

============  =======  ===============================================================
name          kind     what it needs
============  =======  ===============================================================
TFIDF         memory   nothing -- in-process, offline
BM25          memory   nothing -- in-process, offline
WordOverlap   memory   nothing -- in-process, offline
NoContext     control  nothing -- the floor: retrieves nothing, answers closed-book
FixedContext  control  nothing -- the corpus unranked, capped by the token budget
FullContext   control  nothing -- the corpus uncapped: the ceiling retrieval aims at
AtomicMemory  memory   a running engine, at `base_url=` or ATOMICMEMORY_API_URL
Hindsight     memory   a running engine, at `base_url=` or HINDSIGHT_API_URL
Supermemory   memory   a running engine, at `base_url=` or SUPERMEMORY_BASE_URL
Mem0          memory   the mem0 SDK, or a running engine at MEM0_HTTP_URL
Native        memory   a running translator of memrank's system contract, NATIVE_API_URL
============  =======  ===============================================================

A control is not a memory system. It exists so a number has something to mean: a system that
does not beat `NoContext` has not earned its tokens, and `FullContext` is what retrieval is
trying to reach. `memrank.catalog()` prints this table at runtime.
"""

from __future__ import annotations

from memrank.adapters.atomicmemory import AtomicMemory
from memrank.adapters.bm25 import BM25
from memrank.adapters.controls import FixedContext, FullContext, NoContext
from memrank.adapters.hindsight import Hindsight
from memrank.adapters.mem0 import Mem0
from memrank.adapters.native import Native
from memrank.adapters.supermemory import Supermemory
from memrank.adapters.tfidf import TFIDF
from memrank.adapters.word_overlap import WordOverlap
from memrank.instrument.catalog import ShippedSystem

#: The same table the module docstring carries, in the form `memrank.catalog()` prints. One
#: entry per name in `memrank.adapters.REGISTRY`, which `tests/instrument` holds it to.
SHIPPED: tuple[ShippedSystem, ...] = (
    ShippedSystem("TFIDF", "tfidf", "memory", "nothing -- in-process, offline"),
    ShippedSystem("BM25", "bm25", "memory", "nothing -- in-process, offline"),
    ShippedSystem("WordOverlap", "word-overlap", "memory", "nothing -- in-process, offline"),
    ShippedSystem("NoContext", "no-context", "control",
                  "nothing -- the floor: retrieves nothing, answers closed-book"),
    ShippedSystem("FixedContext", "fixed-context", "control",
                  "nothing -- the corpus unranked, capped by the token budget"),
    ShippedSystem("FullContext", "full-context", "control",
                  "nothing -- the corpus uncapped: the ceiling retrieval aims at"),
    ShippedSystem("AtomicMemory", "atomicmemory", "memory",
                  "a running engine, at base_url= or ATOMICMEMORY_API_URL"),
    ShippedSystem("Hindsight", "hindsight", "memory",
                  "a running engine, at base_url= or HINDSIGHT_API_URL"),
    ShippedSystem("Supermemory", "supermemory", "memory",
                  "a running engine, at base_url= or SUPERMEMORY_BASE_URL"),
    ShippedSystem("Mem0", "mem0", "memory",
                  "the mem0 SDK, or a running engine at MEM0_HTTP_URL"),
    ShippedSystem("Native", "native", "memory",
                  "a running translator of memrank's system contract, at NATIVE_API_URL"),
)

__all__ = [
    "BM25",
    "SHIPPED",
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
]
