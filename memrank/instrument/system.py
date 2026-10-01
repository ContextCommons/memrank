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
"""System -- the thing under test, and the kinds a person subclasses.

The kind IS the base class, so a kind cannot be declared wrong. Each kind's required verbs are
abstract: a class that does not supply one cannot be instantiated, and a run refuses before
touching anything when the verb is missing all the same, so the reason is stated rather than
inferred from a TypeError.

The optional declarations -- version, token usage, internal timing, stored state -- are plain
methods returning ``None``. Declaring nothing is recorded as nothing, never as zero.

`Memory` is not defined here but in `memrank.instrument.kinds`, because it needs the value
types in `memrank.contract` and this module is the one `memrank.core` imports. `memrank.core`
re-exports it, and keeps `MemoryAdapter` and `MemoryEngine` as deprecated aliases of that same
class object; the CLI keeps its own nouns.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # `memrank.core` imports this module, so the dependency only points one way.
    from memrank.core import Document


class System(ABC):  # noqa: B024 - the base names no verb; each KIND does, and that is the point
    """The thing being measured. How it is built, versioned and addressed is its own business.

    memrank records what it can establish: the class it observed, the kind that class declares
    by subclassing, and whatever the system will tell it when asked.
    """

    #: Where the system is reached, when it is reached over one. `None` means in-process.
    address: str | None = None

    def declared_version(self) -> str | None:
        """What this system says its version is. Optional; `None` is "did not state"."""
        return None

    def declared_tokens(self) -> dict[str, float | None] | None:
        """What a provider billed it. Optional -- only the system is told this."""
        return None

    def declared_timings(self) -> dict[str, list[float]] | None:
        """Millisecond samples of time only the system can see. Never the wall clock."""
        return None

    def declared_state(self) -> str | None:
        """A fingerprint of what the system is holding, when it can say. Optional."""
        return None


class Model(System):
    """Complete a prompt."""

    @abstractmethod
    def complete(self, prompt: str) -> str:
        """Return the completion of ``prompt``."""


class Retriever(System):
    """Rank over a frozen corpus. Nothing is told to it; the corpus is its own."""

    @abstractmethod
    def rank(self, query: str, k: int) -> Sequence[Document]:
        """Return the top-``k`` documents of the corpus this retriever already holds."""


class Assistant(System):
    """Respond to messages."""

    @abstractmethod
    def respond(self, messages: Sequence[dict[str, str]]) -> str:
        """Return the reply to a message list in the usual ``role``/``content`` shape."""


#: The verbs each kind requires, by the kind's class name. Read by the run's refusal check, so
#: a missing verb is a stated reason rather than an instantiation error a caller has to read.
REQUIRED_VERBS: dict[str, tuple[str, ...]] = {
    "Memory": ("prepare", "ingest", "retrieve", "cleanup"),
    "Model": ("complete",),
    "Retriever": ("rank",),
    "Assistant": ("respond",),
}

#: The kind names a person reads and writes, mapped to the class that defines the kind.
KIND_NAMES: dict[str, str] = {
    "Memory": "memory",
    "Model": "model",
    "Retriever": "retriever",
    "Assistant": "assistant",
}


def kind_of(system: System) -> str | None:
    """The kind this system declares by what it subclasses, or `None` when it declares none."""
    for base in type(system).__mro__:
        if base.__name__ in KIND_NAMES:
            return KIND_NAMES[base.__name__]
    return None


def missing_verbs(system: System) -> tuple[str, ...]:
    """The required verbs of this system's kind that it does not actually supply."""
    for base in type(system).__mro__:
        if base.__name__ in REQUIRED_VERBS:
            return tuple(
                verb for verb in REQUIRED_VERBS[base.__name__]
                if getattr(type(system), verb, None) is None
                or getattr(getattr(type(system), verb), "__isabstractmethod__", False)
            )
    return ()
