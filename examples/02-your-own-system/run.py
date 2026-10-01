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
"""Your own memory, measured -- the four verbs its kind requires.

    uv run python examples/02-your-own-system/run.py

The kind IS the base class, so a kind cannot be declared wrong: subclass
`memrank.Memory` and you are told things, asked for what is relevant, and
cleared on request. Nothing is registered, nothing is named, and no file is
written inside memrank. The evaluation is the one 01 ran.
"""

import memrank
from memrank import Document, Recall


class TinyMemory(memrank.Memory):
    """A memory that ranks passages by the words they share with a question."""

    def __init__(self) -> None:
        self.notes: list[Document] = []

    def declared_version(self) -> str:
        """Returns what only this system knows about itself."""
        # Optional. Declaring nothing is recorded as "did not state", never as
        # a zero. Latency does not belong here: memrank times it itself.
        return "1"

    def prepare(self, isolation_unit: str) -> None:
        """Starts a fresh store for one group of tasks that share state."""
        # State that leaks into the next group means every value after the
        # first was measured against a system that had already seen answers.
        self.notes = []

    def ingest(self, documents: list[Document]) -> None:
        """Takes the documents this group is told."""
        self.notes.extend(documents)

    def retrieve(self, query: str, k: int, user_id: str,
                 query_timestamp=None) -> Recall:
        """Returns at most k documents, best first."""
        # ORDER is the measurement here, so never pad the list to reach k.
        query_words = set(query.lower().split())

        def shared_word_count(note: Document) -> int:
            return len(query_words & set(note.content.lower().split()))

        ranked = sorted(self.notes, key=shared_word_count, reverse=True)
        return Recall(
            documents=ranked[:k],
            declared={"considered": len(self.notes)},
        )

    def cleanup(self) -> None:
        """Drops everything this system was told."""
        self.notes = []


system = TinyMemory()
evaluation = memrank.evaluation("demo")
result = evaluation.run(system=system)

print(result)
