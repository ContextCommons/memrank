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
"""Your system against an engine you did not write, on identical tasks.

    uv run python examples/07-against-a-known-engine/run.py

The engine here is `WordOverlap`, a system memrank ships, because it runs
offline and in a second. Point it at a real one by swapping that one name --
`memrank.catalog()` prints the others and the README beside this file says
what each needs. Nothing else changes, which is the point: only the system
differs.
"""

import memrank
from memrank import Document, Recall
from memrank.evaluations import Demo
from memrank.systems import WordOverlap


class TinyMemory(memrank.Memory):
    """Your system: it ranks passages by the words they share with a query."""

    def __init__(self) -> None:
        self.notes: list[Document] = []

    def prepare(self, isolation_unit: str) -> None:
        """Starts a fresh store for one group of tasks that share state."""
        self.notes = []

    def ingest(self, documents: list[Document]) -> None:
        """Takes the documents this group is told."""
        self.notes.extend(documents)

    def retrieve(self, query: str, k: int, user_id: str,
                 query_timestamp=None) -> Recall:
        """Returns at most k documents, best first."""
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


# One evaluation behind both runs, so the tasks and the measures are equal.
evaluation = Demo()
mine = TinyMemory()
known_engine = WordOverlap()
result_mine = evaluation.run(system=mine)
result_known_engine = evaluation.run(system=known_engine)
reading = memrank.paired(result_mine, result_known_engine)

print(reading)
