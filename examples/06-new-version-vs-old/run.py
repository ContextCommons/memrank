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
"""Did the change help? Two versions of your own system, read side by side.

    uv run python examples/06-new-version-vs-old/run.py

Read three things. The GAP is the difference in means. The FLIPS are the
tasks whose value changed, and which way -- a version that fixes three and
breaks two has a gap near zero and is not the same system. The CAUTION
appears when there is too little here to characterise either.
"""

import memrank
from memrank import Document, Evaluation, Expected, Recall, Task, WordMatch

# The words every question shares, so matching on them says nothing. Dropping
# them is the whole of version 2 below.
COMMON_WORDS = frozenset({
    "a", "an", "the", "is", "was", "of", "to", "in", "on", "for",
    "what", "who", "when", "where", "did", "does", "do", "and",
})

# The last two notes answer nothing and share the words every question has.
NOTES = (
    Document(id="t1", user_id="acme",
             content="Acme moved to the enterprise plan in March."),
    Document(id="t2", user_id="acme",
             content="Acme's outage was traced to an expired webhook secret."),
    Document(id="t3", user_id="acme",
             content="What is the plan is what we do on the board."),
    Document(id="t4", user_id="acme",
             content="The plan is what the team does on Mondays."),
)

PLAN_TASK = Task(
    id="q_plan", prompt="What plan is Acme on?", group="acme", context=NOTES,
    expected=Expected(required_spans=("enterprise",),
                      evidence_doc_ids=("t1",)),
)

OUTAGE_TASK = Task(
    id="q_outage", prompt="What caused Acme's outage?", group="acme",
    context=NOTES,
    expected=Expected(required_spans=("webhook secret",),
                      evidence_doc_ids=("t2",)),
)

REFUND_TASK = Task(
    id="q_refund", prompt="Was Acme promised a refund?", group="acme",
    context=NOTES,
    expected=Expected(forbidden_spans=("refund approved",),
                      polarity="negative"),
)

TICKETS = Evaluation(
    name="tickets",
    version="internal@2026-09",
    measures=(WordMatch(),),
    tasks=(PLAN_TASK, OUTAGE_TASK, REFUND_TASK),
)


class TinyMemory(memrank.Memory):
    """A memory that ranks passages by the words they share with a question."""

    def __init__(self) -> None:
        self.notes: list[Document] = []

    def prepare(self, isolation_unit: str) -> None:
        """Starts a fresh store for one group of tasks that share state."""
        self.notes = []

    def ingest(self, documents: list[Document]) -> None:
        """Takes the documents this group is told."""
        self.notes.extend(documents)

    def overlap(self, query: str, document: Document) -> int:
        """Counts the words a document shares with the query."""
        # THE ONE METHOD VERSION 2 CHANGES. The rest is shared by inheritance.
        query_words = set(query.lower().split())
        return len(query_words & set(document.content.lower().split()))

    def retrieve(self, query: str, k: int, user_id: str,
                 query_timestamp=None) -> Recall:
        """Returns at most k documents, best first."""
        ranked = sorted(self.notes,
                        key=lambda note: self.overlap(query, note),
                        reverse=True)
        return Recall(
            documents=ranked[:k],
            declared={"considered": len(self.notes)},
        )

    def cleanup(self) -> None:
        """Drops everything this system was told."""
        self.notes = []


class TinyMemoryV2(TinyMemory):
    """Version 1, with the common words dropped from the overlap."""

    def overlap(self, query: str, document: Document) -> int:
        """Counts the shared words that are not words every question has."""
        query_words = set(query.lower().split())
        shared = query_words & set(document.content.lower().split())
        return len(shared - COMMON_WORDS)


# k=2 is what makes ranking matter: two of the four notes answer nothing and
# share the common words, so a memory that counts those words spends both of
# its slots on them.
version_1 = TinyMemory()
version_2 = TinyMemoryV2()
result_before = TICKETS.run(system=version_1, k=2)
result_after = TICKETS.run(system=version_2, k=2)
reading = memrank.paired(result_before, result_after)

print(reading)
