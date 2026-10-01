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
"""The mandatory baseline arms -- the hypothesis test, run on every row.

``docs/PRD_Memrank_v2.md`` requires three controls beside every engine: (a) no-memory,
(b) a **token-matched** in-context baseline at the same retrieval budget, and (c) full-context where
the knowledge base fits. Together they answer the project's central hypothesis: external memory
"make[s] the decisive difference when the knowledge base is too large to fit in the model's context
window -- and [is] worth [its] tokens against a naive baseline that just reads what it can."

None of these retrieve in any meaningful sense, which is the point. They are deliberately *not*
memory systems: ``no-context`` supplies nothing, and the in-context arms supply the corpus unranked, so
whatever the engine under test wins by is attributable to retrieval rather than to context size.
"""

from __future__ import annotations

from datetime import datetime

from memrank.core import Document, Memory, Recall
from memrank.instrumentation import LatencyCollector, TokenCollector


class _InProcessControl(Memory):
    """Shared plumbing: per-isolation storage, no network, no engine."""

    version = "0.1.0"
    engine_version = "in-process"
    transport = "in-process"
    cleanup_is_destructive = True

    def __init__(self) -> None:
        self._store: dict[str, list[Document]] = {}
        self._isolation: str | None = None
        self.latency = LatencyCollector()
        self.tokens = TokenCollector()

    def prepare(self, isolation_unit: str) -> None:
        self._isolation = isolation_unit
        self._store.setdefault(isolation_unit, [])

    def ingest(self, documents: list[Document]) -> None:
        assert self._isolation is not None, "call prepare() before ingest()"
        with self.latency.track("ingest"):
            self._store[self._isolation].extend(documents)

    def cleanup(self) -> None:
        if self._isolation is not None:
            self._store.pop(self._isolation, None)
        self._isolation = None

    def latency_metrics(self) -> dict[str, float]:
        return self.latency.as_metrics()

    def token_metrics(self) -> dict[str, float | None]:
        return self.tokens.as_metrics()


class NoContext(_InProcessControl):
    """(a) no-memory: retrieves nothing, so the reader answers closed-book.

    The floor the hypothesis is measured against. An engine that does not beat this is not earning
    its tokens. Equivalent to the ``no_context_control`` that ``judge.judge_query`` already runs per
    query, promoted to a selectable arm so it produces a score, a row, and a receipt.
    """

    name = "no-context"
    context_budget = "none"

    def retrieve(self, query: str, k: int, user_id: str,
                 query_timestamp: datetime | str | None = None) -> Recall:
        with self.latency.track("retrieve"):
            pass
        return Recall(documents=[], declared={"results": []})


class FixedContext(_InProcessControl):
    """(b) token-matched ICL: hands over the corpus unranked, capped by the shared token budget.

    "A naive baseline that just reads what it can" -- it performs no selection at all, so the
    ``--token-budget`` cap decides how much of the corpus the reader sees. Being token-MATCHED with
    the engine under test is what makes the comparison fair: same context size, and the only variable
    is whether ranked retrieval beats reading from the start.

    ``k`` is ignored on purpose. It bounds ranked retrieval; here the token budget is the bound.
    """

    # Returns the corpus in INGEST order, not a ranked list, so rank-cut retrieval
    # metrics do not apply -- see Memory.ranks_results. Inherited by full-context.
    ranks_results = False

    name = "fixed-context"

    def retrieve(self, query: str, k: int, user_id: str,
                 query_timestamp: datetime | str | None = None) -> Recall:
        with self.latency.track("retrieve"):
            docs = list(self._store.get(user_id) or self._store.get(self._isolation or "") or [])
        return Recall(documents=docs, declared={"results": [d.id for d in docs]})


class FullContext(FixedContext):
    """(c) full-context: the corpus, uncapped -- the ceiling where the KB fits.

    Deliberately rare. The PRD demotes any benchmark whose knowledge base fits inside a current
    context window (LoCoMo, at 16k-26k tokens, is named), so this arm shows the ceiling on small
    corpora rather than appearing on every row. Its manifest declares an uncapped budget; the run
    records the tokens actually sent and whether truncation still occurred.
    """

    name = "full-context"
    context_budget = "uncapped"


#: Deprecated aliases of the three classes above -- the same class objects, so an out-of-tree
#: import and every ``isinstance`` against the older spellings keep holding. The suffix went
#: because a reader copies the class name out of a first result, and ``Adapter`` is memrank's word
#: for the wrapper rather than the reader's word for the system. Removing them is plan step 22
#: (ATO-2151).
NoContextAdapter = NoContext
FixedContextAdapter = FixedContext
FullContextAdapter = FullContext
