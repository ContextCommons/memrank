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
"""In-process word-overlap retrieval -- the "dumb memory" floor.

Ranks documents by how many words they share with the question. Gives the reader a floor: a real
engine must beat dumb keyword matching to be interesting. Runs in-process (labeled as such);
excluded from HTTP latency ranking.

Named for the mechanism at the level the mechanism lives. It was `baseline` until 2026-08-19 -- a
ROLE every control arm occupies, carrying no information -- and `lexical` until 2026-08-20, which
was BEIR's FAMILY label where an instance-level name existed. Not `bm25`, which would be false: the
score below is an unweighted set intersection, no IDF, no term frequency, no length normalisation.
"""

from __future__ import annotations

from datetime import datetime

from memrank.core import Document, Memory, Recall
from memrank.instrumentation import LatencyCollector, TokenCollector


def _tokens(text: str) -> set[str]:
    return {t for t in text.lower().split() if t}


class WordOverlap(Memory):
    """Token-overlap retrieval over an in-process per-isolation store."""

    name = "word-overlap"
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

    def retrieve(self, query: str, k: int, user_id: str,
                 query_timestamp: datetime | str | None = None) -> Recall:
        with self.latency.track("retrieve"):
            q = _tokens(query)
            fallback = self._store.get(self._isolation or "", [])
            scored = [
                (len(q & _tokens(d.content)), d)
                for d in self._store.get(user_id, fallback)
            ]
            ranked = [d for score, d in sorted(scored, key=lambda x: x[0], reverse=True)
                      if score > 0][:k]
        return Recall(documents=ranked, declared={"results": [d.id for d in ranked]})

    def cleanup(self) -> None:
        # Destructive: drop this isolation's stored docs (matches cleanup_is_destructive).
        if self._isolation is not None:
            self._store.pop(self._isolation, None)
        self._isolation = None

    def latency_metrics(self) -> dict[str, float]:
        return self.latency.as_metrics()

    def token_metrics(self) -> dict[str, float | None]:
        return self.tokens.as_metrics()


#: Deprecated alias of the class above -- the same class object, so an out-of-tree import and
#: every ``isinstance`` against the older spelling keep holding. The suffix went because a reader
#: copies the class name out of a first result, and ``Adapter`` is memrank's word for the wrapper
#: rather than the reader's word for the system. Removing it is plan step 22 (ATO-2151).
WordOverlapAdapter = WordOverlap
