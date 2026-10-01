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
"""What the lexical memories share, so the two of them differ only in how they score.

`TFIDF` and `BM25` are the same retriever twice over -- one in-process store per isolation unit,
the same tokenisation of the question and of every stored document, the same rank-then-cut --
and the only thing that tells them apart is the weight each puts on a term. That split lives
here: the plumbing and the tokeniser are written once, and each module under
`memrank/adapters/` supplies one method, :meth:`_LexicalMemory._score`.

Sharing the tokeniser is not tidiness. A comparison between TF-IDF and BM25 whose two arms split
words differently measures the tokenisers as much as the weighting, and neither published method
specifies one -- `rank_bm25` does no preprocessing at all and says so. Fixing it in one place is
what makes the two rows on a result comparable to each other.
"""

from __future__ import annotations

import re
from abc import abstractmethod
from collections import Counter
from datetime import datetime

from memrank.core import Document, Memory, Recall
from memrank.instrumentation import LatencyCollector, TokenCollector

#: A token is a run of ASCII letters and digits. Everything else -- whitespace, punctuation,
#: hyphens, apostrophes -- separates. Applied after lowercasing, so the pattern needs no
#: upper-case range and a reader can see the whole rule in one line.
_TOKEN = re.compile(r"[a-z0-9]+")


def tokenise(text: str) -> list[str]:
    """Lowercase, then split on anything that is not a letter or a digit.

    Returns a LIST and not a set: both methods weight a term by how often it occurs, so dropping
    repeats here would silently turn each of them into the unweighted overlap they exist to
    improve on.
    """
    return _TOKEN.findall(text.lower())


def _ranked(documents: list[Document], scores: list[float],
            k: int) -> list[tuple[Document, float]]:
    """The top ``k`` documents that scored above zero, ties broken by ingest order.

    Never padded: the list length is how many documents the method found relevant, and a run
    cannot tell a padded result from a found one. A document scoring exactly zero shares no
    weighted term with the question, which is a finding rather than a weak match.
    """
    ordered = sorted(range(len(documents)), key=lambda i: (-scores[i], i))
    return [(documents[i], scores[i]) for i in ordered if scores[i] > 0.0][:k]


class _LexicalMemory(Memory):
    """Per-isolation in-process storage and the rank-then-cut, for one scoring rule to fill in."""

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
            fallback = self._store.get(self._isolation or "", [])
            documents = self._store.get(user_id, fallback)
            corpus = [Counter(tokenise(d.content)) for d in documents]
            scores = self._score(Counter(tokenise(query)), corpus)
            ranked = _ranked(documents, scores, k)
        return Recall(documents=[document for document, _ in ranked],
                      declared={"results": [document.id for document, _ in ranked],
                                "scores": [round(score, 6) for _, score in ranked]})

    def cleanup(self) -> None:
        # Destructive: drop this isolation's stored docs (matches cleanup_is_destructive).
        if self._isolation is not None:
            self._store.pop(self._isolation, None)
        self._isolation = None

    def latency_metrics(self) -> dict[str, float]:
        return self.latency.as_metrics()

    def token_metrics(self) -> dict[str, float | None]:
        return self.tokens.as_metrics()

    @abstractmethod
    def _score(self, query: Counter[str], corpus: list[Counter[str]]) -> list[float]:
        """One score per document of ``corpus``, in the order the documents were ingested.

        ``query`` and each entry of ``corpus`` are term counts from :func:`tokenise`. The
        statistics a method needs -- how many documents there are, how many contain a term, how
        long each one is -- are all derivable from ``corpus``, which is why the whole corpus is
        passed rather than one document at a time.
        """


def document_frequency(corpus: list[Counter[str]]) -> Counter[str]:
    """How many documents of ``corpus`` contain each term at least once."""
    frequency: Counter[str] = Counter()
    for document in corpus:
        frequency.update(document.keys())
    return frequency


__all__ = ["document_frequency", "tokenise"]
