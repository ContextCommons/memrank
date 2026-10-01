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
"""Okapi BM25 in plain Python -- the lexical baseline the retrieval literature ranks against.

Robertson and Zaragoza (2009) is the account of it. BM25 keeps TF-IDF's two ideas and fixes what
each one gets wrong at the edges: a term occurring ten times does not make a document ten times
more relevant, so term frequency SATURATES; and a long document has more chances to contain a
term without being more about it, so the saturation point scales with document length.

The exact rule, over the ``N`` documents stored for the isolation unit being queried::

    score(d, q) = sum over t in q of
        idf(t) * (f(t, d) * (k1 + 1)) / (f(t, d) + k1 * (1 - b + b * |d| / avgdl))

    idf(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))

``f(t, d)`` is the raw count of ``t`` in ``d``, ``|d|`` its length in tokens and ``avgdl`` the
mean length across the corpus. The ``1 +`` inside the logarithm is the standard smoothing: the
unsmoothed probabilistic idf goes NEGATIVE for a term in more than half the documents, which on
a corpus of three makes a common word actively penalise the documents holding it.

`k1` governs how fast term frequency saturates and `b` how strongly length is normalised; 1.5
and 0.75 are the conventional defaults and are constructor arguments, so a run that moves them
says so rather than being a different method under the same name.

No dependency, no index, no model. `TFIDF` is the same corpus under the older weighting, sharing
`memrank.adapters.lexical`'s tokeniser so the scoring is the only difference between them.
"""

from __future__ import annotations

import math
from collections import Counter

from memrank.adapters.lexical import _LexicalMemory, document_frequency

#: Term-frequency saturation. Robertson and Zaragoza report 1.2-2.0 as the usual range.
K1_DEFAULT = 1.5
#: Document-length normalisation: 0 ignores length entirely, 1 normalises fully.
B_DEFAULT = 0.75


class BM25(_LexicalMemory):
    """Okapi BM25 over an in-process per-isolation store.

    Args:
        k1: Term-frequency saturation. Higher means a repeated term keeps counting for longer.
        b: Document-length normalisation, between 0 and 1.
    """

    name = "bm25"

    def __init__(self, k1: float = K1_DEFAULT, b: float = B_DEFAULT) -> None:
        super().__init__()
        self.k1 = k1
        self.b = b

    def _score(self, query: Counter[str], corpus: list[Counter[str]]) -> list[float]:
        total = len(corpus)
        if total == 0:
            return []
        frequency = document_frequency(corpus)
        lengths = [sum(document.values()) for document in corpus]
        average_length = sum(lengths) / total
        if average_length == 0.0:
            # Every stored document is empty, so nothing can match and the length term is
            # undefined. Saying so here keeps the division below unconditional.
            return [0.0] * total
        idf = {term: math.log(1 + (total - count + 0.5) / (count + 0.5))
               for term, count in frequency.items()}
        return [self._document_score(query, document, length, average_length, idf)
                for document, length in zip(corpus, lengths, strict=True)]

    def _document_score(self, query: Counter[str], document: Counter[str], length: int,
                        average_length: float, idf: dict[str, float]) -> float:
        """One document's BM25 score: the saturated, length-normalised weight of every query term.

        The length term depends on the document and not on the term, so it is computed once
        rather than once per query word.
        """
        saturation = self.k1 * (1 - self.b + self.b * length / average_length)
        total = 0.0
        for term in query:
            occurrences = document.get(term, 0)
            if occurrences == 0 or term not in idf:
                continue
            total += idf[term] * occurrences * (self.k1 + 1) / (occurrences + saturation)
        return total


__all__ = ["B_DEFAULT", "BM25", "K1_DEFAULT"]
