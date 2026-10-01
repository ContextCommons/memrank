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
"""TF-IDF retrieval in plain Python: **cosine similarity** of TF-IDF vectors.

Term frequency times inverse document frequency, the weighting Sparck Jones (1972) introduced
and Salton's vector space model ranks with. A term is worth what it is worth because of how
often it occurs in a document and how rare it is across the corpus, and a question and a
document are compared as vectors over those weights.

The exact rule, stated because every implementation differs and a number nobody can reproduce is
not a baseline:

- ``tf(t, d)`` is the RAW count of term ``t`` in document ``d``.
- ``idf(t) = ln(N / df(t))``, the textbook form, over the ``N`` documents stored for the
  isolation unit being queried. A term in every document weighs exactly nothing.
- A document's vector is ``tf(t, d) * idf(t)`` over its own terms; the question's is the same
  over the question's terms.
- The score is the **cosine** of those two vectors -- the dot product divided by both norms --
  and NOT the bare dot product. Cosine is what makes a short document that is mostly about the
  question outrank a long one that merely mentions it.

No dependency, no index, no model. `BM25` is the same corpus scored by the method that replaced
this one for ranking, and the two share `memrank.adapters.lexical`'s tokeniser exactly so the
weighting is the only thing between them.
"""

from __future__ import annotations

import math
from collections import Counter

from memrank.adapters.lexical import _LexicalMemory, document_frequency


def _weighted(counts: Counter[str], idf: dict[str, float]) -> dict[str, float]:
    """One TF-IDF vector: raw term count times the term's inverse document frequency."""
    return {term: count * idf[term] for term, count in counts.items() if term in idf}


def _cosine(query: dict[str, float], document: dict[str, float]) -> float:
    """The cosine of two sparse vectors; zero when either has no length to speak of."""
    query_norm = math.sqrt(sum(weight * weight for weight in query.values()))
    document_norm = math.sqrt(sum(weight * weight for weight in document.values()))
    if query_norm == 0.0 or document_norm == 0.0:
        return 0.0
    overlap = sum(weight * document.get(term, 0.0) for term, weight in query.items())
    return overlap / (query_norm * document_norm)


class TFIDF(_LexicalMemory):
    """TF-IDF retrieval over an in-process per-isolation store, ranked by cosine similarity."""

    name = "tfidf"

    def _score(self, query: Counter[str], corpus: list[Counter[str]]) -> list[float]:
        total = len(corpus)
        if total == 0:
            return []
        frequency = document_frequency(corpus)
        idf = {term: math.log(total / count) for term, count in frequency.items()}
        query_vector = _weighted(query, idf)
        return [_cosine(query_vector, _weighted(document, idf)) for document in corpus]


__all__ = ["TFIDF"]
