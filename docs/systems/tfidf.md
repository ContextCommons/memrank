# TFIDF -- `tfidf`

> **Deprecated:** this page describes memrank's older, pre-agent surface, which still ships but will be removed. To evaluate an agent, use `memrank run`; see the [README](../../README.md).

```python
from memrank.systems import TFIDF

system = TFIDF()
```

Keyword search weighted by how rare each word is: the classic lexical baseline, in plain Python.

## How it works

It stores documents in a Python list, one list per group of
[tasks](../reference/task.md). When asked a question it splits the question and every stored
document into words -- lowercased, cut at anything that is not a letter or a digit -- and assigns
each word a weight based on term frequency and inverse document frequency.

Term frequency counts occurrences in a document. Inverse document frequency measures rarity
across the group: `ln(N / df)`, where `N` is how many documents there are and `df` how many of
them contain the word. A word in every document scores exactly zero, which is what stops *the*
and *was* from deciding the ranking.

Question and document each become a vector of those weights, and the score is the **cosine**
between them -- the overlap divided by both lengths. Dividing by the document's length is what
lets a short document that is mostly about the question outrank a long one that merely mentions
it. Documents scoring zero are dropped, and the rest come back highest first.

The implementation needs no search index, model or additional dependency. [`BM25`](bm25.md)
uses the same document store and tokenizer with a different scoring rule.

## Why it matters

Use TF-IDF as a weighted keyword baseline for your evaluation. Compare scores under the same
retrieval limit and measure to assess what your system adds over this method.

TFIDF is used in the local installation check. It runs in the Python process, so its latency
should not be ranked directly against HTTP clients; see [methodology](../methodology.md).

## What it needs

No service, API key, network access or download.

## References

- [system](../reference/system.md) -- what a system is.
- [`BM25`](bm25.md) -- an alternative keyword scoring method.
- [`WordOverlap`](word-overlap.md) -- an unweighted keyword baseline.
- [Methodology](../methodology.md) -- measurement rules and comparison limits.
- Karen Sparck Jones, [*A statistical interpretation of term specificity and its application in
  retrieval*](https://doi.org/10.1108/eb026526), Journal of Documentation 28(1), 1972 -- where
  inverse document frequency comes from.
- Gerard Salton, A. Wong and C. S. Yang, [*A vector space model for automatic
  indexing*](https://doi.org/10.1145/361219.361220), Communications of the ACM 18(11), 1975 --
  the vector-space model and cosine similarity.
