# WordOverlap -- `word-overlap`

> **Deprecated:** this page describes memrank's older, pre-agent surface, which still ships but will be removed. To evaluate an agent, use `memrank run`; see the [README](../../README.md).

```python
from memrank.systems import WordOverlap

system = WordOverlap()
```

A local retrieval baseline that ranks documents by the number of distinct words they share
with the query.

## How it works

It stores documents in a Python list for each group of [tasks](../reference/task.md). It
lowercases the query and document text, splits on whitespace, and counts shared distinct words.
Documents with no shared words are excluded; the remaining documents are returned in descending
score order, up to the requested limit.

It uses no term-frequency weighting, inverse document frequency, model or search index.

## Why it matters

Use it to compare a system with a simple keyword retrieval method. A higher score shows an
improvement over this baseline on the selected evaluation and measure; it does not establish
value on other workloads or justify the system's cost.

[TFIDF](tfidf.md) and [BM25](bm25.md) provide more sophisticated keyword baselines. All three
run in the Python process, so their latency should not be ranked directly against HTTP clients.
See [methodology](../methodology.md) for transport comparison rules.

## What it needs

No service, API key, network access or data download. Evaluation and answer-writer requirements
are separate.

## References

- [System reference](../reference/system.md) -- system types and methods.
- [Methodology](../methodology.md) -- retrieval baselines and diagnostic controls.
- [BEIR](https://arxiv.org/abs/2104.08663) -- evaluation of lexical and other retrieval methods.
