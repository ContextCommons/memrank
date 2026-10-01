# NoContext -- `no-context`

> **Deprecated:** this page describes memrank's older, pre-agent surface, which still ships but will be removed. To evaluate an agent, use `memrank run`; see the [README](../../README.md).

```python
from memrank.systems import NoContext

system = NoContext()
```

A diagnostic control that returns no documents. With an answer writer, it tests responses
without retrieved context.

## How it works

It accepts and stores the supplied documents, but returns an empty list for every query.
The [run](../reference/run.md) proceeds normally. If an answer writer is configured, it receives
no retrieved context and must answer using the information already available to it.

## Why it matters

Compare answers with and without retrieved context to investigate whether retrieval
helps on the selected tasks. Similar scores can indicate that the reader can answer those
tasks without retrieved evidence; they do not establish that memory has no value elsewhere.

Without an answer writer and answer-scoring [measure](../reference/measure.md), the control tests
retrieval scoring only. It can receive credit for a negative task that requires a forbidden
span to remain absent, but it cannot retrieve evidence for a positive task.

## What it needs

No service, API key, network access or download for the control itself. Answer-quality
comparisons need an answer writer and scoring measure. A model-based judge needs credentials.

## References

- [Methodology](../methodology.md) -- diagnostic controls and their limits.
- [Roberts et al., EMNLP 2020](https://aclanthology.org/2020.emnlp-main.437/) -- answering
  without retrieved context, described as *closed-book* question answering.
- [FixedContext](fixed-context.md) and [FullContext](full-context.md) -- the other controls.
