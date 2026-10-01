# FixedContext -- `fixed-context`

> **Deprecated:** this page describes memrank's older, pre-agent surface, which still ships but will be removed. To evaluate an agent, use `memrank run`; see the [README](../../README.md).

```python
from memrank.systems import FixedContext

system = FixedContext()
```

A diagnostic control that returns documents in ingestion order, without retrieval ranking.
Tracked runs apply the shared context budget to this control.

## How it works

It stores the supplied documents and returns all of them for every query, ignoring `k`.
In a tracked judged run, the reader-context stage applies the shared token budget, subject to
the evaluation's context policy. The control does not truncate documents itself.

The Python `evaluation.run()` interface passes the returned documents to any supplied
`answerer`. To compare answer quality under a fixed budget there, configure the writer to
apply the same limit to every system.

## Why it matters

Use it to compare ranked retrieval with reading from the start of the corpus under the same
context limit. Keep the reader, evaluation and other settings consistent when interpreting
differences in the [results](../reference/result.md).

Ingestion order affects this control: evidence late in the corpus may be truncated. Report
that limitation when comparing it with ranked retrieval.

## What it needs

No service, API key, network access or download for the control itself. Evaluation and
answer-writer requirements are separate.

## References

- [Methodology](../methodology.md) -- context budgets and evaluation-specific exceptions.
- [Laitenberger, Manning & Liu, EMNLP 2025](https://arxiv.org/abs/2506.03989) -- budget matching
  across comparison conditions.
- [NoContext](no-context.md) and [FullContext](full-context.md) -- the other controls.
