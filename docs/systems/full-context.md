# FullContext -- `full-context`

> **Deprecated:** this page describes memrank's older, pre-agent surface, which still ships but will be removed. To evaluate an agent, use `memrank run`; see the [README](../../README.md).

```python
from memrank.systems import FullContext

system = FullContext()
```

A diagnostic control that returns all stored documents in ingestion order and declares an
uncapped context budget.

## How it works

Like [FixedContext](fixed-context.md), it stores the supplied documents and returns all of them
for every query, without ranking or applying `k`. It sets `context_budget = "uncapped"`, which
the tracked-run pipeline uses to bypass the shared retrieval token budget.

The Python `evaluation.run()` interface returns the documents to any supplied `answerer`;
that writer is responsible for fitting them into its model's context window.

## Why it matters

Use this control on small corpora to compare retrieval with supplying the entire corpus.
A full-context score is an observed comparison point, not a guaranteed upper bound: answer
quality still depends on the reader, the task and how it handles relevant and irrelevant text.
A matching score alone does not establish that an evaluation cannot distinguish systems.

## What it needs

No service, API key, network access or download for the control itself. Answer generation may
require a model and credentials. The corpus must fit the reader's context window for a
full-context comparison.

## References

- [Methodology](../methodology.md) -- context budgets and comparison limits.
- [NoContext](no-context.md) and [FixedContext](fixed-context.md) -- the other controls.
