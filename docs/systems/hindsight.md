# Hindsight -- `hindsight`

> **Deprecated:** this page describes memrank's older, pre-agent surface, which still ships but will be removed. To evaluate an agent, use `memrank run`; see the [README](../../README.md).

```python
from memrank.systems import Hindsight

system = Hindsight(base_url="http://localhost:8888", api_key="...")
```

An HTTP client for Vectorize.io's Hindsight agent memory service.

## How it works

The engine runs as a service. Memrank gives each group of [tasks](../reference/task.md) its own
bank, stores each document supplied by the [evaluation](../reference/evaluation.md), and searches
that bank for each question.

Hindsight's own design, as its authors describe it: retaining splits a document into facts;
recall runs several strategies at once -- meaning-based, keyword, graph expansion and temporal --
combines their rankings, reranks candidates and selects results within a token budget.
It limits tokens rather than result count; apply the context-budget rules in
[methodology](../methodology.md) when comparing it with other systems.

## Why it matters

Use this client to evaluate Hindsight under Memrank's protocol. Vendor-published scores may use
different retrieval limits, readers or judges; compare configurations before comparing scores.

## What it needs

A running Hindsight service. Its address comes from `base_url=` or `HINDSIGHT_API_URL`,
defaulting to `http://localhost:8888`, and a key, where auth is enabled, from
`HINDSIGHT_API_KEY`.

## References

- [vectorize-io/hindsight](https://github.com/vectorize-io/hindsight) -- the engine, MIT
  licensed.
- [arXiv:2512.12818](https://arxiv.org/abs/2512.12818) -- *Hindsight is 20/20: Building Agent
  Memory that Retains, Recalls, and Reflects*.
- [Methodology](../methodology.md) -- why a vendor's published number and a memrank row are not
  the same measurement.
