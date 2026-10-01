# Supermemory -- `supermemory`

> **Deprecated:** this page describes memrank's older, pre-agent surface, which still ships but will be removed. To evaluate an agent, use `memrank run`; see the [README](../../README.md).

```python
from memrank.systems import Supermemory

system = Supermemory(base_url="http://localhost:6767")
```

An HTTP client for Supermemory AI's self-hosted memory server.

## How it works

The engine runs as a service. Memrank clears the namespace for a group of
[tasks](../reference/task.md), stores the supplied documents as memories, splitting long documents to meet the engine's
size limit, and searches that namespace for each question.

Supermemory's own design, as its authors describe it: memories are atomic and linked to each
other by exactly three relations -- one memory *updates* another, *extends* it, or is *derived*
from it -- and search combines meaning, keyword and graph. Memrank reads that graph in the
normalized shape the [relation-graph evaluation](../evaluations/relation-graph.md) needs.

## Why it matters

This client evaluates the self-hosted binary, whose memory engine is not open source. The
vendor's published scores use its hosted platform. Treat those as different configurations
when interpreting results.

## What it needs

A running Supermemory server. Its address comes from `base_url=` or `SUPERMEMORY_BASE_URL`,
defaulting to `http://localhost:6767`.

Preparing a group of tasks clears that namespace first. Never point it at a store whose contents
you want to keep.

## References

- [supermemoryai/supermemory](https://github.com/supermemoryai/supermemory) -- the monorepo, MIT
  licensed. It carries the application and SDK layer; the memory engine ships as a prebuilt
  binary and its source is not in the repository.
- [supermemoryai/memorybench](https://github.com/supermemoryai/memorybench) -- the vendor's own
  harness.
- [Methodology](../methodology.md) -- why a hosted headline and a self-hosted row are not the
  same measurement.
