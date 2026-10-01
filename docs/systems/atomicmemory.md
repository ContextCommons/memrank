# AtomicMemory -- `atomicmemory`

> **Deprecated:** this page describes memrank's older, pre-agent surface, which still ships but will be removed. To evaluate an agent, use `memrank run`; see the [README](../../README.md).

```python
from memrank.systems import AtomicMemory

system = AtomicMemory(base_url="http://localhost:3070", api_key="...")
```

An HTTP client for AtomicStrata's AtomicMemory service.

## How it works

Memrank sends each document supplied by the [evaluation](../reference/evaluation.md) to the
service's memories endpoint, then retrieves memories relevant to each question. Each group
of [tasks](../reference/task.md) uses a separate namespace, which the client deletes during
cleanup to isolate independent groups.

The client also returns the engine's graph snapshot in the normalized format used by the
[relation-graph evaluation](../evaluations/relation-graph.md).

## Why it matters

AtomicStrata maintains Memrank and develops AtomicMemory. AtomicMemory is subject to the same
[measures](../reference/measure.md), context-budget rules and diagnostic controls as other
engines. Interpret its results using the recorded configuration and evidence rather than the
maintainer's relationship to the engine.

## What it needs

A running AtomicMemory service. Set its address with `base_url=` or `ATOMICMEMORY_API_URL`.
Where authentication is enabled, supply a key with `api_key=` or `ATOMICMEMORY_API_KEY`.

Cleanup deletes evaluation namespaces. Use a dedicated evaluation store, not data you need
to retain.

## References

- [System reference](../reference/system.md) -- measurements and engine declarations.
- [Engine images](../misc/engine-images.md) -- container availability.
- [Methodology](../methodology.md) -- comparing HTTP clients with local controls.
