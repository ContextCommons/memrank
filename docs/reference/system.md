# system

> **Deprecated:** this page describes memrank's older, pre-agent surface, which still ships but will be removed. To evaluate an agent, use `memrank run`; see the [README](../../README.md).

A **system** is the implementation evaluated by Memrank: a local retrieval method, a client
connected to a memory service, a model or an assistant.

```python
from memrank.systems import TFIDF

system = TFIDF()
print(system.name, system.version)
```

[`TFIDF`](../systems/tfidf.md) ranks documents by the words they share with the query, weighted
by how rare each word is. It needs no engine, no network and no key.

## System and engine

A **system** is the object Memrank calls and measures. An **engine** is an external service or
library used by that object.

- `TFIDF` implements retrieval locally and needs no external engine.
- [`memrank.systems.AtomicMemory`](../systems/atomicmemory.md) is a client for the AtomicMemory
  engine. It sends documents to the service, retrieves memories and clears evaluation state.

A result identifies the system. Engine versions and usage appear as system declarations.

## The four kinds

A system's **kind** is determined by its base class.

| Kind | The verb it is for | What it must implement |
|---|---|---|
| `memrank.Memory` | stores supplied documents and retrieves relevant ones | `prepare`, `ingest`, `retrieve`, `cleanup` |
| `memrank.Model` | given a prompt, returns text | `complete` |
| `memrank.Retriever` | ranks an existing corpus for a query; does not ingest task context | `rank` |
| `memrank.Assistant` | responds to a list of `role`/`content` messages | `respond` |

These methods are required. Memrank measures latency at its call boundary. Systems may also
declare engine versions, token usage and internal timings through optional methods. The default
`None` records missing information, not zero.

A [run](run.md) refuses before touching the system when it is not a kind memrank knows, or when
it lacks a verb its kind requires.

## Who supplies what

You supply the system and its configuration -- `base_url=`, `api_key=`, whatever it needs.
Memrank supplies the run loop, calls your lifecycle between groups of [tasks](task.md), and
records timings and declarations. Your implementation must isolate and clear its state. See
[why a memory has four methods](../systems.md#why-a-memory-has-four-methods).

## The Python names

```python
import memrank
from memrank.systems import TFIDF, NoContext

print(TFIDF().name, NoContext().name)
print(memrank.system("tfidf").name)          # the same thing, by string
print(isinstance(TFIDF(), memrank.Memory))   # its kind, which the class decides
```

- `memrank.System` -- the base every kind derives from; what `result.system` describes.
- `memrank.Memory`, `memrank.Model`, `memrank.Retriever`, `memrank.Assistant` -- the four kinds
  you subclass.
- `memrank.systems` -- the module holding what ships: `TFIDF`, `BM25`, `WordOverlap`,
  `NoContext`, `FixedContext`, `FullContext`, `AtomicMemory`, `Hindsight`, `Supermemory`,
  `Mem0`, `Native`.
- `memrank.system("<name>")` -- construct a system from a string key, for example from configuration.
- `memrank.Document` and `memrank.Recall` -- what `ingest` is given and what `retrieve` returns.

## Going deeper

- [Adding a system](../systems.md) -- the four kinds, their verbs and the optional declarations,
  in full.
- [The translator contract](../system-contract.md) -- a system memrank drives over HTTP rather
  than imports.
- [Supported systems](../systems/README.md) -- implementations and requirements.
- [`examples/02-your-own-system/`](../../examples/02-your-own-system/) -- a working one.
