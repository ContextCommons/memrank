<a id="the-systems-that-ship"></a>
# Supported systems

> **Deprecated:** this page describes memrank's older, pre-agent surface, which still ships but will be removed. To evaluate an agent, use `memrank run`; see the [README](../../README.md).

A [system](../reference/system.md) is the implementation evaluated by Memrank, such as a local
retrieval method or a client connected to a memory service. Memrank includes retrieval baselines,
diagnostic controls and clients for external memory services. The table lists the available
implementations and their requirements.

| System | Key | What it is | What it needs |
|---|---|---|---|
| [TFIDF](tfidf.md) | `tfidf` | TF-IDF keyword retrieval, ranked by cosine similarity | No service or API key |
| [BM25](bm25.md) | `bm25` | Okapi BM25 keyword retrieval with term-frequency saturation and length normalization | No service or API key |
| [WordOverlap](word-overlap.md) | `word-overlap` | Retrieval ranked by the number of distinct words shared with the query | No service or API key |
| [NoContext](no-context.md) | `no-context` | Diagnostic control that returns no documents | No service or API key |
| [FixedContext](fixed-context.md) | `fixed-context` | Diagnostic control that returns documents in ingestion order; tracked runs apply the shared context budget | No service or API key |
| [FullContext](full-context.md) | `full-context` | Diagnostic control that returns all documents and declares an uncapped context budget | No service or API key |
| [AtomicMemory](atomicmemory.md) | `atomicmemory` | HTTP client for AtomicStrata's memory service | A running service; credentials if required |
| [Hindsight](hindsight.md) | `hindsight` | HTTP client for Vectorize.io's agent memory service | A running service; credentials if required |
| [Supermemory](supermemory.md) | `supermemory` | HTTP client for the self-hosted Supermemory server | A running service |
| [Mem0](mem0.md) | `mem0` | Client for Mem0 through its Python SDK or HTTP API | The SDK and its configuration, or a running service |
| [Native](native.md) | `native` | HTTP client for a translator implementing Memrank's system contract | A running translator |

`memrank.catalog()` lists these implementations at runtime. The three retrieval baselines and
three controls run locally without a service, API key or download. An evaluation or answer writer
can introduce additional requirements; check those separately.

<a id="the-standard-every-page-follows"></a>
## Choose an implementation

Use a retrieval baseline to compare your system with a local keyword method, or a diagnostic
control to examine how a measure responds to empty or unranked context. For an external memory
service, read its page for connection settings and prerequisites before constructing the client.
See [methodology](../methodology.md) for context budgets and comparison limits.

## Using one

Pass a system instance to an [evaluation](../evaluations/README.md)'s `run` method:

```python
from memrank.evaluations import Demo
from memrank.systems import TFIDF

result = Demo().run(system=TFIDF())
```

## Where to go from here

- [system](../reference/system.md) -- what a system is, and the four kinds.
- [Adding a system](../systems.md) -- evaluate your own implementation.
- [Available evaluations](../evaluations/README.md) -- choose tasks and measures.
- [Methodology](../methodology.md) -- controls, token budgets and limits on interpretation.
