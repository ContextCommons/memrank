# Mem0 -- `mem0`

> **Deprecated:** this page describes memrank's older, pre-agent surface, which still ships but will be removed. To evaluate an agent, use `memrank run`; see the [README](../../README.md).

```python
from memrank.systems import Mem0

system = Mem0(base_url="http://localhost:8888")   # mode="http", the default
in_process = Mem0(mode="sdk")                     # needs the mem0 extra
```

A client for Mem0's memory layer, using its Python SDK or a local HTTP server.

## How it works

Memrank adds each document it is given to mem0 under a user id of its own per group of
[tasks](../reference/task.md), and searches that user's memories at question time.

Mem0's own design, as its authors describe it: adding a document is not storing it verbatim -- a
model extracts facts from it, and those facts are what is stored, updated and searched.

There are two modes and memrank never chooses between them silently. `http` talks to a running
server and is the default. `sdk` imports mem0 and runs it inside your process, which makes its
latency incomparable with an engine reached over the network; it is opt-in, so installing the SDK
never changes how a run behaves.

## Why it matters

Mem0 can paraphrase facts during extraction. A word-matching [measure](../reference/measure.md)
can therefore miss correct information; use the evaluation's judged protocol when assessing
answer quality.

Published hosted-platform scores use their own retrieval settings and infrastructure. They are
not directly comparable with a local Memrank run unless those conditions match.

## What it needs

Either a running mem0 server -- its address from `base_url=` or `MEM0_HTTP_URL`, defaulting to
`http://localhost:8888` -- or, for `mode="sdk"`, the SDK installed with the `mem0` extra
(`pip install "memrank[mem0]"`) plus whatever model and store credentials mem0 itself reads.

## References

- [mem0ai/mem0](https://github.com/mem0ai/mem0) -- the engine, Apache-2.0 licensed.
- [mem0ai/memory-benchmarks](https://github.com/mem0ai/memory-benchmarks) -- the vendor's own
  harness, where its published settings are stated.
- [Methodology](../methodology.md) -- why a word-match measure is not a quality metric for a
  system that paraphrases.
