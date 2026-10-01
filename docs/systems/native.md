# Native -- `native`

> **Deprecated:** this page describes memrank's older, pre-agent surface, which still ships but will be removed. To evaluate an agent, use `memrank run`; see the [README](../../README.md).

```python
from memrank.systems import Native

system = Native(base_url="http://localhost:8099")
```

An HTTP client for a translator implementing Memrank's system contract. Use it to evaluate
an external memory implementation without adding its integration code to Memrank.

## How it works

You run a translator service that accepts Memrank's requests and calls your engine. The
translator can be written in any language. Memrank requests its description, then calls the
lifecycle endpoints to prepare an isolated group of [tasks](../reference/task.md), ingest
documents, retrieve relevant documents and clean up.

The translator reports its engine configuration because Memrank cannot infer how an external
implementation is configured. Memrank records that report as a declaration. It also records
`translator` as the transport class: measured latency includes the additional service call
and cannot be compared directly with a client that calls an engine without a translator.

Memrank refuses a translator that reports an unsupported contract version.

## Why it matters

A translator lets you evaluate an engine from your own repository, without a Memrank fork or
pull request. You maintain the engine-specific integration alongside your implementation.

## What it needs

A running translator that implements the contract. Its address comes from `base_url=` or
`NATIVE_API_URL`, defaulting to `http://localhost:8099`.

## References

- [System contract](../system-contract.md) -- endpoints, request and response formats, and
  conformance checks.
- [System reference](../reference/system.md) -- measurements and system declarations.
- [Methodology](../methodology.md) -- transport comparison rules.
