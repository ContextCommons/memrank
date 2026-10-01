# Demo -- `demo`

> **Deprecated:** this page describes memrank's older, pre-agent surface, which still ships but will be removed. To evaluate an agent, use `memrank run`; see the [README](../../README.md).

```python
from memrank.evaluations import Demo

evaluation = Demo()
```

One small hand-written conversation and five questions about it -- the dependency-free smoke
evaluation.

## How it works

The bundled scenario is three short messages from one person: they moved city for a job, a
relative is visiting, they have an allergy, they order a particular dinner. Memrank hands those
to the [system](../reference/system.md) as documents, then asks five questions about them --
including one that checks that an unsupported allergy claim is not retrieved.

Each question states the words a correct recall has to contain, and which message is the evidence
for it.

## Why it matters

The bundled scenario is useful for checking a new system or [measure](../reference/measure.md).
Paired with a local system, it runs without a download, service or API key.

It measures two things beside latency and failures: whether the expected words appeared in
something the system recalled, and whether the evidence message was recalled at all. Both are
**retrieval proxies decided by a fixed rule**, not answer correctness, and every value says so.
They are meaningful here because the scenario was written so its answers are literal spans -- a
property of this evaluation, not of word matching in general.


Its benchmark publishes no reader prompt, so a judged run answers with memrank's own
([the reader prompt](../methodology.md#the-reader-prompt)).

## What it needs

No dataset download or API key. The scenario is bundled with the package.

A scenario selected with `data_path=` or `DEMO_DATA_PATH` is not classified as bundled synthetic
data. Review its content before enabling a judge that sends it to an external provider.

## References

- [evaluation](../reference/evaluation.md) -- what an evaluation is.
- [Measures](../measures.md) -- what a word-match value declares about itself.
- [Methodology](../methodology.md) -- why a retrieval proxy is not answer correctness.
