# trace

> **Deprecated:** this page describes memrank's older, pre-agent surface, which still ships but will be removed. To evaluate an agent, use `memrank run`; see the [README](../../README.md).

A **trace** records one [task](task.md) attempt: the context document IDs, response, timings
and any error. Inspect it to understand the evidence behind a measured value.

```python
from memrank.evaluations import Demo
from memrank.systems import WordOverlap

result = Demo().run(system=WordOverlap())
trace = result.traces_of("q_job")[0]

print("task:    ", trace.task.id, "|", trace.task.prompt)
print("given:   ", trace.given.count, "document(s):", trace.given.document_ids)
print("recalled:", [document.id for document in trace.recalled.documents])
print("timed:   ", sorted(trace.timings_ms))
print("error:   ", trace.error)
```

## What it is

Memrank records one trace per task attempt, including failed attempts. The `error` field names
the failed step and message. Built-in task measures return `None` when required evidence is
missing; custom measures define their own handling.

| Field | What it holds |
|---|---|
| `task` | the task definition, with `context` omitted; supplied context IDs are in `given` |
| `group` | the task's group identifier |
| `attempt` | the attempt number |
| `given` | what memrank gave the system before the prompt: the document ids and how many |
| `recalled` | what a memory or retriever returned -- the passages, ranked best first |
| `answered` | what a model or assistant wrote, when the system is one that answers |
| `timings_ms` | memrank's own clock, per step, at memrank's own call boundary |
| `declared` | what only the system knew and chose to state: its version, tokens a provider billed, time only it can see, a fingerprint of its state. `None` means "did not state", never zero |
| `error` | the failed step and message, or `None` |
| `started`, `finished` | attempt timestamps |

Memrank preserves the order of `recalled.documents`. Retrieval systems should return their
highest-ranked documents first; diagnostic controls may deliberately return ingestion order.

Saved traces can be measured again without rerunning the system. A new measure can use only
evidence the traces contain; original context content is not stored in `trace.task`.

## Who supplies what

Memrank records traces during execution. Your system may supply optional declarations.

## The Python names

- `memrank.Trace` -- the class. Fields as above.
- `result.traces` -- every trace of a run, in order.
- `result.traces_of("<task id>")` -- the recorded attempts for one task.
- `trace.recalled` is a `memrank.Recall`: `documents`, and whatever the system chose to
  `declare` about the call.
- `result.save(path)` and `memrank.Result.load(path)` -- traces written to disk and read back,
  typed, in another process.

## Going deeper

- [result](result.md) -- the object holding the traces and the values together.
- [measure](measure.md) -- the rules that read traces and produce values.
- [The translator contract](../system-contract.md) -- which endpoint fills which trace field
  when the system is driven over HTTP.
