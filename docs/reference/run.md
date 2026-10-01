# run

> **Deprecated:** this page describes memrank's older, pre-agent surface, which still ships but will be removed. To evaluate an agent, use `memrank run`; see the [README](../../README.md).

A **run** executes an [evaluation](evaluation.md) against a [system](system.md), records task
attempts and applies the evaluation's measures.

```python
from memrank.evaluations import Demo
from memrank.systems import WordOverlap

result = Demo().run(system=WordOverlap())
print(len(result.traces), "traces,", len(result.values), "values")
```

That call gave `WordOverlap` the three documents of the `demo` conversation, put the five
questions to it one at a time, timed every call at memrank's own boundary, recorded one
[trace](trace.md) per question, and applied the [measures](measure.md) `demo` bundles. It needed
no engine, no network and no key.

## What it is

Call `evaluation.run(system=...)` with a system instance. No registration or configuration
file is required.

The run proceeds in this order:

1. **Validate setup.** Refuse with a reason and no traces if the system type is unsupported,
   a required method is missing, a measure's inputs are unavailable, or the system cannot
   accept the evaluation's context. This check precedes run calls to the system.
2. **Execute tasks.** Prepare and ingest context according to the clearing rule, call the
   system for each task, and record one trace per attempt, including failures. A task failure
   does not stop the remaining tasks. Cleanup is the system implementation's responsibility;
   the result records whether cleanup returned and any failure it reported.
3. **Measure traces.** Apply the evaluation's measures to produce named values.
4. **Return a [result](result.md).**

## Who supplies what

You supply the evaluation and the system, and optionally `k=`, `attempts=` and `answerer=`.
Memrank supplies setup validation, lifecycle calls, timings and one trace per task attempt.

## The Python names

```python
import inspect

from memrank.evaluations import Demo

print(inspect.signature(Demo().run))
```

- `evaluation.run(system=...)` -- the system is the only positional argument; the evaluation is
  the receiver. Every other argument is keyword-only.
- `k=` -- how many documents to ask a memory or retriever for. Default 10.
- `attempts=` -- how many times to attempt each task. Default 1.
- `answerer=` -- a writer that turns what was recalled into an answer, needed when a measure
  reads `answered` and the system only recalls.
- `result.refusal` -- the reason, when the run refused. `None` when it went ahead.

## Going deeper

- [result](result.md) -- what comes back.
- [Methodology](../methodology.md) -- the context-budget control, the arms, and what a run is
  evidence of.
- [`examples/01-first-result/`](../../examples/01-first-result/) -- the shortest working run.
