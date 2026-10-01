# result

> **Deprecated:** this page describes memrank's older, pre-agent surface, which still ships but will be removed. To evaluate an agent, use `memrank run`; see the [README](../../README.md).

A **result** contains a [run](run.md)'s recorded traces and measured values. Use it to inspect
responses and failures, interpret scores, and save evidence for later analysis.

```python
from memrank.evaluations import Demo
from memrank.systems import WordOverlap

result = Demo().run(system=WordOverlap())

print(result.system.name, "|", result.system.kind, "|", result.evaluation.name)
for value in result.values_of("word-match"):
    print(f"{value.task_id:<14} {value.value}  decided by {value.decider.value}")
print(len(result.traces), "traces,", sum(1 for t in result.traces if t.error), "with errors")
```

<!-- output: exact -->
```console
WordOverlap | memory | demo
q_job          1.0  decided by rule
q_animal       1.0  decided by rule
q_visit        1.0  decided by rule
q_diet         0.0  decided by rule
q_allergy_neg  1.0  decided by rule
5 traces, 0 with errors
```

## What it is

A result contains:

- **`traces`** -- one [trace](trace.md) per task attempt, recording context document IDs,
  responses, timings and any error.
- **`values`** -- the outputs of [measures](measure.md), each identified by measure name and
  `decider`, with an optional explanation in `why`. There is no overall score across measures.

Interpret these values against your evaluation's criteria and the comparison's purpose.

- **Missing values:** `None` means unavailable or undeclared. It does not mean zero.
- **Failed tasks:** the trace records the failure and `failure-rate` counts it. Built-in
  task measures return `None` when the evidence they need is unavailable. Custom measures
  must define their own failure handling.
- **Refused runs:** `result.refusal` gives the reason and the result has no traces. The run
  has not called the system lifecycle.



`result.save(path)` writes a result; `memrank.Result.load(path)` loads it with its traces intact. Use `memrank.measure` to apply a new measure without
rerunning the system.

## Who supplies what

A result is returned to you. You supply nothing beyond where to save it.

## The Python names

```python
from memrank import Result
from memrank.evaluations import Demo
from memrank.systems import WordOverlap

result = Demo().run(system=WordOverlap())
result.save("/tmp/demo-run.json")
reloaded = Result.load("/tmp/demo-run.json")

print(len(reloaded.traces), "traces read back;", reloaded.refusal, "refusal")
print(sorted({value.measure for value in reloaded.values}))
```

- `memrank.Result` -- fields `schema_version`, `system`, `evaluation`, `refusal`, `traces`,
  `values`, `started`, `finished`.
- `result.values_of("<measure name>")` -- one measure's values, picked out of the rest.
- `result.traces_of("<task id>")` -- the recorded attempts for one task.
- `result.save(path)` and `Result.load(path)` -- save and load a typed result.
- `memrank.measure(result, MyMeasure())` -- a new measure over traces already stored.
- `memrank.paired(a, b)` -- [two results side by side](paired.md).

## Going deeper

- [trace](trace.md) and [measure](measure.md) -- the two halves a result holds.
- [paired](paired.md) -- comparing two of them.
- [Methodology](../methodology.md) -- measurement rules and interpretation limits.
