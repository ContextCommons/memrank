# Understand results

> **Deprecated:** this page describes memrank's older, pre-agent surface, which still ships but will be removed. To evaluate an agent, use `memrank run`; see the [README](../README.md).

A result records what happened when a system ran an evaluation, plus the values produced by its
measures. Start with the identity and coverage, then read each measure in its own terms. There
is no overall score across measures.

## Read a value in context

| Field | What it tells you |
|---|---|
| `result.system` | The observed class and kind, and the version the system declared, if any. |
| `result.evaluation` | The evaluation name, version, task count and clearing rule. |
| `value.measure` | Which measurement this is; use that definition to interpret the number. |
| `value.decider` | Whether Memrank, a fixed rule, a model or the system itself supplied it. |
| `value.task_id` / `value.group` | The task or group it describes; neither is required for a whole-run value. |
| `value.why` | The rationale, sample count or reason a value is missing, when supplied. |
| `result.traces` | The recorded task attempts, including responses, timings and errors. |

The [installation check](install.md) reports `demo-score`: evidence recall. Higher means more
questions had the session holding their answer retrieved. It does not measure answer
correctness, answer quality or precision. `word-match` in the toy examples is a different
measure: a deterministic span-matching proxy. Read each [evaluation's page](evaluations/README.md)
and [measure definition](measures.md) before comparing numbers with similar ranges.

`latency.retrieve.p50` and `.p95` report median and 95th-percentile times in milliseconds at
Memrank's call boundary. Their `why` states how many samples were used. They summarize observed
calls in one run, not variation across independent runs. System-declared timings describe a
different boundary and should stay labelled as declarations.

## Check failure and missing-value semantics

- **A refusal** has `result.refused == True`, a reason in `result.refusal` and no traces. The run
  was refused before touching the system. Fix the stated incompatibility before comparing it.
- **A failed task** keeps a trace with `trace.error`. Task-level measures record `None` for a
  failed task, and `failure-rate` counts traces with errors. Read the recorded step and message.
- **A missing value** is `None`, not zero. It may mean a measure could not decide or the system
  did not declare something, such as token usage. Read `why` where present.
- **A measured zero** means the measure produced zero. It is a valid result with that measure's
  meaning, not evidence that the task failed.

Clearing is also evidence: `result.evaluation.cleared` is true when at least one cleanup call
completed; `clearing_note` records a cleanup failure when one occurred. Read both. The printed
clearing rule alone is not proof that a remote service deleted its state. Your system must
implement isolation.

Before quoting a score, count errors and missing values and check which tasks contributed to it.
Group-level scorers have their own treatment of partial failures. A high score over the remaining
tasks is not a substitute for the failure rate.

## Inspect one task

After [installation](install.md), this toy example runs offline. It prints the lowest measured
`word-match` value and the recorded evidence behind it. Missing values stay separate from zeros.

```python
from memrank.evaluations import Demo
from memrank.systems import WordOverlap

result = Demo().run(system=WordOverlap())
if result.refused:
    raise RuntimeError(result.refusal)

values = result.values_of("word-match")
decided = [value for value in values if value.value is not None]
print("missing:", sum(value.value is None for value in values))
print("errors:", sum(trace.error is not None for trace in result.traces))
if decided:
    lowest = min(decided, key=lambda value: value.value)
    print(lowest.task_id, lowest.value, lowest.why)
    for trace in result.traces_of(lowest.task_id):
        print(trace.task.prompt, trace.task.expected)
        print(trace.recalled, trace.error)
```

The trace connects a measurement to what the system received and returned. It helps distinguish
a retrieval miss, a scoring rule you did not intend, and an execution error.
[Trace reference](reference/trace.md) describes all recorded fields.

## Save and load the evidence

This complete example saves a local result and loads it back without contacting the system again:

```python
from memrank import Result
from memrank.evaluations import Demo
from memrank.systems import WordOverlap

result = Demo().run(system=WordOverlap())
path = result.save("results/demo-example.json")
reloaded = Result.load(path)

print(reloaded.system.name, reloaded.evaluation.name)
print(len(reloaded.traces), "traces loaded")
```

Use a distinct path for each run; saving to an existing path overwrites it. Saved traces contain
task content and responses, so review them before sharing. A saved result is evidence of the
observed run, not a self-contained copy of your service, dependencies or configuration. Retain
those separately when reproducibility matters. `Result.load` rejects an unknown result schema.

You can [apply a new measure](measures.md#writing-your-own-over-a-run-that-already-happened) to
stored traces with `memrank.measure`. It can only measure evidence those traces contain; it cannot
recover an answer or system state the run never recorded.

## Interpret a comparison

[Compare memory systems and their versions](comparing.md) explains task-level pairing, coverage and the sign of
a gap. Its p-values and bootstrap intervals are not repeated-run variability. A small timing gap,
a high recall score or a significant task-level difference alone does not establish performance
on another workload.

Report the candidates, evaluation version, configuration, measure, task coverage, failures and
limitations alongside a comparison. Use [methodology](methodology.md) for measurement and
publication rules and [result reference](reference/result.md) for the exact API.
