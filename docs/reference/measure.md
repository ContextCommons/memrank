# measure

> **Deprecated:** this page describes memrank's older, pre-agent surface, which still ships but will be removed. To evaluate an agent, use `memrank run`; see the [README](../../README.md).

A **measure** is one named rule that reads [traces](trace.md) and produces named values. Five
measures ship with the package:

| Measure | What it answers | Decided by |
|---|---|---|
| `<evaluation>-score` | the evaluation's own score -- `demo-score` is the first value the quick start prints. Each shipped evaluation says on its own page what its score is and is not | a fixed rule |
| `word-match` | did the expected words appear in something the system recalled? It is retrieval, not answer correctness, and says so on every value | a fixed rule |
| `latency` | how long ingest and retrieve took, at memrank's own call boundary | memrank's clock |
| `failure-rate` | how many traces carry an error | memrank's bookkeeping |
| `judge` | was the answer right? | a model, which is why it needs a key and the others do not |

```python
import memrank

word_match = memrank.WordMatch()
print(word_match.name, word_match.scope.value, word_match.reads, word_match.decider.value)
```

## What it is

A measure declares three things **before** it runs:

1. **`scope`** -- `Scope.TASK` (one value per task) or `Scope.RUN` (one for the whole run);
2. **`reads`** -- which trace fields and which other measures' values it needs;
3. **`decider`** -- who is responsible for the value. `Decider.MEMRANK` is memrank's own clock
   and bookkeeping, `Decider.RULE` a fixed rule, `Decider.MODEL` a model that adjudicated, and
   `Decider.SYSTEM` a system declaration.

Before calling the system, the run checks that each measure names available trace fields or
measure outputs. An invalid dependency produces a refusal listing the available names.

Measures run after task execution. To measure saved traces, `memrank.measure(result, MyMeasure())` returns a new [result](result.md) with
the extra values in it, and the system is not called again.

A measure that produces several values names each one -- `latency.retrieve.p50` and
`latency.retrieve.p95` are told apart by name and never by position. Memrank invents no overall
score across measures.

A measure should return `None` with a reason when it cannot produce a value, rather than
reporting a measured zero.

## Who supplies what

Memrank supplies the five measures above, recorded traces and dependency validation. You supply a measure of your own when the shipped ones do not
answer your question: its `name`, `scope`, `reads`, `decider`, and a `measure()` method taking
traces and values and returning values.

## The Python names

```python
from collections.abc import Sequence

import memrank
from memrank import Decider, Measure, Scope, Trace, Value
from memrank.evaluations import Demo
from memrank.systems import WordOverlap


class RecalledCount(Measure):
    """How many documents came back for each task."""

    name, scope, reads, decider = "recalled-count", Scope.TASK, ("recalled",), Decider.RULE

    def measure(self, traces: Sequence[Trace], values: Sequence[Value]) -> list[Value]:
        return [Value(measure=self.name, decider=self.decider, task_id=trace.task_id,
                      value=float(len(trace.recalled.documents)) if trace.recalled else None,
                      why="documents the system returned")
                for trace in traces]


result = Demo().run(system=WordOverlap())
measured = memrank.measure(result, RecalledCount())
print(measured.values_of("recalled-count")[0].value)
```

- `memrank.Measure` -- the base class. You set `name`, `scope`, `reads`, `decider` and write
  `measure(traces, values)`.
- `memrank.Scope` -- `TASK`, `RUN`.
- `memrank.Decider` -- `MEMRANK`, `RULE`, `MODEL`, `SYSTEM`.
- `memrank.Value` -- what a measure produces: `measure`, `decider`, `task_id`, `group`,
  `value`, `why`.
- `memrank.WordMatch`, `memrank.Latency`, `memrank.FailureRate`, `memrank.Judge` -- four of the
  five included measure classes, which you can instantiate directly.
- `memrank.BenchmarkScore` -- the fifth. Included evaluations construct it around their benchmark scorer.
- `memrank.measure(result, *measures)` -- applying measures to a result already stored.

## Going deeper

- [The measures memrank ships](../measures.md) -- each one in full.
- [Methodology](../methodology.md) -- measurement rules and interpretation limits.
- [`examples/04-your-own-measure/`](../../examples/04-your-own-measure/) -- a working one.
