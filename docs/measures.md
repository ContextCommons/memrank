# Measures

> **Deprecated:** this page describes memrank's older, pre-agent surface, which still ships but will be removed. To evaluate an agent, use `memrank run`; see the [README](../README.md).

A [measure](reference/measure.md) produces named values from recorded traces. This guide covers
the measure contract, the five included measures and how to measure saved results.

Measures run after task execution, so you can apply a new measure to saved traces without
calling the system again.

[`docs/evaluations.md`](evaluations.md) shows a measure bundled into an evaluation, and
[`docs/methodology.md`](methodology.md) says what memrank claims the shipped values mean. The
source is `memrank/instrument/measure.py` and `memrank/instrument/measures.py`.

## What a measure declares

Implement `measure(traces, values)` and declare these four class attributes:

| Declares | Attribute | What it means |
|---|---|---|
| its name | `name` | The name every value of this measure carries, and the name another measure reads it by. |
| its scope | `scope` | `Scope.TASK` -- one value per trace. `Scope.RUN` -- one value (or one per group) for the whole run. |
| what it reads | `reads` | Trace fields, value names, or both. A trace field must be one of `given`, `recalled`, `answered`, `timings_ms`, `declared`, `error` (`memrank.instrument.trace.TRACE_FIELDS`). A value name must be the `name` of a measure the run produces. |
| who decides | `decider` | `Decider.MEMRANK` (memrank's own clock and bookkeeping), `Decider.RULE` (a fixed rule -- string matching, a structural check), `Decider.MODEL` (a model adjudicated it), `Decider.SYSTEM` (a system declaration). |

```python
from collections.abc import Sequence

from memrank import Decider, Measure, Scope, Trace, Value


class AnswerLength(Measure):
    name = "answer-length"
    scope = Scope.TASK
    reads = ("answered",)
    decider = Decider.RULE

    def measure(self, traces: Sequence[Trace], values: Sequence[Value]) -> list[Value]:
        """Read the traces and whatever has been measured so far; produce values."""
        return [Value(measure=self.name, decider=self.decider, task_id=trace.task_id,
                      value=None if trace.answered is None else float(len(trace.answered.text)),
                      why="characters in the answer as written")
                for trace in traces]
```

`values` contains outputs from earlier measures in the same pass. A measure can declare those
names in `reads` and use their values without modifying them.

A measure that produces more than one number gives each value its own name, with a dotted
suffix under the measure's own: `Latency` declares `name = "latency"` and produces
`latency.retrieve.p50`, `latency.retrieve.p95` and so on. Two values of one measure are never
told apart by position.

## The refusal, before the run

A measure that reads a name nothing produces is refused **before the system is touched**,
rather than raising halfway through a run that has already spent time and money. The reason
names every trace field and every value name that run does have:

```python
from collections.abc import Sequence

import memrank
from memrank import Decider, Evaluation, Measure, Scope, Trace, Value
from memrank.systems import WordOverlap


class AnswerLength(Measure):
    name = "answer-length"
    scope = Scope.TASK
    reads = ("anwsered",)                 # a typo, and nothing in the run produces it
    decider = Decider.RULE

    def measure(self, traces: Sequence[Trace], values: Sequence[Value]) -> list[Value]:
        raise AssertionError("never reached: the run refuses before the system is touched")


demo = memrank.evaluation("demo")
mine = Evaluation(name=demo.name, version=demo.version, tasks=demo.tasks,
                  measures=(AnswerLength(),), clearing=demo.clearing)

print(mine.run(system=WordOverlap()))
```

<!-- output: exact -->
```console
system:     WordOverlap (memory), version None
evaluation: demo at memrank-demo@v1+def0, 5 task(s), cleared per-group
REFUSED before the system was touched: measure 'answer-length' reads anwsered, which nothing in this run produces; the trace fields are answered, declared, error, given, recalled, timings_ms and the value names are answer-length
```

A refusal is a result: no traces, a stated reason, and `result.refused` is `True`.

A measure reading `answered` requires a system that generates answers or an `answerer=`
supplied to the run. Otherwise setup is refused:

```console
REFUSED before the system was touched: judge reads `answered`, a memory system only recalls, and no answer writer was given; pass answerer=<your writer> to evaluation.run (memrank ships no reader that runs in-process without a model key)
```

Both checks belong to the run's setup (`memrank/instrument/refusal.py`). `memrank.measure`,
below, applies measures to traces that already exist, so there is no system to protect and no
setup to refuse: a measure applied there reads what the stored traces hold.

## The value it produces

Every value is a `memrank.Value`, and it is never a bare number:

| Field | What it holds |
|---|---|
| `measure` | The measure's name, which is how a reader picks it out: `result.values_of("word-match")`. |
| `decider` | Who decided it. Carried on every value, so no number in a result is anonymous. |
| `task_id` | The task it is about, for a task-scope value. `None` for a run-scope one. |
| `group` | The group it is about, when a measure works at a group's granularity. `None` otherwise. |
| `value` | A float, a bool, or `None`. |
| `why` | Why, when the measure can say: a rationale, the matched span, or the reason it is `None`. |

### Absent is not zero

`0.0` means the measure decided, and the answer was zero. `None` means it could not decide,
and `why` says what stopped it. The two are different facts and memrank never collapses them:

```python
import memrank
from memrank import Document, Recall


class Unreachable(memrank.Memory):
    """A memory whose retrieve fails, so nothing is recalled and nothing can be scored."""

    def prepare(self, isolation_id: str) -> None: ...
    def ingest(self, documents: list[Document]) -> None: ...
    def cleanup(self) -> None: ...

    def retrieve(self, query: str, k: int = 10) -> Recall:
        raise ConnectionError("no route to the service")


result = memrank.evaluation("demo").run(system=Unreachable())

for value in (result.values_of("word-match")[0], *result.values_of("demo-score"),
              *result.values_of("failure-rate")):
    print(f"{value.measure:<13} {value.value!r:<6} {value.why}")
```

`word-match` and `demo-score` are `None` because retrieval failed. `failure-rate` is `1.0`: every
task attempt recorded an error. A failed request must not be treated as successful retrieval
with no relevant documents.

## The measures memrank ships

Five measures ship with the package, implemented in `memrank/instrument/measures.py`:

| Measure | `name` | Scope | Reads | Decider |
|---|---|---|---|---|
| `WordMatch` | `word-match` | task | `recalled` | rule |
| `Judge` | `judge` | task | `answered` | model |
| `Latency` | `latency` | run | `timings_ms` | memrank |
| `FailureRate` | `failure-rate` | run | `error` | memrank |
| `BenchmarkScore` | `<benchmark>-score` | run | `recalled`, `declared` | rule |

**`WordMatch`** asks whether the gold spans appear verbatim in something the system recalled --
the arithmetic of `memrank.SpanRecall`, per task instead of per unit. `1.0` is a span found,
`0.0` a span not found, `None` a task that broke. It is a retrieval proxy and **not answer
correctness**, and says so in the `why` of every value, so the caveat travels with the value.

**`Judge`** has a model adjudicate the answer against what was expected; `value` is a bool and
`why` the model's rationale. Construction needs no key, but measurement requires
`ANTHROPIC_API_KEY` and raises if it is missing. Setup validation separately checks whether
answers will be available. See [methodology](methodology.md) for judging and data disclosure.

**`Latency`** reports p50 and p95 per step, over the timings memrank took at its own call
boundary -- never a figure a system reported about itself. Each value carries the sample count
it was taken over, because a percentile over three samples is a different object from one over
three hundred. It is reported, never ranked and never asserted on.

**`FailureRate`** is the share of traces carrying an error, from memrank's own bookkeeping.
`why` names the failed steps. With no traces at all it is `None`, not `0.0`.

**`BenchmarkScore`** applies an in-tree benchmark's own `score()` over the traces that benchmark
produced. Its name is the benchmark's, suffixed `-score` (`demo-score`), and its scope is the
run because a benchmark's unit **is** the group: it produces one value per group and nothing
that combines them. No run-level number is invented on a benchmark's behalf. A group in which
every task failed, and a benchmark that refuses to score what it was handed, both come back as
`None` with the reason.

The shipped evaluations bundle `BenchmarkScore`, `Latency` and `FailureRate`, plus `WordMatch`
where a substring proxy means anything for that dataset. `Judge` is not bundled; add it explicitly when you need model-based scoring.

## Writing your own, over a run that already happened

`memrank.measure(result, *measures)` applies measures to a result's stored traces and returns a
new result carrying the values it already had plus the new ones. Nothing is run again.

The measure below is run-scope and reads another measure's name rather than a trace field,
which is how measures combine. It runs offline, against `demo`, with no key:

```python
from collections.abc import Sequence

import memrank
from memrank import Decider, Measure, Result, Scope, Trace, Value
from memrank.systems import WordOverlap


class MatchRate(Measure):
    """What share of the tasks word-match decided, it marked as a hit."""

    name = "word-match-rate"
    scope = Scope.RUN
    reads = ("word-match",)
    decider = Decider.RULE

    def measure(self, traces: Sequence[Trace], values: Sequence[Value]) -> list[Value]:
        mine = [v for v in values if v.measure == "word-match"]
        decided = [v for v in mine if v.value is not None]
        if not decided:
            return [Value(measure=self.name, decider=self.decider, value=None,
                          why="word-match decided no task, so there is no share to state")]
        hits = sum(1 for v in decided if v.value)
        return [Value(measure=self.name, decider=self.decider,
                      value=hits / len(decided),
                      why=f"{hits} of {len(decided)} decided task(s) matched; "
                          f"{len(mine) - len(decided)} undecided task(s) left out")]


result = memrank.evaluation("demo").run(system=WordOverlap())
result.save("/tmp/demo-run.json")                 # the traces persist, typed

stored = Result.load("/tmp/demo-run.json")        # a different process, days later
measured = memrank.measure(stored, MatchRate())

for value in measured.values_of("word-match-rate"):
    print(value.value, "--", value.why)
```

The example excludes missing values from the denominator and reports that count in `why`.
It returns `None` if no values are available. Its `decider` is `RULE` because it computes a
fixed arithmetic rule; model-based scoring would declare `MODEL` instead.

To bundle it into a run instead, put it in an evaluation's `measures` alongside `WordMatch`;
order matters, because a measure only sees what the measures before it produced. Bundled that
way it is subject to the refusal above: an evaluation whose benchmark ships no `WordMatch`
refuses this measure before it runs rather than handing it an empty list.

## What memrank will not do for you

- **It combines nothing unless a measure says it does.** There is no overall score across
  measures, and no average across groups a benchmark declined to average.
- **It reports no value without a decider.** Where you cannot say who decided, the value does
  not belong in a result.
- **It substitutes nothing for a value it could not get.** Absent stays `None`, with the reason.
