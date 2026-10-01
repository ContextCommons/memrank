# Adding an evaluation

> **Deprecated:** this page describes memrank's older, pre-agent surface, which still ships but will be removed. To evaluate an agent, use `memrank run`; see the [README](../README.md).

An [**evaluation**](reference/evaluation.md) bundles its [tasks](reference/task.md), the
[measures](reference/measure.md) that read them, and the rule for when the system's state is
cleared. This guide shows how to express tasks and success criteria you have chosen. You decide
whether those cases represent your use case; Memrank does not certify their relevance.

Scoring is implemented by the measures bundled in an evaluation.
Writing your own questions and writing your own measure are separate choices.

memrank's own evaluations and yours are the same kind of object. You do not have to put yours
inside this package, register it anywhere, or give it a name anyone else can resolve.

## 1. Write the evaluation directly

Define tasks as data and select a scoring measure:

```python
import memrank
from memrank import Clearing, Document, Evaluation, Expected, Task

notes = (Document(id="t1", user_id="acme",
                  content="Acme moved to the enterprise plan in March."),)

tickets = Evaluation(
    name="tickets", version="internal@2026-09",
    tasks=(Task(id="q_plan", prompt="What plan is Acme on?", group="acme", context=notes,
                expected=Expected(answers=("enterprise",), required_spans=("enterprise",),
                                  evidence_doc_ids=("t1",))),),
    measures=(memrank.WordMatch(),),
    clearing=Clearing.PER_GROUP)

result = tickets.run(system=memrank.system("word-overlap"))
print(result.values_of("word-match")[0].value)
```

This example requires no registration, configuration file or change to Memrank.

A [**task**](reference/task.md) describes its expected outcome in `Expected`. The measure
applies the scoring rule. With `WordMatch`, `polarity="negative"` tests that forbidden spans
are absent from retrieval.

With `Clearing.PER_GROUP`, tasks with the same `group` share state. The [run](reference/run.md)
ingests each group's documents once, in first-seen order and deduplicated by ID, then calls cleanup
after the group. `Clearing.PER_TASK` and `Clearing.AT_END` are the other two rules, and the
[result](reference/result.md) records which one was in force and whether cleanup returned successfully.

Use `version` to identify the task and scoring definition. `memrank.paired` requires matching
evaluation names and versions; change the version when the definition changes. Where what the evaluation asks cannot be
frozen, `memrank.instrument.evaluation.UNFREEZABLE` says so and is recorded in every result,
rather than a version being invented.

Where the tasks themselves come from is yours: a literal in the file, a CSV beside your script,
a database, a support queue. memrank reads them off the `Evaluation` and asks nothing about
where they were built.

The runnable version is
[`examples/03-your-own-evaluation/`](../examples/03-your-own-evaluation/README.md):

```bash
uv run python examples/03-your-own-evaluation/run.py
```

## 2. Bundle the measures -- or write one afterwards

A [measure](reference/measure.md) is a named rule from traces to values. It declares its scope
(one task, or the whole run), which trace fields and value names it reads, and who decides.
[`docs/measures.md`](measures.md) is that contract in full, and it is the page to read before
writing one.

`measures=` selects the measures applied after task execution. Apply another measure to saved
[traces](reference/trace.md) with `memrank.measure(result, MyMeasure())`, without rerunning the system. A
measure that reads a name nothing in the run produces is refused before the run.

[`examples/04-your-own-measure/`](../examples/04-your-own-measure/README.md) is a worked one,
applied to a result loaded back off disk.

---

<a id="not-core-named-evaluations-and-what-the-command-line-resolves"></a>
## Register an evaluation for the command line

Everything above uses the Python interface. What follows is the catalog of evaluations reachable **by name**,
from [the command line](misc/command-line.md), in the cloud, in a sweep, or by other people. Use this route when an evaluation needs a catalog reference.

That surface keeps its own older words, listed in full under
[its vocabulary](misc/command-line.md#its-vocabulary): an *eval* is its word for a named
evaluation, `Benchmark` is the class a name resolves to, and a *unit* is one scoring context
inside a benchmark, which becomes a task group when the benchmark is converted.

### Naming one: `Benchmark`, with `load()` and `score()`

A named evaluation is built from a `Benchmark`: a **loader** that produces the data, and a
**scorer** that says what a good answer is. `memrank.evaluation(MyBenchmark())` converts one
into an `Evaluation`: each unit becomes a task group, its documents the
group's context, its queries the tasks, and its own `score()` one measure beside the ones that
need no answer writer.

```python
import memrank
from memrank import Benchmark, BenchmarkUnit, Document, SpanRecall


class TicketsBenchmark(Benchmark):
    name = "tickets"
    dataset_version = "internal@2026-08"

    def load(self) -> list[BenchmarkUnit]:
        return [BenchmarkUnit(
            unit_id="acme", isolation_id="acme",
            documents=[Document(id="t1", user_id="acme",
                                content="Acme moved to the enterprise plan in March.")],
            queries=[{"id": "q_plan", "text": "What plan is Acme on?",
                      "required_spans": ["enterprise"]}])]

    def score(self, unit, responses) -> dict:
        return SpanRecall().score(unit, responses)

    def report_template(self) -> str:
        return "tickets"


result = memrank.evaluation(TicketsBenchmark()).run(
    system=memrank.system("word-overlap"))
print(result.values_of("tickets-score")[0].value, result.evaluation.name)
```

Implement `load()`, `score()` and `report_template()` to supply a benchmark.

### Implement `load()`

Return a list of `BenchmarkUnit`. Each unit is a self-contained scoring context: documents to
ingest, queries to ask, and an `isolation_id` so a system can scope its state. Units never see
each other, and a unit becomes a task **group** when the benchmark is converted.

For datasets with a per-conversation isolation primitive (LoCoMo, BEAM), emit one unit per
conversation. For per-question isolation (LongMemEval), emit one unit per question.

A query dictionary needs only `id` and `text`. The span-scoring fields (`required_spans`,
`forbidden_spans`, `evidence_doc_ids`, `kind`) are what `score()` reads; a judged run additionally
wants `gold_answers` and a `category` your judge shape recognizes. Those are the same fields
`Expected` carries on a task, which is what the conversion above maps them onto.

### Implement `score()`

Take a single unit plus the system's responses for that unit and return a dict with at least:

```python
{
    "composite": 0.0,        # float in [0, 1]
    "per_category": {...},   # category breakdown, when applicable
    "metric": "...",         # human-readable metric name
}
```

For reference answers expressed as literal spans, delegate scoring to
`memrank.SpanRecall`, the scorer used by Demo. It returns the dictionary shown above.

```python
from memrank import Benchmark, SpanRecall


class MyBenchmark(Benchmark):
    scorer = SpanRecall()

    def score(self, unit, responses):
        return self.scorer.score(unit, responses)
```

It marks whether a query's gold spans appear verbatim in a retrieved document -- a retrieval
proxy, never answer correctness, which is what its `METRIC_LABEL` says in every unit it scores.
`memrank.WordMatch` is the same arithmetic as a measure, per task rather than per unit, carrying
the same caveat on every value. `memrank.metrics`'s `score_query` and `spec_from_query` are the
per-query pieces, for a scorer that needs the verdict rather than the unit mean.

The default `BinaryJudgeShape` is right for any benchmark whose questions have a single gold
answer, and most benchmarks need nothing else. BEAM is the exception: it scores against a rubric
of atomic nuggets, one judge call each, averaged within the question. Where your answers are
graded differently, override `judge_shape()` and return a `JudgeShape`
(`memrank/judging/shape.py`), which declares what grading one query **costs** -- split into the
cache-shared control half and the per-system context half -- and what it **yields**, a
`JudgedQuery` whose `score` is a float in [0, 1].

The constructor does not accept a `judge` callable; use `judge_shape()` for judge configuration.

#### Declare what your score is

Declare these class attributes to describe the score and its requirements. Defaults suit
benchmarks with literal reference spans; they apply to both instances and registered benchmarks:

| | Default | Set it when |
|---|---|---|
| `substring_recall_supported` | `True` | your gold answers are prose, not verbatim spans -- substring recall is then structurally ~0 and must be withheld in favour of a judge |
| `composite_rankable` | `True` | your raw `composite` is not a score to rank as-is without a judge |
| `quality_metric` | `"substring_recall"` | your composite is something else (`"graph_score"`, ...); it names the displayed column |
| `context_policy` | `"matched"` | your benchmark's own protocol hands the reader everything retrieval returned -- `"uncapped"` |
| `is_synthetic` | `False` | the data is synthetic; this informs the judging disclosure |
| `requires_graph` | `False` | scoring needs an adapter's graph snapshot; cells with a non-graph adapter are then skipped as `not_applicable` |

`substring_recall_supported` and `composite_rankable` are decoupled on purpose: a graph benchmark
sets the first `False` (substring recall is not applicable) and the second `True` (its composite
*is* a real score).

The tracked-run pipeline uses `composite_rankable` when `judge=None`: a benchmark whose composite
is not rankable gets a judge by default, because an unjudged run of it measures latency and cost
and nothing else. `substring_recall_supported` also decides the conversion: a benchmark whose span
proxy is meaningless ships no `WordMatch` measure rather than an inapplicable score.

<a id="bring-one-half-composedevaluation"></a>
### Combine tasks and a scorer with `ComposedEvaluation`

`ComposedEvaluation` composes a `Benchmark`, its units and a `Scorer`. It is the named route's
answer to the question the Python route answers with `measures=`, and the two pieces do not have
to come from the same person:

```python
from memrank import BenchmarkUnit, ComposedEvaluation, Document, Scorer
from memrank.benchmarks.demo import DemoBenchmark

unit = BenchmarkUnit(
    unit_id="acme", isolation_id="acme",
    documents=[Document(id="t1", user_id="acme",
                        content="Acme moved to the enterprise plan in March.")],
    queries=[{"id": "q_plan", "text": "What plan is Acme on?",
              "required_spans": ["enterprise"]}])

# My questions, memrank's scorer. No scoring is written.
mine = ComposedEvaluation(name="tickets", questions=[unit])


# memrank's questions, my scorer. No loader is written.
class MyScorer(Scorer):
    criterion_names = ("composite",)
    identity = "my-own-rules/v1"

    def score(self, unit, responses) -> dict:
        return {"composite": 1.0, "metric": "my own rules"}


theirs = ComposedEvaluation(name="demo+mine", questions=DemoBenchmark(), scorer=MyScorer())

# Both mine.
both = ComposedEvaluation(name="tickets", questions=[unit], scorer=MyScorer())
```

`questions` takes a `Benchmark`, a list of `BenchmarkUnit`s, or a callable returning them. When
it is a `Benchmark`, its own declarations come with it -- dataset version, egress safety, graph
requirement, reader-context policy, run-level rollup and judge shape -- so swapping the scorer
changes what is measured and nothing else. The scorer decides the *kind* of number
(`quality_metric`) and is named in the receipt, because two scorers are not one reproducible run.

**Both halves declare, and a disagreement is refused before any task runs.** A scorer names the
score keys it produces (`criterion_names`); a question source names the ones its own aggregation
expects. When both declare criteria and they disagree, composition raises `CriteriaMismatch` naming both
lists, rather than failing late on a `KeyError` or reducing over whichever criterion happened to
be there. The same check runs on the judge side: a scorer whose `judge_shape()` defines per-type
grading prompts is compared against the prompt keys the questions carry, to prevent applying the wrong grading prompt (`memrank/judging/shape.py`,
`BinaryJudgeShape._prompt_for`).

Declaring nothing is the default and is not a disagreement: all five registered benchmarks
declare no criteria, which is what keeps somebody else's scorer acceptable over their questions.

**What the instance route does not reach.** A benchmark passed as an object has no catalog ref, so
`memrank submit`, cloud placement and `memrank evals ls` do not know it exists. Reaching those means
giving it a name, which is [registration](#register-the-benchmark) below.

### Where the data comes from

`load()` may read whatever you like -- a file beside your script, a database, an API. For a
dataset that should be downloaded and cached, use the cache root
``memrank.benchmarks.cache_root()`` returns, or fetch it from HuggingFace lazily on first call,
and honor a `<NAME>_DATA_PATH` env var so a reader can point at a local copy.

### Register the benchmark

Add the class to `memrank/benchmarks/__init__.py`'s `REGISTRY`. It then has a ref: `memrank evals
ls` lists it, `memrank submit <target> <name>` runs it, and the cloud can resolve it.
Registration decides where the code lives and what may address it, and changes nothing about how
the evaluation is loaded or scored.

### Declare `EvalInfo` for the catalog

A registered benchmark needs `info`: `memrank evals show <name>` renders it, and a registered
benchmark without one fails the first time the catalog is asked. An instance runs without it.

```python
from memrank import Benchmark
from memrank.core import EvalInfo


class MyBenchmark(Benchmark):
    # DECLARED, never loaded -- rendering it must not construct units or touch
    # the dataset cache.
    info = EvalInfo(unit="conversation",
                    units_declared="10 conversations (dataset's own description)",
                    slices=("smoke", "mini"))
```

### Write tests beside the others

Tests for an in-tree benchmark go in `tests/benchmarks/`, beside the families already there. A
file belongs there iff its subject is a `Benchmark` implementation: loading, scoring, or a
methodology claim. Split by concern rather than by file size -- `test_<name>_loading.py`,
`test_<name>_scoring.py`, `test_<name>_methodology.py` is the shape the existing benchmarks use,
and the methodology file is the one a reviewer reads first.

A test that runs your evaluation end to end through the runner is about the run rather than the
evaluation, and goes in `tests/orchestration/`. [`tests/README.md`](../tests/README.md) states
the rule for every directory.

### Document the methodology

Per the vendor-neutral charter ([SPEC.md section 7.3](SPEC.md#73-methodology-changes-are-public)), a methodology change goes
through public proposal and comment before it merges, and does not merge without matching
documentation.

A shared evaluation needs a section in [`methodology.md`](methodology.md): what it scores, how it
is built, what its labels mean, and -- the part readers rely on -- the limits on interpreting its values. Add the axis there too when it introduces one.
