<a id="compare-systems-and-versions"></a>
# Compare memory systems and their versions

> **Deprecated:** this page describes memrank's older, pre-agent surface, which still ships but will be removed. To evaluate an agent, use `memrank run`; see the [README](../README.md).

Run the same evaluation on two memory systems to see which tasks each handles better, how long they
take and where they fail. You can also compare two versions of your memory system to investigate
whether your changes improved its performance. A **result** records one run;
`memrank.paired(a, b)` compares eligible measurements from two results task by task. Read the
scores alongside the recorded responses and errors to understand the differences.

## Set up the comparison

1. Choose tasks and measures that address your question. You own the evaluation's relevance and
   success criteria; [Adding an evaluation](evaluations.md) shows how to express them.
2. Run both candidates on the same evaluation and version. Keep the task content and IDs,
   clearing rule, retrieval limit, answer writer and scoring configuration comparable. Record
   deliberate differences, including the candidate versions and service configuration.
3. Check refusals, task coverage, errors and missing measurements before interpreting the gap.
4. Inspect the tasks that changed. Decide whether the measured tradeoff matters to your work.

Reusing one evaluation helps keep tasks and measures consistent. It does not control external
services, model randomness, changing data or machine load. Pairing checks the evaluation's name
and version; it does not prove that two equally labelled evaluations contain identical tasks,
or that every experimental condition was held fixed.

## Learn task-level pairing with a toy example

After [installation](install.md), this complete script runs offline with no services or keys.
It uses the synthetic `Demo` evaluation to illustrate the API, not to compare vendors or establish
performance on a real workload. `WordOverlap` ranks documents by shared words; `NoContext`
returns no documents. Both receive the same tasks.

```python
import memrank
from memrank.evaluations import Demo
from memrank.systems import NoContext, WordOverlap

evaluation = Demo()
result_a = evaluation.run(system=WordOverlap())
result_b = evaluation.run(system=NoContext())
reading = memrank.paired(result_a, result_b)

print(reading)
```

Read the `word-match` section:

- A is `WordOverlap`; B is `NoContext`. `gap` is **mean B minus mean A**.
- There are five paired tasks. A's mean is `0.800` and B's is `0.200`, a gap of `-0.600`.
  This means fewer expected span outcomes matched under B, not that an answer model was less
  correct. `WordMatch` is a retrieval proxy. One task tests that a forbidden span stays absent,
  so returning nothing can satisfy that task.
- Three tasks differ. The reading names changed task IDs in `flips` and prints up to three.
- The exact McNemar p-value is `0.25` and the reading cautions that too few tasks differ to
  characterize the gap. A p-value is not the probability that B is worse or that the gap is
  "just chance". This small example teaches how to read the output, not how to select a product.

A negative gap means B has a lower value. Whether lower is preferable depends on the measure:
lower failure rate is desirable, while higher recall is desirable. Do not interpret the sign
without the measure's definition.

To inspect what changed, this separate, complete block retrieves the first changed task's two
traces:

```python
import memrank
from memrank.evaluations import Demo
from memrank.systems import NoContext, WordOverlap

evaluation = Demo()
a = evaluation.run(system=WordOverlap())
b = evaluation.run(system=NoContext())
reading = memrank.paired(a, b)
word_match = next(m for m in reading.measures if m.measure == "word-match")

if word_match.flips:
    changed = word_match.flips[0]
    print(changed.task_id, changed.was, "->", changed.now)
    for label, result in (("A", a), ("B", b)):
        trace = result.traces_of(changed.task_id)[0]
        print(label, trace.task.prompt, trace.task.expected)
        print(trace.recalled, trace.error)
```

For every changed task, look at what was asked, what success meant, what came back and whether an
error occurred. The [result guide](results.md) explains these fields and saving the evidence.

## Choose a comparison pattern

These existing scripts demonstrate mechanics with small local systems. Run them from a checkout
as described in the [examples index](../examples/README.md).

| Question | Example | What it demonstrates |
|---|---|---|
| Did my change help? | [New version versus old](../examples/06-new-version-vs-old/README.md) | Two toy memory versions on the same tasks, with a fixed retrieval limit. |
| Does retrieval help this measure? | [Against a baseline](../examples/05-against-a-baseline/README.md) | No-context and full-context controls alongside a retrieval system. Returning all documents is a diagnostic control, not a practical strategy. |
| How do I substitute another implementation? | [Against a shipped system](../examples/07-against-a-known-engine/README.md) | A custom memory compared with local `WordOverlap`; this example calls no vendor service. |

For a live alternative, read its [system page](systems/README.md) and configure the service and
credentials it requires. For an implementation you own, use [Adding a system](systems.md).
Retain the configuration and evaluation version with any comparison you share.

## Know what is paired

`paired` compares common measure names with non-null **task-level** values, matching by task ID.
Its means use only the tasks that have a value on both sides. Missing values are omitted from the
pairing, not counted as zero. Check the pair's `tasks` count against expected coverage and inspect
each result's errors and missing values; a comparison of successful tasks can hide reliability
problems if read alone. `only_in_a` and `only_in_b` report trace IDs missing from one result, not
the measure-level omissions.

The `demo-score` is an aggregate evidence-recall value with no task ID, so it is not
paired. Neither are run-level latency summaries or `failure-rate`. Compare those values in the
individual results with their sample counts, failures and settings. An empty `reading.measures`
means there were no eligible pairs, not that the systems tied.

For binary observed values, the reading reports an exact McNemar test; for continuous values it
reports a 95% paired bootstrap interval, resampling whole task groups where groups are present.
Those describe uncertainty under their assumptions. They do not measure variability across
repeated runs, and this API does not report repeated-run standard deviation. See
[the paired reference](reference/paired.md) for the exact fields and statistical limits.

A comparison can be useful without a clear winner. Report what differed, how much evidence it
rests on, which tasks or values were missing, and what the evaluation leaves unmeasured.
