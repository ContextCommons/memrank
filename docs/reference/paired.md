# paired

> **Deprecated:** this page describes memrank's older, pre-agent surface, which still ships but will be removed. To evaluate an agent, use `memrank run`; see the [README](../../README.md).

A **paired** reading puts two [results](result.md) of the same [evaluation](evaluation.md) side
by side and reports differences and statistical summaries for common task-level values. It does
not choose a winner. For a walkthrough, start with [Compare memory systems and their versions](../comparing.md).

```python
import memrank
from memrank.evaluations import Demo
from memrank.systems import NoContext, WordOverlap

evaluation = Demo()
reading = memrank.paired(evaluation.run(system=WordOverlap()),
                         evaluation.run(system=NoContext()))

print(reading.system_a, "vs", reading.system_b, "on", reading.evaluation)
print(reading)
```

## What it is

`memrank.paired` reads two existing results and returns a `Paired` comparison object without
rerunning either system.

It **refuses** a refused run or results with different evaluation names or versions, then pairs
non-null values by task ID and [measure](measure.md) name. Equal names and versions do not prove
equal task content or experimental conditions; the caller must keep those consistent. It reports:

- the mean of each side and the gap between them;
- the tasks whose values changed, named in `flips`;
- an exact two-sided McNemar p-value when all paired values are 0 or 1, otherwise a 95%
  cluster-resampled paired bootstrap interval for the mean difference;
- a caution with fewer than ten discordant tasks.

The printed comparison reminds readers to interpret the gap in the context of the measure
and evaluation.

## Who supplies what

You supply two results of the same evaluation at the same version, and the purpose and conditions
of the comparison. Memrank supplies the compatibility check, task pairing, statistical summaries,
changed task IDs and caution.

## Coverage and interpretation

Each measure's means use only task IDs with non-null values on both sides. Run-scope or
group-only values without a task ID are omitted, including `demo-score`, latency summaries and
`failure-rate`. Check errors and missing values in the original results. An empty `measures`
tuple means no measure had eligible pairs, not that the systems tied.

`gap` is `mean_b - mean_a`. A positive gap means a larger value under B, which is only an
improvement when larger is preferable for that measure. `flips` holds all changed task values;
the printed reading lists up to three.

For binary values, McNemar's exact test evaluates the imbalance between A-only and B-only
successes under equal marginal success rates. Its p-value is not the probability that either
system is better. This implementation does not cluster the binary test by group; correlated
tasks can undermine its independence assumption.

For continuous values, the bootstrap resamples whole groups, or individual task pairs when
there is no group. Its percentile interval describes the observed mean difference under that
resampling scheme. Very few independent groups limit what it can establish. Neither statistic
reports variation across repeated runs or repeated-run standard deviation.

## The Python names

```python
import memrank
from memrank.evaluations import Demo
from memrank.systems import NoContext, WordOverlap

evaluation = Demo()
reading = memrank.paired(evaluation.run(system=WordOverlap()),
                         evaluation.run(system=NoContext()))

print(reading.evaluation, reading.version)
print("measures compared:", sorted(m.measure for m in reading.measures))
print("only in A:", reading.only_in_a, "| only in B:", reading.only_in_b)
```

- `memrank.paired(a, b)` -- the reading. `resamples=` and `seed=` control the bootstrap and
  make it deterministic.
- `memrank.Paired` -- what comes back: `evaluation`, `version`, `system_a`, `system_b`,
  `only_in_a`, `only_in_b`, `measures`.
- `only_in_a` / `only_in_b` -- trace IDs present on one side only. These do not count missing
  values within a measure.
- Each entry in `measures` carries `measure`, `kind`, `tasks`, `mean_a`, `mean_b`, `gap`,
  `discordant`, `flips` and `caution`. Binary entries also carry `both`, `neither`, `only_a`,
  `only_b` and `p_value`; continuous entries carry `ci_low`, `ci_high`, `resamples` and `seed`.

## Going deeper

- [Understand results](../results.md) -- errors, missing measurements and stored traces.
- [Methodology](../methodology.md) -- controls, budgets and limits on claims.
- [Compare memory systems and their versions](../comparing.md) -- a complete example and links to version
  and baseline comparison scripts.
