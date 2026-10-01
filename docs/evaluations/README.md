<a id="the-evaluations-that-ship"></a>
# Available evaluations

> **Deprecated:** this page describes memrank's older, pre-agent surface, which still ships but will be removed. To evaluate an agent, use `memrank run`; see the [README](../../README.md).

An [evaluation](../reference/evaluation.md) defines tasks, measures and when to clear system
state. Use this catalog to choose a task set and check its scoring method and requirements.

| Evaluation | Key | What it measures | What it needs |
|---|---|---|---|
| [Demo](demo.md) | `demo` | Substring retrieval proxy and evidence recall on a synthetic conversation | Bundled data; no download or API key |
| [RelationGraph](relation-graph.md) | `relation_graph` | Structural graph correctness across four memory scenarios | Bundled fixtures and a graph-capable system |
| [LoCoMo](locomo.md) | `locomo` | Answer correctness over ten dated conversations and about 1,500 scored questions | a one-time download; a judge for quality |
| [LongMemEval](longmemeval.md) | `longmemeval` | Answer correctness on 500 questions, each with its own set of chat sessions | a one-time download; a judge for quality |
| [BEAM](beam.md) | `beam` | Answer quality across ten memory abilities in long conversations | a one-time download; a judge for quality |

`memrank.catalog()` lists these evaluations at runtime. Demo and RelationGraph use bundled
data. System and answer-writer requirements apply separately.

<a id="the-standard-every-page-follows"></a>
<a id="proxy-or-judged-and-why-it-is-on-every-page"></a>
## Choose a scoring method

LoCoMo, LongMemEval and BEAM require generated answers and a model-based judge for quality
scores. Without judging, they report timings, failures and counts. The Python API requires you
to configure the answer writer and judge; the command line enables judging by default for these
evaluations.

Demo measures answer-substring and evidence retrieval proxies. RelationGraph scores graph structure. These fixed rules need no judge;
none measures generated-answer correctness.

## Using one

Pass a [system](../systems/README.md) instance to the evaluation's `run` method:

```python
from memrank.evaluations import Demo
from memrank.systems import WordOverlap

result = Demo().run(system=WordOverlap())
```

## Where to go from here

- [evaluation](../reference/evaluation.md) -- what an evaluation is.
- [Adding an evaluation](../evaluations.md) -- bringing your own tasks and clearing rule.
- [Measures](../measures.md) -- the measure contract, and measuring stored traces afterwards.
- [Supported systems](../systems/README.md) -- implementations and requirements.
- [Methodology](../methodology.md) -- what each shipped measure actually measures.
