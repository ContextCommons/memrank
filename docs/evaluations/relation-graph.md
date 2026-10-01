# RelationGraph -- `relation_graph`

```python
from memrank.evaluations import RelationGraph

evaluation = RelationGraph()
```

Four scenarios testing whether a memory graph represents updates, entity identity and
inferred relations correctly.

## How it works

The four fixtures test structural properties that retrieval recall does not capture:

- a preference update that should supersede an earlier statement;
- two different people share a name, and a fact about one must not attach to the other;
- something is implied rather than said, alongside a distractor that would support the wrong
  inference;
- one document at once updates an earlier fact, extends another, and implies a third.

Memrank hands each fixture to the [system](../reference/system.md), then reads the graph the
system reports -- its memories and the relations between them, in one normalized shape -- and
scores the structure: did the superseded fact get marked superseded, did the collision stay
separate, was the inference drawn and the distractor left alone.

## Why it matters

A system can retrieve the correct document while retaining contradictory facts or merging two
people's identities. This evaluation checks the reported graph for those structural errors.

Its score is **structural and decided by a fixed rule**. Like `demo`'s it needs no judge; unlike it, it scores what the
system holds rather than what it retrieved.

## What it needs

No dataset download. It requires a system that reports its memory graph in the normalized format.
A system that does not is still run -- memrank cannot know before it asks -- and its
`relation_graph-score` values are `None`, each carrying the reason: no graph snapshot in the
system's responses. It is never scored as though it had reported an empty graph. (The command
line skips a system not declared graph-capable as `not_applicable` before running it.)

## References

- [evaluation](../reference/evaluation.md) -- what an evaluation is.
- [Adding a system](../systems.md) -- what a system declares to report a graph.
- [supermemory](../systems/supermemory.md) and [atomicmemory](../systems/atomicmemory.md) -- two
  shipped systems that report one.
