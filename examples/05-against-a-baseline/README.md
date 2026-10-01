# 05 -- against a baseline

`uv run python examples/05-against-a-baseline/run.py`

Compare `WordOverlap` with two diagnostic controls on the same tasks. `NoContext` returns no
documents; `FullContext` returns every stored document. This example measures retrieval proxies,
not generated-answer quality, and the controls are not guaranteed score bounds.

The script prints the catalog and a paired comparison against each control: means, difference,
changed tasks and statistical cautions. Use these controls to understand what the measure
rewards before interpreting a system's score.

## Two ways to name a system, and when to use which

Both forms reach the same class, and `memrank.catalog()` prints them side by side.

- **The Python name** -- `from memrank.systems import WordOverlap`, `WordOverlap()`, and
  `from memrank.evaluations import Demo`, `Demo()`. Use it when you are writing code: your
  editor follows it to the definition, hover shows what it needs, and the keyword arguments
  are typed. Example 01 is written this way.
- **The string name** -- `memrank.system("word-overlap")`, `memrank.evaluation("demo")`. Use it
  when the name arrives as data: from a config file, from a command-line argument, from a loop
  over several systems, as here. `memrank.system("atomicmemory", base_url=...)` is also how a
  client for a running service is given its address.
