<a id="reference-the-seven-words-one-page-each"></a>
# Python API reference

> **Deprecated:** this page describes memrank's older, pre-agent surface, which still ships but will be removed. To evaluate an agent, use `memrank run`; see the [README](../../README.md).

Use `evaluation.run(system=...)` to run evaluation tasks against an implementation and record
the results. The pages below define the objects involved, their fields and methods, and how to
use them from Python.

| Concept | What it is | Who supplies it |
|---|---|---|
| [**system**](system.md) | the implementation under evaluation: a memory, model, retriever or assistant | you or Memrank |
| [**evaluation**](evaluation.md) | a named, versioned set of tasks, measures and a rule for when to clear system state | you or Memrank; tasks and measures can come from different sources |
| [**task**](task.md) | one evaluation case: its context, prompt and expected outcome | the evaluation |
| [**trace**](trace.md) | the recorded inputs, responses, timings and errors for one task attempt | Memrank |
| [**measure**](measure.md) | a named rule that produces values from traces and declares its inputs and source of judgment | the evaluation, or you when measuring saved traces |
| [**run**](run.md) | execution of an evaluation against a system, followed by measurement of the recorded traces | you initiate it with evaluation.run(system=...) |
| [**result**](result.md) | the recorded traces and measured values, with each value identifying its measure and source of judgment | Memrank returns it to you |

Two additional functions help you compare results and discover available implementations:

| Reading | What it is |
|---|---|
| [**paired**](paired.md) | `memrank.paired(a, b)` compares common task-level values from two results of the same evaluation |
| [**catalog**](catalog.md) | `memrank.catalog()` lists the included systems and evaluations with their requirements |

## Where to go from here

- [The README](../../README.md) -- what memrank is, and a first result.
- [Adding a system](../systems.md) -- evaluate your own implementation.
- [Adding an evaluation](../evaluations.md) -- your own tasks and measures.
- [Methodology](../methodology.md) -- measurement rules and limits on interpretation.
- [`examples/`](../../examples/README.md) -- runnable scripts organized by task.
