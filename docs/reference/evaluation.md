# evaluation

> **Deprecated:** this page describes memrank's older, pre-agent surface, which still ships but will be removed. To evaluate an agent, use `memrank run`; see the [README](../../README.md).

An **evaluation** is a set of questions and the rules for asking them, and it is the object you
call `run(system=...)` on.

```python
from memrank.evaluations import Demo
from memrank.systems import TFIDF

evaluation = Demo()
print(evaluation.name, evaluation.version, len(evaluation.tasks), "tasks")

result = evaluation.run(system=TFIDF())
```

[`Demo`](../evaluations/demo.md) is one synthetic scenario of five questions bundled with the
package, and needs no download and no key. [`LoCoMo`, `LongMemEval` and `BEAM`](../evaluations/README.md) are public
benchmarks memrank can download. `tickets` below is one you write yourself.

## What it is

An evaluation has a name, version and three components:

1. its **[tasks](task.md)** -- the cases to execute;
2. the **[measures](measure.md)** that score the recorded traces;
3. its **clearing rule** -- when Memrank calls the system's cleanup method. With
   `Clearing.PER_GROUP`, tasks with the same `group` share context and cleanup follows each
   group. `Clearing.PER_TASK` isolates each task; `Clearing.AT_END` shares state across the run.
   The system implementation must enforce isolation and cleanup.

The version is part of the identity. Two results are comparable only when they are of the same
evaluation at the same version, and [paired](paired.md) refuses when they are not.

## Who supplies what

Memrank ships five evaluations, each bundling its own measures, and enforces the clearing rule
during the run. You supply your own tasks, your own measures, or both -- they are separate
choices, and `memrank.evaluation("demo")` returns exactly the kind of object you would write by
hand.

## The Python names

```python
import memrank
from memrank import Clearing, Document, Evaluation, Expected, Task
from memrank.systems import WordOverlap

notes = (Document(id="t1", user_id="acme",
                  content="Acme moved to the enterprise plan in March."),)

tickets = Evaluation(
    name="tickets", version="internal@2026-09",
    tasks=(Task(id="q_plan", prompt="What plan is Acme on?", group="acme", context=notes,
                expected=Expected(answers=("enterprise",), required_spans=("enterprise",),
                                  evidence_doc_ids=("t1",))),),
    measures=(memrank.WordMatch(),),
    clearing=Clearing.PER_GROUP)

result = tickets.run(system=WordOverlap())
```

- `memrank.Evaluation` -- the class. Fields: `name`, `version`, `tasks`, `measures`,
  `clearing`, `metadata`.
- `evaluation.run(system=...)` -- the [run](run.md), returning a [result](result.md).
- `memrank.Clearing` -- `PER_TASK`, `PER_GROUP`, `AT_END`.
- `memrank.evaluations` -- the module holding what ships: `Demo`, `RelationGraph`, `LoCoMo`,
  `LongMemEval`, `BEAM`.
- `memrank.evaluation("<name>")` -- the same things by string.
- `evaluation.groups()` and `evaluation.group_of(task_id)` -- which tasks share state.

## Going deeper

- [Adding an evaluation](../evaluations.md) -- supply tasks, measures or both.
- [Available evaluations](../evaluations/README.md) -- task sets and requirements.
- [Methodology](../methodology.md) -- what the shipped evaluations measure.
- [`examples/03-your-own-evaluation/`](../../examples/03-your-own-evaluation/) -- a working one.
