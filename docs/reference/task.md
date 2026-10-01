# task

> **Deprecated:** this page describes memrank's older, pre-agent surface, which still ships but will be removed. To evaluate an agent, use `memrank run`; see the [README](../../README.md).

A **task** is one question, the documents the system should have been given before it is asked,
and what a right answer looks like. `q_job`, the first task of
[`demo`](../evaluations/demo.md), asks what Alex does for a living.

```python
from memrank.evaluations import Demo

task = Demo().tasks[0]
print(task.id, "|", task.prompt)
print("group:", task.group, "| context documents:", len(task.context))
print("expected answers:", task.expected.answers)
```

## What it is

The main task fields are:

- **`prompt`** -- the question or instruction sent to the system.
- **`context`** -- the documents to give it *before* the prompt. Under `Clearing.PER_GROUP`, Memrank ingests a group's documents once and calls cleanup
  between groups. Other clearing rules change that boundary; see [evaluation](evaluation.md).
- **`group`** -- the identifier used to group tasks that share state.
- **`expected`** -- `answers` are acceptable answers, `required_spans` must appear,
  `forbidden_spans` must not, `evidence_doc_ids` names the documents that hold the answer, and
  `rubric` is prose for a judge to apply.

`expected` describes the desired outcome. A [measure](measure.md) applies a scoring rule to
that expectation and the recorded response; every value identifies the measure that produced it.

A task with `polarity="negative"` is one whose right answer is a refusal. `q_allergy_neg` asks
whether Alex is allergic to peanuts, and "allergic to peanuts" is its forbidden span.

## Who supplies what

Memrank supplies the tasks of every evaluation that ships, gives the context, puts the prompt,
and records exactly one [trace](trace.md) per task per attempt, including when the task fails.
You supply task data for your own evaluation and choose its scoring measures separately.

## The Python names

```python
from memrank import Document, Expected, Task

task = Task(id="q_plan", prompt="What plan is Acme on?", group="acme",
            context=(Document(id="t1", user_id="acme",
                              content="Acme moved to the enterprise plan in March."),),
            expected=Expected(answers=("enterprise",), required_spans=("enterprise",),
                              evidence_doc_ids=("t1",)))

print(task.id, task.expected.required_spans, task.expected.polarity)
```

- `memrank.Task` -- fields `id`, `prompt`, `expected`, `group`, `context`, `category`,
  `metadata`.
- `memrank.Expected` -- fields `answers`, `required_spans`, `forbidden_spans`,
  `evidence_doc_ids`, `rubric`, `polarity`.
- `memrank.Document` -- one piece of context: `id`, `content`, `user_id`, `timestamp`,
  `context`, `messages`, `metadata`.

## Going deeper

- [evaluation](evaluation.md) -- the bundle a task belongs to, and its clearing rule.
- [trace](trace.md) -- what is recorded when a task runs.
- [Adding an evaluation](../evaluations.md) -- writing tasks of your own.
