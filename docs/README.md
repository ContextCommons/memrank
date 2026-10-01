---
status: active
last_reviewed: 2026-09-24
---

# Documentation

> **Deprecated:** this page describes memrank's older, pre-agent surface, which still ships but will be removed. To evaluate an agent, use `memrank run`; see the [README](../README.md).

Evaluate your agent on your tasks.

Memrank evaluates agents: it runs evaluation tasks and reports scores and timings, with recorded
responses and errors to inspect. Use these results to understand how your agent performs, compare
it with an earlier version, an alternative or a baseline, and investigate differences on
individual tasks.

<a id="the-entry-path"></a>
## Start a task

| I want to... | Start here |
|---|---|
| Try Memrank and understand the output | [Start here](getting-started.md): installation check, result interpretation and next steps |
| Install or upgrade the package | [Installing Memrank](install.md): project-local setup and an offline check |
| Compare memory systems and their versions | [Memory systems comparison guide](comparing.md): conditions, task-level pairing, differences and limitations |
| Connect my implementation | [Adding a system](systems.md): use a shipped client or implement the lifecycle |
| Use my own tasks and success criteria | [Adding an evaluation](evaluations.md): express your evaluation without registering it |
| Read, inspect or save a result | [Understand results](results.md): meanings, errors, missing values and stored traces |
| Run a complete script | [Runnable examples](../examples/README.md): small offline examples by task |

These guides use the Python interface. A **system** is the implementation you test; an **evaluation** supplies
tasks and measurement rules. The detailed definitions are available when you need them in the
[reference](reference/README.md).

## Understand the measurements

- [Understand results](results.md) explains what a value says, where it came from and what it
  cannot establish.
- [Methodology](methodology.md) describes scoring, context budgets, controls and evidence classes.
- [Measures](measures.md) explains how to implement a scoring rule or apply one to saved traces.
- [System lifecycle](systems.md#why-a-memory-has-four-methods) explains why Memrank separates
  preparation, ingestion, retrieval and cleanup.

## Reference

| Look up | Page |
|---|---|
| Exact Python types and methods | [Python reference](reference/README.md) |
| Supported systems and their requirements | [Systems catalog](systems/README.md) |
| Available evaluations and what each measures | [Evaluations catalog](evaluations/README.md) |
| How two results are paired | [Paired reference](reference/paired.md) |
| The HTTP protocol for a memory system in any language | [System contract](system-contract.md) |
| Measurement and governance commitments | [Specification](SPEC.md) |

<a id="not-core-the-command-line-and-the-catalog"></a>
## Command-line workflows

The command line provides tracked, named and placed runs. It uses its own vocabulary and result
format; follow its guide when using that interface rather than mixing its examples with the
Python result API.

| Task | Page |
|---|---|
| Submit, watch and inspect a tracked run | [The command line](misc/command-line.md) |
| Check container availability and prerequisites | [Engine images](misc/engine-images.md) |

## Working on memrank itself

[Contributing and getting help](contributing.md) is the contributor entry point.
[Local development](local-development.md) covers checkout setup and checks;
[the test tree](../tests/README.md) explains where tests belong.
