# Memrank examples

Choose a script for the task you want to learn. Each folder has a self-contained `run.py` and a
README explaining what it shows. The small local examples teach API mechanics; their results
are not evidence about vendor performance or your production workload.

## Run an example

These files are in the repository, not the installed wheel. Follow
[local development](../docs/local-development.md) to clone and set up a checkout, then run from
its root:

```bash
env -u UV_PROJECT_ENVIRONMENT uv sync --extra dev
env -u UV_PROJECT_ENVIRONMENT uv run python examples/01-first-result/run.py
```

You can also copy an individual `run.py` into a project with Memrank installed. The scripts import
Memrank without importing other example files.

<a id="compare-systems-and-versions"></a>
## Compare memory systems and their versions

Read the [memory systems comparison guide](../docs/comparing.md) for conditions, coverage and interpretation.

| Task | Example | Needs after installation |
|---|---|---|
| Compare two versions of your system | [06-new-version-vs-old](06-new-version-vs-old/README.md): two toy memory versions, changed tasks and cautions | No service or key |
| Compare with diagnostic controls | [05-against-a-baseline](05-against-a-baseline/README.md): no context, a retrieval system and full context | No service or key |
| Substitute another implementation | [07-against-a-known-engine](07-against-a-known-engine/README.md): your system versus shipped `WordOverlap`, a local toy rather than a vendor service | No service or key |

## Build and inspect a run

| Task | Example | Needs after installation |
|---|---|---|
| Check the installation | [01-first-result](01-first-result/README.md): the TFIDF/Demo quick-start pair | No service or key |
| Implement a memory | [02-your-own-system](02-your-own-system/README.md): the four lifecycle methods | No service or key |
| Express your own tasks | [03-your-own-evaluation](03-your-own-evaluation/README.md): context, expected outcomes and measures | No service or key |
| Measure saved evidence | [04-your-own-measure](04-your-own-measure/README.md): load a result and apply a new measure | No service or key |

All seven scripts above run offline with bundled or inline data. They do not download a dataset
or call a paid API. Replacing a local system with a live client changes those requirements; read
its [system page](../docs/systems/README.md) first.

[Understand results](../docs/results.md) explains values, traces, errors and saving evidence.
[More examples](more/README.md) covers the command-line path and integrations that can require a
live engine or Docker. Those have separate prerequisites and use the command line's result format.
