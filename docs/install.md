# Installing memrank

> **Deprecated:** this page describes memrank's older, pre-agent surface, which still ships but will be removed. To evaluate an agent, use `memrank run`; see the [README](../README.md).

This page installs Memrank's Python package into a project and checks it with a local retrieval
evaluation. Installing needs the network; the run below needs no network, API key or service.
For the comparison workflow and how to read the output, see [Start here](getting-started.md).

This page is written to be followed by a coding agent as well as by a person. An agent should do
the steps in order, stop at the first failure, and report what it saw.

## 1. Prerequisites

- **[uv](https://docs.astral.sh/uv/), or pip into a virtualenv.** `uv --version` says whether uv
  is there. If it is not: `curl -LsSf https://astral.sh/uv/install.sh | sh` on macOS or Linux.
- **Python >= 3.10.** Check
  the project's interpreter with `uv run python -V`, or `python -V` inside a virtualenv.
  Do not assume the system Python is recent enough. An older interpreter can make pip report
  that no matching distribution exists.
- **A project to install into.** If you have none, `uv init memrank-try` makes one in a new
  directory. Then enter it with `cd memrank-try` and select Python with `uv python pin 3.12`.
  With pip instead, create and activate a virtualenv using Python >= 3.10 before installing.

## 2. Install

```bash
uv add memrank                  # or, into a virtualenv you already have: pip install memrank
```

Run all remaining commands in that same project or activated virtualenv. For a repository
checkout, use [local development](local-development.md) instead.

## 3. Run the quick-start evaluation

Save this as `first_run.py`:

```python
from memrank.evaluations import Demo
from memrank.systems import TFIDF

evaluation = Demo()
result = evaluation.run(system=TFIDF())

print(result)
```

Run it with `uv run python first_run.py`, or with `python first_run.py` in the virtualenv you
installed into.

[`TFIDF`](systems/tfidf.md) is keyword search weighted by how rare each word is.
[`Demo`](evaluations/demo.md) is one synthetic scenario of five questions bundled with the package.
Its scores are retrieval proxies, not answer correctness.

<a id="4-what-done-looks-like"></a>
## 4. Verify the installation

The run prints a `system:` line naming `TFIDF`, an `evaluation:` line naming `demo` with 5
tasks, the measured values, and a `traces:` line reading `5 recorded, 0 with errors`. That is
the installation check. An agent reports those three lines and any failure it encountered.
It does not establish performance on your workload. [Read the output](getting-started.md#read-the-installation-check),
then [compare memory systems and their versions](comparing.md).

[`memrank.catalog()`](reference/catalog.md) lists every system and evaluation installed, and what
each one needs.

<a id="what-not-to-do"></a>
## Installation boundaries

- Install into the project for this tutorial. The `memrank` command is documented separately in
  [the command line](misc/command-line.md).
- Do not edit shell configuration, PATH or any file outside the project.
- If a step fails, stop and report the error. For `ModuleNotFoundError`, check that the script
  runs in the environment where you installed the package. For other errors, include the
  package version and a minimal reproduction when [asking for help](contributing.md#get-help).

## Upgrading

```bash
uv lock --upgrade-package memrank && uv sync      # or: pip install --upgrade memrank
```

This prints the installed version:

```python
import memrank

print(memrank.__version__)
```
