# Contributing and getting help

Memrank helps people building agents find the memory that works best for them, by evaluating the
agent end to end. It aims to be a neutral evaluation tool for agent memory that the whole industry
can stand behind, built together with its ecosystem. This is the beginning of that journey, and
anything you think would make it better is welcome: ideas, issues, discussions or code.

Some places to start: a way to reach another agent stack, a new evaluation, a new baseline, better
grading, or a clearer command line and report. [Where each contribution lives](#where-each-contribution-lives)
shows where these sit in the code.

## Get help

Use the [GitHub issue tracker](https://github.com/ContextCommons/memrank/issues) for bugs,
questions about a run, and methodology disagreements. Include the Memrank version
(`memrank --version`), the command you ran, the block the run ended with, the result you expected
and what happened instead. Issues are public, so keep them to what is safe to share: remove keys
and private data before posting.

Open an issue before a large change, so we can agree on its shape first.

## Development setup

Memrank uses Python 3.12 for development (3.10 is the supported floor) and
[uv](https://docs.astral.sh/uv/) for environments and dependencies. From a clone:

```bash
uv sync --extra dev
uv run memrank --help
```

`uv run memrank ...` runs the command line from your checkout, so a change is live without
reinstalling.

Before you open a pull request, run the same checks CI runs:

```bash
uv run python -m pytest
uv run ruff check .
uv run mypy memrank
```

Tests that need a live service skip when it is absent; everything else must pass. Tests never
assert on timing, and new code fails loudly rather than falling back to a degraded mode.

### Running an evaluation while you work

`memrank run` records every run in your organisation's run history, so it needs a memrank.ai
login (`memrank auth login`) even from a checkout. A run can still cost nothing: the repository
has a model-free test agent, and `--no-judge` records answers, failures and latency without
calling a judge. From the repository root:

```bash
uv run memrank run locomo --agent tests/fixtures/agents/keyword-command.yaml --cases 2 --questions 3 --no-judge
```

`memrank evals check <file>` validates an evaluation file and summarises what a run would ask,
without running an agent or signing in.

## Where each contribution lives

### A connector or preset for your agent stack

`memrank/connect/` is how Memrank reaches an agent. An agent file (`memrank/connect/spec.py`)
picks a connector: `command` (`command.py`) runs a program per step, and `http` (`http.py`) maps
reset, feed and ask onto requests you describe. A preset (`presets.py`) is only an HTTP mapping
with the agent-specific parts left open; `openai-chat` is one. If your stack speaks a common API,
add a preset to `PRESETS` rather than a new code path, and add tests beside the existing ones in
`tests/connect/`. A new connector is the right change only when a stack cannot be described as a
command or an HTTP mapping at all.

### An evaluation

A shipped evaluation is a benchmark loader in `memrank/benchmarks/` (a `Benchmark` subclass that
loads cases and caches its dataset under `memrank.benchmarks.cache_root()`, honouring its own
data-path environment variable), registered in `memrank/benchmarks/__init__.py`.
`memrank/definitions/shipped.py` turns those loaders into the definitions `memrank run` takes, and
`memrank/service/protocol.py` lists which ones are runnable. Many evaluations do not need code at
all: an evaluation file (`memrank/definitions/file.py`, starter in
`memrank/definitions/starter.yaml`) holds cases inline, in JSONL or from a program, and can be
shared as it is. Changing how an evaluation is scored needs a matching update to its
documentation page.

### A baseline

Baselines are the agents runs are compared against. Each one is an agent file in
`memrank/agents/` (its file name is the name `--agent` takes), usually starting a small server
that speaks the OpenAI Chat Completions API. `full-context` is the example: its file is
`memrank/agents/full-context.yaml` and its server is `memrank/reference/full_context.py`, started
through `memrank agents serve` in `memrank/cli/agents.py`. A new baseline should keep each
session's memory separate, refuse requests it cannot attribute to a session, and say in its
`version` what it answers with.

### Grading fixes

The judge lives in `memrank/judging/`: the prompts in `prompts.py`, the parsing and voting in
`judge.py`, and the per-benchmark judging shape in `shape.py`. `memrank/service/grading.py` and
`memrank/service/scoring.py` turn verdicts into scores and intervals, and the graders for
evaluation files are in `memrank/definitions/graders.py`. A change to a judge prompt changes what
a score means, so bump `JUDGE_PROMPT_VERSION` in `prompts.py`, add a test under
`tests/judging/`, and explain the change in the pull request.

### The command line and the run report

Commands are in `memrank/cli/`. What a run prints and leaves behind is in `memrank/loop/`: the
ending block (`ending.py`), the score readout (`readout.py`), and the HTML report
(`report_html.py`). Keep messages actionable: every ending says what happened and the exact
command to run next. `tests/cli/` and `tests/loop/` hold the matching tests.

## Pull requests

Keep a pull request to one change, with tests for the behaviour it adds or fixes, and say in its
description what you ran. If a change affects how a score is produced, say so plainly: people
compare runs across versions, and a silent change to scoring makes those comparisons wrong.

By contributing, you agree that your contribution is licensed under
[Apache-2.0](../LICENSE), the license of this repository.
