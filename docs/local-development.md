# Local development

Set up a checkout to contribute to Memrank and run its checks. To use the released package,
follow [Installing Memrank](install.md).

memrank uses Python 3.12 locally and [uv](https://docs.astral.sh/uv/) for interpreter, environment
and dependency management. Run every command from the repository root.

## Set up the environment

```bash
git clone https://github.com/ContextCommons/memrank
cd memrank
env -u UV_PROJECT_ENVIRONMENT uv sync --extra dev
```

The repository is public, so cloning needs no credential. To use this checkout from a project of
your own instead of the released package, add it from that project:
`uv add --editable ../memrank`.

`uv sync --locked --extra dev` installs the exact versions in `uv.lock` and fails rather than
resolving anything fresh. Use it when you want this repository's resolution rather than today's:
reproducing a published result, or bisecting a failure that may be a dependency's.

`.python-version` pins the local interpreter to 3.12 and uv installs it if the machine has none.
**Python 3.10 is the supported floor** -- `pyproject.toml` declares `requires-python = ">=3.10"`
and mypy targets 3.10 -- so using a newer language feature requires an explicit compatibility change.

Optional extras, each added to the same environment:

```bash
uv sync --extra dev --extra mem0          # the Mem0 SDK, for the SDK-backed `Mem0` system
uv sync --extra dev --extra mcp           # the MCP server (`python -m memrank.mcp_server`)
```

## Check the checkout works

This is the check to run after any change to the run loop.
[`WordOverlap`](systems/word-overlap.md) and [`Demo`](evaluations/demo.md) ship with the package
and between them need no engine, no network and no key:

```python
from memrank.evaluations import Demo
from memrank.systems import WordOverlap

result = Demo().run(system=WordOverlap())
print(result)
```

Put that in a file and run it with `uv run python <file>`, which runs it inside the synced
environment and so always against the working tree. It prints the system and evaluation it ran,
one line per value with the measure that produced it and who decided it, and the trace count --
`5 recorded, 0 with errors`. The four `latency.*` values are memrank's own clock and differ on
every machine and every run; nothing else in the output does.
[result](reference/result.md) is what to read out of `result.values` and `result.traces`.

<a id="not-core-the-command-line"></a>
## Command-line development

The examples above use Python. The [command line](misc/command-line.md) provides tracked and
placed runs; what is particular to a checkout is here.

### Run it from the checkout

```bash
uv run memrank --help
uv run memrank targets ls
uv run memrank evals ls
```

`uv run memrank` runs the console script inside the synced environment, so it always reflects the
working tree. `uv run python -m memrank.runner --help` is equivalent and is what to reach for when
you need to be certain no installed copy is shadowing the checkout.

To get `memrank` on your PATH pointing at the working tree instead:

```bash
uv tool install --editable .
```

Code edits then apply immediately; changing *dependencies* needs a re-install.

### Run one through it

The same synthetic pair as above, submitted through the command line rather than called -- which
is where the command line's own older words apply, `target` for the system and `eval` for the
evaluation:

```bash
uv run python -m memrank.runner submit word-overlap demo --on none
```

Always choose placement explicitly for a local check. A bare submission can use a configured
cloud default and start a paid run. The Python check above has no placement or submission step.

For live engine runs, start the target backend first and set its URL if it differs from the
default:

```bash
export ATOMICMEMORY_API_URL=http://localhost:3070
export MEM0_HTTP_URL=http://localhost:8888
export HINDSIGHT_API_URL=http://localhost:8888
export SUPERMEMORY_BASE_URL=http://localhost:6767

uv run memrank submit atomicmemory locomo:smoke --on none
```

`--on local` provisions a disposable, isolated stack per run from the target's manifest instead,
which needs Docker. `memrank targets show <ref>` prints the composition and marks ✔/✘ per secret
before anything is spent.

Judged runs send evaluation content to Anthropic:

```bash
export ANTHROPIC_API_KEY=...
uv run memrank submit atomicmemory beam:100k-smoke --judge --on none
```

Datasets download on first use and cache under `MEMRANK_CACHE_DIR`. Each loader also honours a
local path override -- `LOCOMO_DATA_PATH`, `BEAM_DATA_PATH`,
`LONGMEMEVAL_DATA_PATH`, `DEMO_DATA_PATH` -- so you can point one at a copy you already have.

Run output lands under `results/`, which is gitignored.

## Tests and checks

For runtime changes, run the repository's applicable quality checks:

```bash
env -u UV_PROJECT_ENVIRONMENT uv run python -m pytest
env -u UV_PROJECT_ENVIRONMENT uv run ruff check .
env -u UV_PROJECT_ENVIRONMENT uv run mypy memrank
```

Tests that need a live memory engine skip when none is running; the static contract suite still
has to pass. During iteration, run the focused suite for what you touched:

| You changed | Run |
|---|---|
| anything under `memrank/adapters/`, or its `REGISTRY` | `env -u UV_PROJECT_ENVIRONMENT uv run python -m pytest tests/live/conformance/test_adapter_contract.py` |
| latency or token collection | `env -u UV_PROJECT_ENVIRONMENT uv run python -m pytest tests/instrumentation/` |
| the CLI or the runner | `env -u UV_PROJECT_ENVIRONMENT uv run python -m pytest tests/cli/test_runner_help.py`, plus a read-only command by hand |
| judging | `env -u UV_PROJECT_ENVIRONMENT uv run python -m pytest tests/judging/` |
| documentation only | the relevant documentation guards below, plus link, diff and snippet verification |
| a core contract in `memrank/core.py` | everything |

Use the worktree's own environment: `UV_PROJECT_ENVIRONMENT` can otherwise redirect uv to a
shared environment. After running checks, confirm the import points inside your checkout with
`env -u UV_PROJECT_ENVIRONMENT uv run python -c "import memrank; print(memrank.__file__)"`.
Invoke pytest as `python -m pytest` so a missing development dependency cannot silently select
a different pytest executable from PATH.

The documentation guards in `tests/repo/` check syntax, examples and factual claims:

- **Form.** Every relative link and anchor resolves (`test_doc_links.py`,
  `test_doc_anchors_resolve.py`, `test_readme_links_are_absolute.py`); every Python block runs
  (`test_runnable_blocks.py`); no document teaches a retired command, flag
  (`test_docs_teach_the_current_cli.py`) or Python name (`test_python_vocabulary.py`); and the
  entry surfaces carry the one-line definition (`test_entry_sentence.py`).
- **Claims.** A `console` fence marked `<!-- output: exact -->` is what its block prints
  (`test_console_fences_match_output.py`); a count or list of what ships is the registry's
  (`test_prose_matches_the_catalog.py`); the quick-start pair is
  `memrank.instrument.catalog.QUICK_START` wherever it is named
  (`test_the_quick_start_is_named_once.py`); a `memrank <verb>` named in prose exists
  (`test_prose_commands_exist.py`); and a documented service address is the code's default
  (`test_documented_defaults_match_the_code.py`).

The claim guards read fixed shapes of sentence, so they do not replace reading. When a change
demotes a name, changes what ships or changes a method, search the published tree for the old
fact and read every hit -- not only the pages the change is about.

<a id="where-things-live"></a>
## Repository layout

```
memrank/
├── instrument/ ........ the seven: system, evaluation, task, trace, measure, run, result
├── core.py ............ the contracts: Document, AdapterResponse, BenchmarkUnit,
│                        MemoryAdapter, Benchmark
├── runner.py .......... the Typer CLI's entry point
├── runs/ .............. a run's life on disk: registry, status, artifacts
├── cli/ ............... one module per CLI noun (auth, runs, secrets, targets, ...)
├── adapters/ .......... memory-engine adapters and the adapter REGISTRY
├── benchmarks/ ........ dataset loaders, scorers, and the benchmark REGISTRY
├── evaluation/ ........ the pure evaluation loop
├── orchestration/ ..... a run's lifecycle around that loop
├── instrumentation/ ... latency and token collectors
├── judging/ ........... LLM-as-judge: scorer, client, versioned prompts
├── metrics/ ........... judge-free scoring arithmetic
├── provenance/ ........ reproducibility receipts
├── targets/ ........... what is under test -- the named compositions
├── placement/ ......... where it runs -- none / local / cloud
└── term/ .............. terminal output, formatting, progress
```

Each package's `__init__.py` says what belongs in it and what does not. Read that before adding a
module.

`tests/live/conformance/test_adapter_contract.py` is the suite every entry in
`memrank/adapters/`'s `REGISTRY` must satisfy.
`publish.toml` classifies every path public or internal, and `tests/repo/test_public_boundary.py`
checks that the published tree respects that classification.

<a id="adding-something"></a>
## Add an integration

- **A system of your own** -- [adding a system](systems.md): pass the instance as
  `evaluation.run(system=...)`, and register it only when it needs a name.
- **An evaluation of your own** -- [adding an evaluation](evaluations.md), the same way.
- **A system, without writing Python at all** -- write a translator that speaks the
  [system contract](system-contract.md) over HTTP, in any language.
  [`examples/more/native-adapter/`](../examples/more/native-adapter/README.md) is a working one.

A change that affects how anything is scored needs a matching change to
[methodology.md](methodology.md). Document the scoring rule so readers can interpret and reproduce the measurement.

## Why this repository carries the program that produces it

`tools/` contains the public-tree projection utility. `publish.toml` classifies every path in the source repository as
public or internal, and `python -m tools.project --out <dir> --rev <sha>` writes the public tree
from a committed revision. The rules and projection utility are public, so readers can inspect how the tree is produced.
`tests/repo/test_public_boundary.py` and `tests/repo/test_projection.py` check those rules in
the published tree.
