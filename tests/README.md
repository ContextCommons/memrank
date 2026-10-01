# The test tree

Tests mirror `memrank/`'s subsystem map. A directory here is named after the package it exercises,
so the question "where does my new test go?" has a mechanical answer rather than a taxonomic one:
find the subsystem the test is about, and put it in the directory of that name.

Two directories are not package names, and both earn their place: `core/` holds the package's
top-level modules, and `repo/` holds assertions about the repository tree rather than about runtime
behavior.

## Where a file goes

A rule is executable when two people reading it put a new file in the same directory.

| Directory | A file belongs here iff |
|---|---|
| `adapters/` | its subject is a shipped system's behavior against the `Memory` contract (`memrank/instrument/kinds.py`), with no live backend |
| `application/` | it drives `memrank.application.*` (submission, planning, catalogs) |
| `benchmarks/` | its subject is a `Benchmark` implementation -- the operator layer a named evaluation resolves to -- loading, scoring, or a methodology claim |
| `cli/` | it invokes a `memrank` command through Typer/`CliRunner` and asserts on the command surface |
| `composition/` | its subject is `memrank.composition`: how several evaluations compose into one |
| `core/` | its subject is a top-level module of the package: `core`, `config`, `settings`, `rate_limit`, `atomic_json`, or the shared fakes those contracts are exercised with |
| `instrument/` | its subject is one of the seven as `memrank/instrument/` defines them: system, evaluation, task, trace, measure, run, result. A test about a **measure** or a **trace** belongs here, not in `metrics/` or `provenance/` |
| `instrumentation/` | its subject is `LatencyCollector` / `TokenCollector` |
| `judging/` | its subject is `memrank.judging.*`: the client, the prompts, the shape, the budget |
| `live/` | see [the carve-out](#the-one-carve-out-live) below |
| `metrics/` | its subject is `memrank.metrics.*`: scoring, cost, ordering, headline, recall |
| `orchestration/` | it runs a benchmark end to end through `memrank.runner` / `memrank.orchestration` and asserts on the run's behavior rather than on one component's |
| `placement/` | its subject is where and how a run is placed: local, cloud, ECS, artifacts, reconciliation |
| `provenance/` | its subject is the receipt or what feeds it: engine, environment, install, config hash |
| `repo/` | it asserts a property of the repository tree -- the public/internal boundary, the projection, import weight, doc links, docs-vs-CLI agreement -- not of the package's runtime behavior |
| `runs/` | its subject is the run record, registry, status, or progress model |
| `secrets/` | its subject is `memrank.secrets.*` or credential handling |
| `targets/` | its subject is `memrank.targets.*`: the catalog, manifest, resolution, engine env |
| `term/` | its subject is `memrank.term.*`: rendering, tables, progress, the output chokepoint |
| `tracking/` | its subject is the MLflow export/import path |
| `internal/` | `publish.toml` classifies it internal, so it does not ship in the public projection |

`instrument/` and `instrumentation/` are one letter apart and are about different things.
`instrument/` is the seven words the Python interface is written in -- `memrank/instrument/`,
where `Memory` is defined and where a measure, a trace and a result have their types.
`instrumentation/` is the two collectors that gather latency and token samples during a run.
A test that asks what a measure decides goes in the first; a test that asks how a sample is
collected goes in the second.

The tree, in full:

```text
tests/
  __init__.py  conftest.py  fakes.py  withheld.py
  fixtures/                          static data, never collected

  adapters/  application/  benchmarks/  cli/  composition/  core/  instrument/
  instrumentation/  judging/  metrics/  orchestration/  placement/  provenance/
  repo/  runs/  secrets/  targets/  term/  tracking/

  live/                              the carve-out
    conformance/

  internal/                          the publish.toml prefix
```

`fixtures/` is static data and collects nothing. `conftest.py`, `fakes.py`, `withheld.py` and
`__init__.py` stay at the root of `tests/`: the root conftest is the common ancestor of every test
by construction, and the other two are imported absolutely (`from tests.fakes import FakeAdapter`,
`from tests import withheld`) from a dozen directories, so neither belongs under any one of them.

`withheld.py` is the one way a test says "this tree does not carry that builtin target". `mem0` and
`supermemory` name engine images no outsider can pull (`docs/misc/engine-images.md`), so the
published package ships neither manifest, and a test written against one skips there with the
reason stated -- the same shape `live/` uses for an absent backend, and never a weakened assertion.

## The one carve-out: `live/`

`live/` is the only directory that is not a subsystem. A file belongs there iff, with nothing
running, at least one of its tests reports **skipped** because a reachability probe or an explicit
opt-in (`MEMRANK_LIVE_TESTS`) failed -- that is, its full coverage needs a process memrank does not
start. It holds the adapter conformance suite and the live judge test.

That one prefix gives two exact lanes:

```bash
uv run pytest --ignore=tests/live   # everything that needs no service
uv run pytest tests/live            # everything that does
```

The rule is evaluated per directory for `live/conformance/`, which is the adapter contract suite as
a whole even though one of its modules is fully static.

There is no `e2e/` directory. The candidates run in-process, against fakes and `word-overlap`, with
no external service and no different runtime, so such a directory would describe no executable
dependency boundary. Those tests live in `orchestration/`, whose rule says the same thing while
naming the subsystem.

## Every directory is a package

**Add an `__init__.py` to every directory you create here, in the same commit that creates it.**

Under pytest's default `prepend` import mode, where a test module sits decides what it is called.
Inside a package it takes the dotted path from the package root -- `tests.core.test_config` --
and outside one it takes its bare basename, `test_config`, with its own directory prepended to
`sys.path`. Two modules collide only when both resolve to the *same* name, which takes two
same-named files in two *non-package* directories. One package on either side is enough to keep
them apart.

Five basenames are deliberately duplicated across the public and internal halves:
`test_config.py`, `test_settings.py`, `test_development_evidence.py`, `test_catalog.py`,
`test_registry.py`. Each has its internal half under `tests/internal/`, which is packaged the whole
way down, so every pair is already safe on that ground alone -- deleting a public destination's
`__init__.py` does not produce a collision today, and it is honest to say so.

So the rule is a convention rather than the thing holding those five apart, and it is worth keeping
for what it prevents next: it keeps the top-level module namespace free of test modules, so no
directory added later can collide with anything already here, and it gives every test module a
stable dotted name for the `from tests.<dir>.<module> import ...` helpers that a dozen files
share.

The audit is one command, and it must return only `tests/fixtures` and its children:

```bash
find tests -type d ! -path '*__pycache__*' -exec test ! -e '{}/__init__.py' \; -print
```

## Fixtures

A fixture resolves by walking up from the test, so where a file sits decides what it inherits.
The root `conftest.py` is the common ancestor of everything -- it isolates `MEMRANK_CONFIG_DIR`,
`MEMRANK_RUNS_DIR`, credentials and the repo `.env` for every test -- and beyond it the whole tree
defines six:

```text
tests/conftest.py                      the common ancestor, suite-wide
tests/placement/conftest.py            autouse: strips MEMRANK_ENGINES* from the environment
tests/secrets/conftest.py
tests/tracking/conftest.py
tests/internal/{cost_estimate,events,ws}/conftest.py
```

Moving a file into one of those directories gives it that conftest's fixtures, autouse ones
included; moving a file into any other directory gives it nothing beyond the root. Adding a new
conftest is the one change that can alter a test's behavior with no edit to the test, so add one
only when a directory genuinely needs shared setup.
