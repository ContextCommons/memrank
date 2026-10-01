# Adding a system

> **Deprecated:** this page describes memrank's older, pre-agent surface, which still ships but will be removed. To evaluate an agent, use `memrank run`; see the [README](../README.md).

A **[system](reference/system.md)** is the implementation you put under test. First check the
[shipped systems](systems/README.md): a client may already support your engine. Otherwise,
implement the appropriate kind below, then pass an instance to the evaluation.

Pass an instance from your own project; no package contribution or catalog registration is required.

Memrank calls the lifecycle at the evaluation's boundaries, records traces, times calls and
applies [measures](reference/measure.md). Your implementation must honor isolation and clear its
own state. System-declared information, such as token usage, remains a declaration; Memrank
cannot infer it from an HTTP response time.

**Compatibility aliases.** The memory contract is defined as `Memory`, and `MemoryAdapter` and
`MemoryEngine` are deprecated aliases of that same class object, so every memory class already
written is a `Memory` and needs no edit. Write `memrank.Memory` in anything new. Existing system classes retain their old `Adapter` names as aliases;
`memrank/adapters/__init__.py` lists them.

## The four kinds, and the verbs each one requires

The base class determines the system kind. `KIND_NAMES` and `REQUIRED_VERBS` in `memrank.instrument.system`
are where this table is stated; the memory contract itself lives in `memrank.instrument.kinds`,
because it is the one kind that needs the value types in `memrank.contract`.

| Kind | Subclass | What it is for | Verbs it must supply |
|---|---|---|---|
| memory | `memrank.Memory` | stores documents, retrieves relevant ones and clears state on request | `prepare`, `ingest`, `retrieve`, `cleanup` |
| model | `memrank.Model` | completes a prompt | `complete` |
| retriever | `memrank.Retriever` | ranks an existing corpus without ingesting task context | `rank` |
| assistant | `memrank.Assistant` | responds to a list of messages in the usual `role`/`content` shape | `respond` |

`kind_of` finds the kind by walking your class's MRO for one of those four, so you declare a kind
by subclassing and in no other way. A run validates this **before calling your system** (`memrank/instrument/refusal.py`) and returns a refusal -- a result with no traces and a
stated reason -- rather than failing somewhere inside the loop. It refuses when the object is not
a `System` at all, when it subclasses `System` but none of the four kinds, and when it is a kind
whose required verb it does not supply.

Two refusals are about the fit between the system and the evaluation rather than about the class:

- an evaluation whose tasks carry **context** needs a kind that supports ingestion, which today
  is memory alone;
- a measure that reads `answered` needs a kind that answers -- model or assistant -- or an
  `answerer=` passed to the run, because a memory only recalls.

```python
import memrank


class Echo(memrank.Model):
    """A model that answers with the prompt it was given."""

    def complete(self, prompt: str) -> str:
        return prompt


result = memrank.evaluation("demo").run(system=Echo())
print(result.refused)
print(result.refusal)
```

System declarations are optional: `declared_version`, `declared_tokens`,
`declared_timings` and `declared_state` are plain methods on `System` that return `None` until
you write one, and missing information is recorded as `None`, not zero. The memory contract
keeps its own older spellings of the last three -- `token_metrics`, `declared_latency` and
`state_fingerprint` -- and memrank reads both, so an existing class needs no edit.

## 1. Run your own system: pass the instance

`evaluation.run(system=...)` takes the instance, so a system that exists only in your codebase
runs the same loop as the ones memrank ships:

```python
import memrank
from memrank import Document, Recall


class MyMemory(memrank.Memory):
    name = "myengine"           # a label on the result, not an address
    version = "0.1.0"           # your class's own version
    engine_version = "1.2.3"    # the engine you are wrapping

    def prepare(self, isolation_unit: str) -> None:
        self.held: list[Document] = []

    def ingest(self, documents: list[Document]) -> None:
        self.held.extend(documents)

    def retrieve(self, query: str, k: int, user_id: str,
                 query_timestamp=None) -> Recall:
        wanted = set(query.lower().split())
        ranked = sorted(self.held, reverse=True,
                        key=lambda held: len(wanted & set(held.content.lower().split())))
        return Recall(documents=ranked[:k])

    def cleanup(self) -> None:
        self.held = []


result = memrank.evaluation("demo").run(system=MyMemory())
print(result.values_of("demo-score")[0].value, result.system.name, result.system.kind)
```

### Why a memory has four methods

Each method separates a part of the experiment that would otherwise be hidden inside your
application. For each isolation unit selected by the evaluation's clearing rule:

1. `prepare(isolation_unit)` starts an isolated store. The toy resets its list; a client can
   select a fresh namespace so documents from another unit do not influence this one.
2. `ingest(documents)` receives the context before questions arrive. Your system may build an
   index or send the documents to an engine. Memrank times this call separately from retrieval.
3. `retrieve(query, k, user_id, query_timestamp)` returns a `Recall` for each question. Preserve
   ranking order and honor the supported user and time boundaries. The toy ranks shared words;
   it does not implement a multi-user service or temporal retrieval.
4. `cleanup()` removes the state created for that isolation unit. For a remote engine, delete
   only the namespace reserved for this evaluation, never shared application data.

The evaluation decides when to clear: per task, per group or at the end. That lets tasks which
belong to one conversation share context while independent cases start separately. See
[clearing rules](reference/evaluation.md) and [run behavior](reference/run.md). Observing that
cleanup returned does not prove that an external service erased state; test that contract in
your implementation.

Those four verbs are the whole third-party contract for the memory kind, and they are abstract:
a run refuses before touching anything when one is missing. [The translator
contract](system-contract.md) states each one in full.

`retrieve` returns a `Recall`: the documents in the order your system ranked them -- ranking affects retrieval metrics, so return at most `k` and never pad -- and `declared`, whatever your
system wants to say about the call, recorded untouched. Both halves land on the
[trace](reference/trace.md).

Memrank times ingest and retrieve at its call boundary. Do not report those timings from your
system; use optional declarations for internal timing information.

No registry or package changes are needed. Construct the evaluation with
`memrank.evaluation("demo")`, `memrank.evaluations.Demo()`, or an `Evaluation` of your own (see
[adding an evaluation](evaluations.md)).

An offline example of this pattern is
[`examples/02-your-own-system/`](../examples/02-your-own-system/README.md):

```bash
uv run python examples/02-your-own-system/run.py
```

Next, [compare this system with another](comparing.md) and [inspect its results](results.md).
The Python result records the observed system and evaluation; it does not package your engine's
build or environment. For tracked and placed runs through the command line, see
[registration and targets](#register-a-system-for-the-command-line). Those use a
separate result and provenance format.

## 2. Declare what only your system knows

Memrank measures call latency. Your system can optionally report token usage and internal timings.

**Token usage** is the one measurement only a system can make, because only it is told what a
provider billed it. Record into a `TokenCollector` and return `self.tokens.as_metrics()` from
`token_metrics()` -- or from the kind-neutral `declared_tokens()`, which memrank reads first:

```python
import memrank
from memrank.instrumentation import TokenCollector


class MyMemory(memrank.Memory):
    def __init__(self) -> None:
        self.tokens = TokenCollector()

    def prepare(self, isolation_unit: str) -> None: ...
    def ingest(self, documents) -> None: ...
    def retrieve(self, query, k, user_id, query_timestamp=None): ...
    def cleanup(self) -> None: ...

    def token_metrics(self) -> dict[str, float | None]:
        return self.tokens.as_metrics()


system = MyMemory()
system.tokens.record_call("query", input_tokens=120, output_tokens=30)
print(system.token_metrics()["tokens_per_query_mean"])
```

**Hold the collector as `self.tokens`, under that name**, even though the run never looks for it.
A tracked run at `workers > 1` builds one system per worker and pools the *collectors*, to calculate statistics over the combined samples -- and it finds them by
that attribute (`memrank/evaluation/measurement.py`). A class that overrides `token_metrics`
while keeping no collector is refused there with that reason stated.

Declaring nothing is the common case: the result then reports `None` for each token bucket, which
says nobody counted rather than claiming the system spent zero.

**Internal timing** is the second, narrower declaration: `declared_latency()` -- or the
kind-neutral `declared_timings()` -- for time only your system can see inside the hop memrank
measured around it, a translator's `engine_ms`. Return SAMPLES per bucket, in milliseconds, never
percentiles: the run puts them on the [trace](reference/trace.md) at
`trace.declared.engine_timings`, and a
tracked run pools them across the systems a `workers > 1` run builds and renders
`<bucket>_p50_ms` / `<bucket>_p95_ms` itself, so a run at any width reports the same statistic of
the same population. The conventional buckets are `ingest_engine` and `retrieve_engine`. A bucket
that would render one of memrank's own six measured keys is refused: system declarations cannot replace Memrank's measured latency.

A `memrank.instrumentation.LatencyCollector` is the shape those samples come from, and it is
yours: wrap your own calls with `collector.track("ingest_engine")`, then hand the samples back
from the declaration. A bucket you never sampled reports `None`, meaning "not reported".

## Transport and dependency neutrality

The `memrank.Memory` subclass *is* the neutral boundary. Neutrality does **not** mean "everything
over HTTP": some engines expose an HTTP API, others ship only as an in-process library or SDK,
and both are first-class.

- **memrank core depends on nothing vendor-specific.** It knows only the memory kind's four
  verbs.
- **The system owns its transport.** Wrap an HTTP API or a vendor SDK, whatever the engine
  provides.
- **Vendor SDKs are optional extras.** Where the class ships with memrank, declare them in
  `pyproject.toml` (`memrank[<engine>]`); either way, import them lazily inside the class and
  fail loud with an install hint when missing -- never make core import a vendor package.
- **Label the transport honestly.** Set `transport` to the real surface (`"http"` / `"sdk"` /
  `"in-process"`), per instance in `__init__` when your system supports more than one. Latency is
  comparable only within a transport class (see [methodology](methodology.md)), so a wrong label
  misleads readers.
- **Record the engine's internal config** -- the LLM or embedder it uses -- where you can, so
  what was actually compared is explicit rather than implied.

<a id="not-core-registration-targets-and-the-command-line"></a>
## Register a system for the command line

The examples above use the Python interface. The rest of this document describes the command
line's catalog of named systems and its vocabulary: a **target** is its word for a named system, and
an **adapter** its word for the class that drives one. [The command line](misc/command-line.md)
is where that surface is documented.

Continue when the system should be runnable **by name**: from the command
line, at `workers > 1`, in the cloud, or by other people. Registration decides where the code
lives and what may address it; it never changes what the four verbs do, and it never changes what
may be claimed about a measurement.

There are two ways to take that step.

| | **Write an in-tree adapter** | **Write a translator** |
|---|---|---|
| What you write | a Python `memrank.Memory` subclass, in this repo | a program serving [the translator contract](system-contract.md), in any language |
| Where it lives | `memrank/adapters/` | your repository |
| Needs a PR? | yes, plus entries in ~8 tables and 6 test lists | no |
| Works today on | every placement, including cloud | `--on local`, from a source checkout |
| Evidence class | artifact-backed, publishable | `development_observation`, `publishable: false` |
| Verify with | `pytest tests/live/conformance/test_adapter_contract.py` | `memrank targets verify <ref>` |

Which one to pick:

- **In-tree** when an engine should appear on the public leaderboard. Publishable evidence
  requires a pinned, reproducible artifact, which a source-launched engine is not. Sections
  [3](#3-register-the-system-in-tree) to [6](#6-submit-a-pr) are that route.
- **A translator** when your engine is not Python, or when you would rather memrank never
  imported your code. Read [the translator contract](system-contract.md) and copy
  [`examples/more/native-adapter/`](../examples/more/native-adapter/README.md).

### 3. Register the system (in-tree)

Add the class to `memrank/adapters/__init__.py`, which is still the registry's file name:

<!-- runnable: no -- the registration line names `memrank.adapters.myengine`, the module you are about to write -->
```python
from memrank.adapters.myengine import MyAdapter
REGISTRY["myengine"] = MyAdapter
```

### 4. Make it a target (stack engines)

A registered class alone is reachable from Python -- `memrank.system("myengine")` -- against a
backend you started yourself. For `memrank submit myengine ... --on local|cloud` to launch the engine, every
configuration table below needs an entry -- each fails loudly, or is covered by an enumeration test, when
missing. `hindsight` is the worked example to mirror: its image is the vendor's own
(`ghcr.io/vectorize-io/hindsight`, anonymous pull), so every file below is one you can read and
run without credentials.

| Configuration | File | What to add |
| --- | --- | --- |
| Target manifest | `memrank/targets/builtin/myengine.yaml` | name/kind/adapter/transport, `engine: {artifact, port}`, declared `components`, `compose`, `service` |
| Compose graph | `memrank/targets/builtin/myengine.compose.yaml` | the engine's containers, image pinned by public reference |
| Presets (optional) | `memrank/targets/builtin/myengine-<preset>.yaml` | `from: myengine` + the component overrides (see `hindsight-matched.yaml`) |
| Engine env tables | `memrank/targets/engine_env.py` | one entry in each of `ENGINE_ENV`, `BASE_URL_ENV`, `ENGINE_SETTINGS`, `ENGINE_COMMAND`, `READINESS`, `PROVENANCE_FIELDS` (+ `PAIRED_TOKENS`/`SECRET_ENV` when applicable). Empty dict / `None` are positive statements; absence is an error. |
| Launch requirements | `memrank/secrets/requirements.py` | `REQUIREMENTS["myengine"]` with canonical providers; extend `PROVIDER_KEY_ENV` if the engine speaks its own provider vocabulary |
| Provenance | `memrank/provenance/engine.py` | `PROVENANCE["myengine"]` (purl identity, manufacturer/supplier, pedigree). `source_repo` is for a repository a reader of the receipt can actually fetch; a source that is not public sets `source_repo: None` plus `source_visibility: "private"` and a public `source_name`, and the commit is pinned by a locationless `pkg:generic/<name>@<commit>` purl instead of by a name nobody can resolve. |
| Enumerating tests | `tests/targets/test_catalog.py` (`EXPECTED`), `tests/live/conformance/test_adapter_contract.py` (`_backend_url`), `tests/placement/test_taskdef_render.py`, `tests/placement/test_local_placement.py`, `tests/provenance/test_engine_provenance.py`, `tests/secrets/test_requirements.py` | add the new name to each pinned list |

An engine setting omitted from the manifest fails rendering
(`component_env`): declare every supported setting explicitly, even "the built-in default" -- an engine whose
deterministic fallback extractor is the default still has to say so, or the receipt cannot report
what ran.

#### Credentials: let the target say, don't extend the table

`REQUIREMENTS` derives an engine's credentials from its component *providers* -- right for a
vendor (`llm: {provider: anthropic}` means `ANTHROPIC_API_KEY`), and only that. It cannot express
a credential that is not one key per vendor. A target states its own instead:

```yaml
# names the engine reads directly
secrets: [LLM_API_KEY, EMBEDDING_API_KEY]

# a rename: what memrank resolves on the left, what the engine reads on the right
secrets:
  ANTHROPIC_API_KEY: HINDSIGHT_API_LLM_API_KEY

# anything at all -- memrank never interprets these
secrets: [MYENGINE_USER, MYENGINE_PASSWORD, MYENGINE_TENANT]
```

Each credential is resolved in priority order -- process environment, then org
secrets, then the encrypted wallet -- and injected under the variable the target names. Memrank does not interpret the credential names: an engine authenticating with a
user and a password, a client id and a tenant, or a token under a name nobody anticipated needs
no change to memrank at all.

Declared secrets are **added to** what the providers imply, so a target can use both. Reach for
`PROVIDER_KEY_ENV` only when adding a genuine vendor that many engines will name.

#### Named native development target

Keep engine repositories independent of memrank. To evaluate an unpublished checkout *by name*,
place one complete, manually authored target in
`${MEMRANK_CONFIG_DIR:-~/.config/memrank}/targets/`:

```yaml
schema_version: 1
name: myengine:dev
kind: stack
interface:
  adapter: myengine
  transport: http
binding:
  kind: source
  root: /absolute/path/to/myengine
launch:
  command: "cargo run -p myengine-server -- --bind 127.0.0.1:{port}"
  requires: [Cargo.toml]
network:
  port: 8080
  readiness: {path: /health}
components:
  llm: {provider: regex}
```

The command is split into argv and executed directly; memrank adds no implicit shell. Use an
explicit `sh -lc '...'` only when shell behavior is genuinely required. `{port}` expands to the
target's declared `network.port`. The optional `requires` paths give an early wrong-checkout
error. The developer then runs:

```bash
memrank submit myengine:dev demo --on none
```

Endpoint operations still belong in the named adapter. The target owns launch and readiness, and
the engine repository needs no memrank descriptor, Dockerfile or publishing workflow. There is no
target-creation command: author and review the central YAML directly.

This form requires an adapter that already knows the engine's wire protocol -- it is how a *fork*
of a known engine is evaluated. For an engine memrank has never seen, use `adapter: native` and
point `launch.command` at a translator instead; see
[system-contract.md section 9](system-contract.md#9-declaring-a-target).

### 5. Run the conformance suite

```bash
env -u UV_PROJECT_ENVIRONMENT uv run python -m pytest tests/live/conformance/test_adapter_contract.py -k myengine
```

Static checks -- class attributes, abstract method coverage, metric key shape -- run
unconditionally. Live smoke tests skip cleanly when your engine is not reachable.

The conformance suite is what a shared system is held to. For a system you only pass as an
instance, `memrank targets verify` and your own tests are the equivalent, and the most
consequential check either one runs is that state does not leak between groups of tasks.

#### Write tests beside the others

Anything you assert about *your* system's own behavior goes in `tests/adapters/`, one file per
system, named `test_<name>_adapter.py` alongside the ones already there. That directory is for
behavior that needs no live backend -- request shapes, partitioning, config resolution, what the
system does with a refusal. Anything that skips when your engine is not running belongs in
`tests/live/` instead, which is the one directory organized by what a test *needs* rather than
what it is about. [`tests/README.md`](../tests/README.md) states the rule for every directory.

### 6. Submit a PR

Per the vendor-neutral charter, AtomicStrata commits to reviewing valid PRs within 7 days.
Include the class under `memrank/adapters/<name>.py`, its registry entry, tests under
`tests/adapters/`, documentation for any new env vars, and instructions for starting the
backend locally.

The first seven fields of `AdapterRegistration` have no defaults on purpose. Each is read where
absence is either a loud failure far from its cause or -- for `provenance` -- a silent one:
without that row `build_provenance` returns `{}`, and the receipt loses not only the engine's
purl but the workspace pins (commit, dirty flag, working-tree delta) a source-bound target exists
to record.

**What this route does not confer.** A module found on `sys.path` has no version and no
resolvable origin, so registering through it earns no reproducibility claim of its own. The run's
evidence class still derives from how its *artifact* was bound, exactly as for a built-in --
which for a source-bound target means `development_observation`, `publishable: false`.
Registration decides where the code lives, never what may be claimed about the measurement.
