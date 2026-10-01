# The memrank system contract, v1

> **Deprecated:** this page describes memrank's older, pre-agent surface, which still ships but will be removed. To evaluate an agent, use `memrank run`; see the [README](../README.md).

Implement this HTTP contract to connect a memory engine to Memrank through a **translator**.
The translator receives evaluation requests and calls your engine. It can run outside the
Memrank repository and be written in any language.

A reference implementation is in
[`examples/more/native-adapter/`](../examples/more/native-adapter/). It stores documents in a
Python dictionary and demonstrates the protocol.

On the wire, `adapter` identifies the translator and `engine` identifies the memory engine
behind it. The Python `Native` system is the client that calls the translator.

[Adding a system](systems.md) says when to write a translator rather than a class in this tree,
and [methodology.md](methodology.md) has the comparability rules a translator is subject to.

---

## 1. What a translator is, and why the shapes are fixed

memrank talks to a system as an HTTP service. A translator is an HTTP service too, so memrank
cannot tell the difference -- it provisions it, waits for readiness, drives the four lifecycle
operations, and tears it down using exactly the machinery that runs a system memrank ships.

The translator implements the engine-specific integration: call an SDK, open a socket, spawn a
subprocess, render documents into whatever shape your system wants, page through results,
retry. Memrank does not implement those operations for your engine.

Use the request and response schemas below so Memrank can record and compare responses
consistently. Missing required fields are errors; translators cannot substitute their own schemas.

## 2. Transport and lifecycle

A translator serves HTTP on a port memrank tells it to bind, and implements five endpoints under
`/memrank/v1/`. memrank drives them in this order, once per **group** -- a set of
[tasks](reference/task.md) that share system state, which the wire names with the
`isolation_unit` field:

```
GET  /memrank/v1/describe        once, at startup -- also the readiness probe
POST /memrank/v1/prepare         per group
POST /memrank/v1/ingest          per group, possibly several times
POST /memrank/v1/retrieve        per task in the group
POST /memrank/v1/cleanup         per group
```

All request and response bodies are `application/json`, UTF-8.

**Readiness.** memrank polls `GET /memrank/v1/describe` until it answers with a status below 500,
then starts the [run](reference/run.md). Do not answer before your system can actually serve
traffic: "ready" means "can answer the task memrank is about to put", not "the port is open". A
translator that reports ready too early turns a start-up failure into a measurement.

## 3. The `Document` shape

`Document` is memrank's canonical shape for content, used in both directions. It mirrors
`memrank.Document`:

```json
{
  "id": "conv-42-turn-7",
  "content": "The full text payload.",
  "user_id": "u_1234",
  "timestamp": "2026-03-11T09:15:00Z",
  "context": "optional free-form scene/entity context",
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "metadata": {"doc_id": "conv-42-turn-7"}
}
```

| Field | Type | Notes |
|---|---|---|
| `id` | string | Required. Stable identifier. |
| `content` | string | Required. The canonical text payload. |
| `user_id` | string \| null | The isolation scope for this document. |
| `timestamp` | string \| null | ISO-8601. **When the content happened**, not when it was ingested. |
| `context` | string \| null | Optional free-form context. |
| `messages` | array \| null | Present when the source has multi-turn structure. |
| `metadata` | object | Always present, possibly empty. |

**On `timestamp`.** If your system supports temporal reasoning, pass this through. A system that
ignores it and stamps ingestion time instead dates every memory to the moment of the run, which
silently destroys every temporal task in the [evaluation](reference/evaluation.md).

**On `messages` versus `content`.** They are the same content in two forms: `content` is the
canonical text, `messages` the structured form when one exists. Use whichever your system takes,
and render one into the other yourself.

## 4. The endpoints

### `GET /memrank/v1/describe`

Identity and capabilities. Called once at startup, and used as the readiness probe.

```json
{
  "contract_version": "v1",
  "adapter": {"name": "myengine-translator", "version": "0.1.0"},
  "engine": {"name": "myengine", "version": "1.4.2"},
  "components": {
    "llm": {"provider": "anthropic", "model": "claude-sonnet-4-5"},
    "embedder": {"provider": "voyage", "model": "voyage-4-large", "dims": 1024}
  },
  "capabilities": {
    "graph_snapshot": false,
    "context_budget": "matched"
  }
}
```

| Field | Required | Notes |
|---|---|---|
| `contract_version` | yes | Must be `"v1"`. memrank refuses a version it does not implement. |
| `adapter.name` / `.version` | yes | Your translator's own identity, recorded in the receipt. |
| `engine.name` / `.version` | yes | The system under test, under the wire's own older field name. Use `"unknown"` only if you genuinely cannot determine it. |
| `components.llm` | yes | `{provider, model}`, or `null` if the system uses no LLM. |
| `components.embedder` | yes | `{provider, model, dims}`, or `null` if the system has no embedder. |
| `capabilities.graph_snapshot` | yes | Whether `retrieve` returns `raw.graph_snapshot`. |
| `capabilities.context_budget` | yes | `"matched"` for memory engines. See section 6. |

**Report the configured components.** For a system memrank ships, memrank injects component
configuration through env vars it knows by name, and the manifest merely *asserts* what was
configured. It cannot do that for yours, because it does not know which variables your process
reads. Configure the engine in your launch command and report its components here. Memrank records
that declaration as `verified: "engine"`; this label identifies the report's source.

A `null` component means "this system has no such part" -- a positive statement. Do not use `null`
to mean "I did not check": report the real value or fail to start.

### `POST /memrank/v1/prepare`

Begin a fresh group. **Nothing ingested under a previous `isolation_unit` may be visible
to this one.** How you achieve that is yours -- a namespace, a fresh collection, a tenant id, a
wiped directory.

```json
// request
{"isolation_unit": "locomo-conv-17"}

// response
{}
```

State from another group can contaminate scores. Test that each group accesses only its own
documents.

### `POST /memrank/v1/ingest`

Load documents into the current group. May be called more than once per group.

```json
// request
{"documents": [ {Document}, ... ]}

// response
{"usage": {"total_tokens": 4210}, "engine_ms": 8123.4}
```

| Field | Required | Notes |
|---|---|---|
| `usage.total_tokens` | no | Omit the whole `usage` object if you cannot measure it. See section 5. |
| `engine_ms` | no | Milliseconds your system spent, excluding your own overhead. See section 7. |

### `POST /memrank/v1/retrieve`

Answer one task's prompt against the current group.

```json
// request
{
  "query": "Where did we agree to meet?",
  "k": 10,
  "user_id": "u_1234",
  "query_timestamp": "2026-03-14T17:02:00Z"
}

// response
{
  "documents": [ {Document}, ... ],
  "raw": { ... },
  "usage": {"total_tokens": 812},
  "engine_ms": 143.2
}
```

| Field | Required | Notes |
|---|---|---|
| `documents` | yes | **Ranked, best first.** Memrank preserves this ranking for scoring. |
| `raw` | yes | Your system's untouched response. It lands on the [trace](reference/trace.md) as what the system declared, and is never read as a measurement. Any JSON object. |
| `usage` | no | As above. |
| `engine_ms` | no | As above. |

`query_timestamp` may be `null`. When present it is the moment the task is being put, which
matters for a system that reasons about time.

**On `k`.** It is a request for the top *k*, and most systems have a matching parameter. If yours
does not model result counts at all -- some pack a token budget instead -- return what it returns and
say so in your README. Do not pad the list to reach `k`, and do not invent scores. memrank caps
every system at the same `--token-budget` subject to the evaluation's context policy; see section 6.

Put a relevance figure, if you have one, in each returned document's `metadata.score` -- report
what your system computed, never invent one.

### `POST /memrank/v1/cleanup`

End the current group and release its state.

```json
// request
{}

// response
{}
```

Called even when a group fails -- a failed task is a trace with a reason, and the run does not stop
on it. Make it safe to call twice, and safe to call before any `prepare`.

### What each endpoint puts on the trace

memrank keeps exactly one [trace](reference/trace.md) per task per attempt, including on failure,
and every [measure](reference/measure.md) reads traces rather than your system. So what you return
is not consumed and discarded -- it is stored, typed, under a field whose name says what kind of
claim it is. Which endpoint fills which field:

| What you return | Where it lands | Read as |
|---|---|---|
| `describe`'s `engine.version` | `trace.declared.version` | what the system said about itself |
| `describe`'s `components` | the run record, marked `verified: "engine"` | not a trace field; it is the run's provenance |
| `describe`'s `capabilities.graph_snapshot` | whether memrank asks you for `raw.graph_snapshot` at all | a capability, announced once |
| `prepare`'s `isolation_unit` | `trace.group` | which group's state was in force |
| the documents memrank hands `ingest` | `trace.given` -- their ids and how many | what the system was given before the prompt |
| `retrieve`'s `documents`, **in your order** | `trace.recalled.documents` | the rank. Nothing re-sorts it |
| `retrieve`'s `raw` | `trace.recalled.declared` | the system's own words, kept for forensics, never a measurement |
| `usage.total_tokens` from either | `trace.declared.tokens`, in the `ingest` or `query` bucket | what a provider billed you. Absent stays `None`, never `0` |
| `engine_ms` from either | `trace.declared.engine_timings`, as `ingest_engine` or `retrieve_engine` samples | time only your system can see |
| a non-2xx status and its `error` | `trace.error` -- the step and your message | why this task has no value |
| -- | `trace.timings_ms`, keyed `ingest` and `retrieve` | memrank's own clock, at memrank's own call boundary. You never supply it |

`cleanup` puts nothing on a trace: it ends the group rather than answering a task.

Two consequences worth stating. A group's `ingest` is recorded on the trace of the first task that
group ran, not spread across them. And the line between `declared` and everything else is the line
between what you said and what memrank observed -- a [result](reference/result.md) never mixes the
two, so the source of each field remains explicit.

## 5. Token usage: absent is not zero

Pass token consumption through as `usage.total_tokens` where your system reports it, and **omit
the `usage` object entirely** where it does not.

Send `{"usage": {"total_tokens": 0}}` only for measured zero usage. Omit `usage` when usage is
unavailable; Memrank distinguishes missing information from zero.

## 6. Context budget

`capabilities.context_budget` must be `"matched"` for memory engines. It means the
context you return is capped at the shared `--token-budget` every system is held to, so a system
cannot win by returning more text.

The other two values exist only for memrank's own baseline arms and are not available to a
translator: `"uncapped"` for the full-context control, `"none"` for the no-memory control.

## 7. Latency, and why a translator is its own transport class

memrank measures wall-clock time at its own call boundary, including the
translator process and additional network call.
So memrank declares translator-backed systems as `transport: translator`, and
[methodology.md](methodology.md) rules that latency compares only within a transport class: a
translator is compared against other translators and never silently ranked against a system
memrank drives directly. Quality comparisons still require matching evaluations and experimental settings.

Report `engine_ms` where you can. It lands on the trace beside memrank's own wall-clock figure,
so a reader can see how much of the measured time was yours.

## 8. Errors

Return a non-2xx status with a JSON body:

```json
{"error": "myengine rejected the batch: document 12 exceeds the 100k character limit"}
```

memrank records the failure on that task's trace, with the step and your message, and carries on.
**Never swallow an error and return an empty result.** An empty `documents` list is a legitimate
answer meaning "nothing matched", so a system that returns it on failure measures exactly like the
no-memory control arm and reads as a real finding rather than a broken integration.

<a id="not-core-declaring-a-target-and-the-command-line"></a>
## Launch a translator from the command line

To launch a translator through [the command line](misc/command-line.md), register a **target**
descriptor. A target names the system and launch configuration; its **adapter** is the client
class that calls the translator.

Follow the remaining sections when your translator should be runnable **by name** -- from the
command line, in a sweep beside the [catalog](reference/catalog.md), or by other people. None of
it changes what the five endpoints do or what may be claimed about a measurement.

### 9. Declaring a target

*Target* is the command line's word for a named system. Put the descriptor next to the translator
it launches, in a folder you own:

```
~/evals/
├── myengine.yaml
└── translators/
    └── myengine.py
```

```yaml
# ~/evals/myengine.yaml
schema_version: 1
name: myengine:dev
kind: stack
interface:
  adapter: native          # the built-in client for this contract
  transport: translator
binding:
  kind: source             # root defaults to "." -- this descriptor's own directory
launch:
  command: "python translators/myengine.py --port {port}"
  requires: [translators/myengine.py]
network:
  port: 8099
  readiness: {path: /memrank/v1/describe}
```

Tell memrank where to look, once:

```bash
memrank config set targets.path ~/evals        # or MEMRANK_TARGETS_PATH, or several, os.pathsep-separated
memrank targets ls                             # myengine:dev, with the directory it came from
memrank targets verify myengine:dev            # check your translator against this contract
memrank submit myengine:dev demo
```

**The descriptor uses relative paths**, so the folder can be copied,
shared, or committed, and a colleague needs only their own `targets.path` line. `binding.root`
defaults to the directory the descriptor was read from, and a relative value is resolved against
it, so `launch.command` runs from the folder root.

**The folder can be a repository or an ordinary directory** -- its own repo, a scratch directory, or a `bench/` subfolder
inside your system's repo. Where there is a git repository memrank records the commit, dirty flag
and working-tree delta (from a subfolder, that is the *containing* repo's commit); where there is
not, it records `memrank:workspace_unversioned` and no commit rather than an empty one. Both are
`development_observation`.

One caveat for the `bench/`-inside-the-system layout: checking out an old revision takes the
translator with it, so it cannot evaluate revisions that predate the integration.

`{port}` expands to `network.port`. The command is split into argv and executed directly in
`binding.root` -- there is no implicit shell, so use `sh -lc '...'` explicitly if you need one.
`requires` is optional and names files that must exist, giving an early, clear error on a wrong
directory.

Do **not** declare a `components:` block on a native target. memrank cannot deliver component
configuration to a system whose environment it does not know, so declaring one is refused; your
launch command configures the system and `describe` reports it.

#### Credentials your translator needs

Name them, and memrank delivers them into your process's environment:

```yaml
secrets: [MYENGINE_TOKEN]                    # your translator reads MYENGINE_TOKEN

secrets:                                     # or a rename, if the two differ
  MYENGINE_TOKEN: AUTH_BEARER

secrets: [MYENGINE_USER, MYENGINE_PASSWORD]  # whatever shape your auth takes
```

Credentials are resolved in priority order -- process environment, then org secrets, then
the encrypted wallet -- so `memrank secrets set MYENGINE_TOKEN` is enough and the value never
appears in your shell history, a config file or the run record. A declared secret that resolves
nowhere is refused at preflight, naming every missing one at once. memrank interprets none of
them, whatever shape your auth takes.

#### Comparing against the catalog

A translator-backed target can sweep alongside anything else:

```bash
memrank submit myengine:dev,mem0,hindsight demo
```

Each target is provisioned and torn down before the next begins. Quality compares across all of
them; latency compares only within a transport class, so your `translator` row is never silently
ranked against a directly-driven `http` system (section 7). Evidence is per row -- yours is a
`development_observation` while the catalog rows keep whatever they had.

### 10. Evidence class

A source-bound target runs locally and is recorded as `development_observation` with
`publishable: false`. That is not a judgement about your system: memrank launched code from a
mutable working tree, so the run identifies a checkout rather than a reproducible artifact. Such
runs do not synchronize to an organization when the tree is dirty, and never reach the public
leaderboard.

Publishable evidence requires a pinned, reproducible artifact, and that path is not open to
translators yet. [Adding a system](systems.md) has the in-tree route in the meantime.

### 11. Conformance

`memrank targets verify <ref>` launches your target and checks contract conformance: the `describe` shape
and contract version, that a second `prepare` does not see the first group's documents, that
`retrieve` returns ranked documents, and that usage is either absent or a number. Run it before you
trust any value your translator produces.
