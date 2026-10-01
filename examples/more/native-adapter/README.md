# Reference translator

A working implementation of the [memrank system contract](../../../docs/system-contract.md), using
standard-library Python.

Write one of these and memrank can measure your memory system without a fork, a pull request, or
any change to memrank itself. memrank launches your program, drives it over HTTP, and never
imports your code -- so it can be in any language.

This template stores documents in a dictionary and ranks them by word overlap. Replace that
logic with calls to your engine before evaluating its behavior.

## Try it

```bash
python examples/more/native-adapter/translator.py --port 8099
```

Then, from another shell:

```bash
curl -s http://127.0.0.1:8099/memrank/v1/describe | python -m json.tool
```

<a id="not-core-point-memrank-at-it-by-name"></a>
## Register the translator as a target

Use a target descriptor to launch the translator through
[the command line](../../../docs/misc/command-line.md). You can also connect to an already-running
translator from Python with [`Native`](../../../docs/systems/native.md).

Make a folder you own -- anywhere, git or not -- and put a descriptor beside the translator:

```
~/evals/
├── reference.yaml
└── translators/
    └── reference.py          # a copy of translator.py
```

```yaml
# ~/evals/reference.yaml
schema_version: 1
name: reference:native
kind: stack
interface:
  adapter: native
  transport: translator
binding:
  kind: source                # root defaults to "." -- this descriptor's own directory
launch:
  command: "python3 translators/reference.py --port {port}"
  requires: [translators/reference.py]
network:
  port: 8099
  readiness: {path: /memrank/v1/describe}
```

```bash
memrank config set targets.path ~/evals
memrank targets verify reference:native
memrank submit reference:native demo
```

No absolute path appears in the descriptor, so the folder is shareable as-is: a colleague unpacks
it anywhere and sets their own `targets.path`. If the folder is a git repository memrank records
its commit and any uncommitted delta; if not, it records that the directory was unversioned. Either
way the run is a `development_observation` and cannot be published.

## Wrapping a real engine

Keep the five handlers and replace their bodies:

| Handler | What yours does |
|---|---|
| `_describe` | Report the engine's real name, version, LLM and embedder. memrank records this as engine-verified, so report the truth. `null` means "no such component"; it does not mean "did not check". |
| `_prepare` | Create or reset a namespace for `isolation_unit`. Nothing from a previous unit may remain visible. |
| `_ingest` | Write documents. Render `messages` into whatever shape your engine wants, chunk oversized content, and pass `timestamp` through so memories are dated to when they happened. |
| `_retrieve` | Search, and return documents **ranked best-first**. Put relevance in `metadata.score`. Return fewer than `k` rather than padding. |
| `_cleanup` | Drop the namespace. Must be safe to call twice and before any `prepare`. |

Follow these requirements:

1. **Omit `usage` if you do not measure tokens.** `{"total_tokens": 0}` claims zero usage. Omit the field when usage is unavailable.
2. **Never return an empty result on failure.** Raise, so memrank fails the run and shows your
   message. An empty list is a valid answer meaning "nothing matched", so a disguised failure
   scores exactly like the no-memory control arm and reads as a real result.
3. **Do not answer `describe` until your engine can serve traffic.** memrank uses it as the
   readiness probe, so respond only when the engine can handle evaluation requests.

## Evidence

Runs against a source-bound target are recorded as `development_observation` with
`publishable: false`, and a run from a dirty working tree will not sync anywhere. That is about
memrank having launched code from a mutable checkout, not about your engine. See
[the system contract](../../../docs/system-contract.md) section 10.
