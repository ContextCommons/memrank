# The command line

> **Deprecated:** this page describes memrank's older, pre-agent surface, which still ships but will be removed. To evaluate an agent, use `memrank run`; see the [README](../../README.md).

Use the command line to submit tracked runs, choose where they run, inspect their status and
save reproducibility receipts. For direct Python evaluation, see [Start here](../getting-started.md).

The command line uses a separate run pipeline and stored result format from
`evaluation.run()` in Python.

## Its vocabulary

The command line uses these terms in commands and stored records:

| Word here | What the Python interface calls it |
|---|---|
| **target** | a **system** that the catalog knows by name -- a named composition of a system plus the embedder and LLM it is configured with, identified by its configuration |
| **eval** | an **evaluation** that the catalog knows by name (`demo`, `locomo`, `beam`, `longmemeval`) |
| **adapter** | the class a target's entry resolves to: the code that talks to that system |
| **benchmark** | the class an eval's entry resolves to: the loader and scorer behind the name |

Target refs are `[namespace/]name[:preset]`. A bare ref is the vendor's own configuration, and
memrank's budget-matched comparison arm carries the suffix (`hindsight` vs `hindsight:matched`).

## Installing it

```bash
uv tool install memrank
memrank --version      # the version, and where this install came from
```

`memrank --version` prints where the install came from -- for a git install, the commit -- which is
useful information to include in an error report. `uv tool upgrade memrank` upgrades it.
[Local development](../local-development.md) has the source and editable installs.

## The loop

`demo` and `word-overlap` can run locally without a service or API key. Select `--on none`
explicitly for a local check: a configured cloud default can otherwise submit a paid run.
The transcript below shows the tracked-run workflow; it omits that placement flag.

```console
$ memrank submit word-overlap demo
run 20260826-213813__demo__7becda  (word-overlap × demo)
track: memrank watch 20260826-213813__demo__7becda

$ memrank runs ls
ID                             TARGET        EVAL  PLACE  STATE  SYNCED   AGE  DONE  SCORE
20260826-213813__demo__7becda  word-overlap  demo  local  done   pending   8s  100%  0.8000
```

`submit` always returns immediately with the run id(s); `watch <id>` blocks on them (its exit code
is the outcome: 0 all done, 1 any failed, 2 still going at timeout), `runs ls` lists them,
`runs show <id>` gives the full record -- state, where it ran, exit code, artifact location -- and
`kill <id>` stops one.

```bash
memrank runs ls --live           # what's still going (memrank ps says the same thing)
memrank logs <id> -f             # the run's log, wherever it ran, followed
memrank runs sync                # push finished local runs to your org, pull finished cloud ones
```

## Finding your way around

```bash
memrank targets ls               # what can be evaluated (hindsight, atomicmemory, word-overlap, ...)
memrank evals ls                 # what to evaluate against (locomo, beam, longmemeval, demo, ...)
memrank targets show hindsight   # the exact composition, and ✔/✘ per secret it needs
memrank evals show locomo        # units, slices, tiers, quality metric, judge requirement
memrank submit --help            # every flag, grouped
```

`memrank targets show <ref>` marks ✔/✘ per secret **before** anything is spent, so a missing key
surfaces as a refusal rather than a half-finished run. Store one with `memrank secrets set <NAME>`;
`memrank secrets ls` shows what the local wallet holds, with values masked and the source of each.

## Where a run happens

`--on` selects the placement. The default is `defaults.on` (`memrank config ls`), which is `none`
on a fresh install.

| | |
|---|---|
| `--on none` | an engine you are already running, at its configured URL |
| `--on local` | a disposable, isolated stack provisioned per run from the target manifest -- needs Docker |
| `--on cloud` | submitted to the hosted memrank platform -- needs a signed-in session (below) |

```bash
export HINDSIGHT_API_URL=http://localhost:8888
memrank submit hindsight locomo:smoke --on none
```

The variable above points at an engine you started. Whether you can obtain that engine's image at
all differs per target -- [engine images](engine-images.md) says which, and what to run for the
two you cannot pull.

Engine URL defaults, each overridable by its environment variable:

| Engine | Variable | Default |
|---|---|---|
| AtomicMemory | `ATOMICMEMORY_API_URL` | `http://localhost:3070` |
| Mem0 (HTTP) | `MEM0_HTTP_URL` | `http://localhost:8888` |
| Hindsight | `HINDSIGHT_API_URL` | `http://localhost:8888` |
| Supermemory | `SUPERMEMORY_BASE_URL` | `http://localhost:6767` |

Judged runs send evaluation content to Anthropic and need `ANTHROPIC_API_KEY`. They are on by
default for `locomo`, `longmemeval` and `beam`, whose only quality metric is the judge's, and off
where the evaluation scores itself. `--no-judge` measures latency and cost without a judged quality score.

## The hosted platform

Local runs work without signing in and store their records on your machine. Signing in is
needed only to submit runs to AtomicStrata's hosted platform and to share a run record with an
organisation.

```bash
memrank auth login
memrank auth status     # who you are, which orgs, when the session expires
memrank config ls       # every setting, its value, and where that value came from
```

A browser opens; approve with GitHub. Login also configures the machine -- it writes your default
org and points submissions at the platform, so later submissions can use those defaults. Use `--on none` explicitly for local checks. Only the *first* login writes those defaults, so a later sign-in never overrides one you
chose since (change it with `memrank config set defaults.org <slug>`).

> **"no default org"?** Membership is not self-served yet. Hosted runs are, for now, limited to
> accounts AtomicStrata has provisioned; local placements are not.

Hosted runs are held to your org's limits -- a quota on how many you may have started and still
running, and, for a run composed in the browser rather than submitted from the CLI, that org's
run-shape ceilings; a submission over either is refused naming the limit it crossed.

<details>
<summary>Signing in over SSH, or anywhere with no browser</summary>

When no browser is available, `auth login` prints a URL:

```console
$ memrank auth login
No browser here. Open this on any device to sign in:

      https://memrank.dev/signin?flow=Xk7pQ2...&client=cli

  signed in -- token expires 2026-09-12T...
```

Open it anywhere -- your laptop, your phone -- and approve with GitHub. The terminal is polling and
picks it up within a few seconds; the link is good for ten minutes.

Force this path with `--no-browser` when a browser exists but cannot actually open, such as an
`ssh -X` display that will not come up.

`MEMRANK_TOKEN` still overrides everything and is what CI should use -- but for interactive use on a
remote machine, use the browser flow above, and it does not put a 30-day credential in a shell
history or a second machine's dotfiles.
</details>

<details>
<summary>Where the session token is stored</summary>

`~/.memrank/credentials`, mode `0600` -- the model `aws`, `gcloud` and `kubectl` use.

Not the OS keychain by default, deliberately. memrank runs as a Python entry point, so the macOS
dialog names *python* rather than memrank, and it returns after every reinstall because the
permission binds to the interpreter. This can cause repeated permission prompts.

Prefer the keychain anyway? `memrank config set auth.keyring true`, then `memrank auth login`
again. `MEMRANK_TOKEN` overrides both, and is what CI should use.
</details>

## Connecting an AI agent

memrank does not currently install an MCP server command. To have a coding agent run an
evaluation, point it at `memrank run --help`.

## Every command

The table lists available commands. Run `memrank <command> --help` for options.

| Command | What it does |
|---|---|
| `memrank --version` / `memrank version` | the version, and where this build came from |
| `memrank submit TARGET EVAL [K=V...]` | submit a (target, eval) cell -- or a comma sweep -- and print one id per target |
| `memrank ps` | list running evals; `--all` also shows finished, failed and stale ones |
| `memrank watch RUN_ID...` | attach to run(s) until they end |
| `memrank kill RUN_ID...` | stop background run(s) |
| `memrank logs RUN_ID` | print, or with `-f` stream, a run's log, wherever it ran |
| `memrank runs ls` | list runs -- yours wherever they ran, newest first |
| `memrank runs show RUN_ID` | show one run in full: its state, and the results it recorded |
| `memrank runs sync` | push finished local runs to your organization and pull finished cloud runs |
| `memrank runs logs` / `watch` / `kill` | the same commands as the flat spellings above |
| `memrank targets ls` | list every known target |
| `memrank targets show REF [K=V...]` | a target's fully-resolved manifest and the secrets it needs |
| `memrank targets link [REF] [PATH]` | point a source target at a checkout on this machine |
| `memrank targets verify REF` | check a translator against the memrank system contract |
| `memrank targets render REF --for cloud\|local` | render the deployment document that runs a target |
| `memrank evals ls` | print every runnable eval, one per line |
| `memrank evals show REF` | one eval's units, slices, tiers, metrics and judge requirements |
| `memrank secrets set NAME` | store an API key -- in the local wallet, or for an org with `--org` |
| `memrank secrets ls` | the wallet with values masked, or an org's vault with `--org` |
| `memrank secrets rm NAME` | remove a secret from the wallet, or from an org's vault with `--org` |
| `memrank secrets import-env` | copy API keys out of a `.env` file into the wallet, once |
| `memrank secrets path` | print the wallet file path |
| `memrank auth login` | sign in through the browser, and configure this machine to submit |
| `memrank auth logout` | forget the stored CLI token on this machine |
| `memrank auth status` | who you are signed in as, and which orgs you can act on |
| `memrank config set KEY VALUE` | store a setting |
| `memrank config get KEY` | one setting's effective value |
| `memrank config ls` | every setting: its effective value, and which source supplied it |

Some older names still exist as stubs that refuse and name their replacement -- `login`, `logout`,
`whoami`, `list-benchmarks`, `list-runs`, `run`, `secrets list`, `status` and `preflight`, and the
`submit` flags `--adapter`, `--benchmark`, `--all-adapters`, `--tier`, `--slice`, `--ack-egress`
and `--max-judge-calls`. Typing one tells you what to type instead.

## Known limitations

- **A finished cloud run shows no SCORE** -- `runs ls` prints `—` and `runs show` says
  `results: (none recorded yet)`. The run did succeed and its artifact was written; nothing pulls
  it back into the record yet.
- **`memrank watch` of a cloud run** downloads artifacts straight from object storage on success,
  which needs credentials most people will not have. Follow cloud runs with `runs ls` instead.
- **Engine-backed cloud targets** (`mem0`, `atomicmemory`, `hindsight`, `supermemory`) are not
  serving yet. The same targets work under `--on none` and `--on local`.

Local placements -- the ones this page leads with -- are not affected by any of the above.

<a id="telling-us-something-broke"></a>
## Report a problem

`memrank --version` and the run id from `runs ls` make a report actionable. The version line says
where the install came from as well as what it is -- a released version identifies itself, and a
git install adds the commit, so two people reporting "0.2.0" are distinguishable. Paste the whole
error text: messages are written to name the fix, and one that doesn't is itself worth reporting.
