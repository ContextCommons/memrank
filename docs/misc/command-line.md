# The command line

> **Deprecated:** this page describes memrank's older, pre-agent surface. `memrank submit` has been removed; the commands that inspect earlier runs still ship but will be removed. To evaluate an agent, use `memrank run`; see the [README](../../README.md).

Use the command line to inspect tracked runs and their reproducibility receipts. For direct Python evaluation, see [Start here](../getting-started.md).

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

`memrank submit`, which started tracked runs of named targets, has been removed: evaluate an agent
with `memrank run` instead (see the [README](../../README.md#first-run)). The commands below that
inspect earlier runs still work.

```bash
memrank runs ls                  # runs, newest first
memrank runs show <id>           # one run in full
```

## Finding your way around

```bash
memrank targets ls               # what can be evaluated (hindsight, atomicmemory, word-overlap, ...)
memrank evals ls                 # what to evaluate against (locomo, beam, longmemeval, demo, ...)
memrank targets show hindsight   # the exact composition, and ✔/✘ per secret it needs
memrank evals show locomo        # units, slices, tiers, quality metric, judge requirement
memrank run --help               # every flag, grouped
```

`memrank targets show <ref>` marks ✔/✘ per secret **before** anything is spent, so a missing key
surfaces as a refusal rather than a half-finished run. Store one with `memrank secrets set <NAME>`;
`memrank secrets ls` shows what the local wallet holds, with values masked and the source of each.

## Where a run happens

Placement belonged to the removed `memrank submit`. `memrank run`
runs next to your agent and uploads every run to your organisation. Whether you can obtain an
engine's image differs per target -- [engine images](engine-images.md) says which.

## The hosted platform

`memrank run` needs a sign-in: it uploads every run to your organisation's run history, and keeps
a local run folder as the working record.

```bash
memrank auth login
memrank auth status     # who you are, which orgs, when the session expires
memrank config ls       # every setting, its value, and where that value came from
```

A browser opens; approve with GitHub. Login also configures the machine -- it writes your default
org. Only the *first* login writes that default, so a later sign-in never overrides one you
chose since (change it with `memrank config set defaults.org <slug>`).

> **"no default org"?** Membership is not self-served yet. Hosted runs are, for now, limited to
> accounts AtomicStrata has provisioned; local placements are not.

Hosted runs are composed in the browser and held to your org's limits -- a quota on how many you
may have started and still running, and that org's run-shape ceilings; a submission over either is
refused naming the limit it crossed.

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
| `memrank auth login` | sign in through the browser, and configure this machine's default org |
| `memrank auth logout` | forget the stored CLI token on this machine |
| `memrank auth status` | who you are signed in as, and which orgs you can act on |
| `memrank config set KEY VALUE` | store a setting |
| `memrank config get KEY` | one setting's effective value |
| `memrank config ls` | every setting: its effective value, and which source supplied it |

Some older names still exist as stubs that refuse and name their replacement -- `login`, `logout`,
`whoami`, `list-benchmarks`, `list-runs`, `secrets list`, `status` and `preflight`. Typing one
tells you what to type instead.

## Known limitations

- **A finished cloud run shows no SCORE** -- `runs ls` prints `—` and `runs show` says
  `results: (none recorded yet)`. The run did succeed and its artifact was written; nothing pulls
  it back into the record yet.
- **`memrank watch` of a cloud run** downloads artifacts straight from object storage on success,
  which needs credentials most people will not have. Follow cloud runs with `runs ls` instead.
- **Engine-backed cloud targets** (`mem0`, `atomicmemory`, `hindsight`, `supermemory`) are not
  serving yet.

<a id="telling-us-something-broke"></a>
## Report a problem

`memrank --version` and the run id from `runs ls` make a report actionable. The version line says
where the install came from as well as what it is -- a released version identifies itself, and a
git install adds the commit, so two people reporting "0.2.0" are distinguishable. Paste the whole
error text: messages are written to name the fix, and one that doesn't is itself worth reporting.
