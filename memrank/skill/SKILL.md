---
name: memrank
description: Evaluate an AI agent with Memrank - on a shipped benchmark (locomo, longmemeval, beam) or on the user's own cases - and report its score and results link. Use when asked to evaluate, benchmark, score or compare an agent, chatbot or memory layer with Memrank, or to run `memrank run`. Not for editing Memrank itself.
---

# Evaluating an agent with Memrank

The user asked you to evaluate **their agent** with Memrank. Done means: a run of their agent
finished, and you reported its score line, its likely range and the `View results` link.

Memrank runs next to the user's agent: it feeds the agent past conversations, asks it questions,
judges the answers with the organisation's Anthropic key, and records the run in the
organisation's run history, where it has a page.

Rules that hold throughout:

- Run everything yourself, from the folder of the project that contains their agent.
- Never print, log or commit a secret, and never ask the user to paste a key into the chat.
  Refer to keys by environment variable name only.
- Do not use `memrank submit`: an older command that starts paid cloud runs.
- Every `memrank run` ends with a block saying what happened and what to do (step 7). Follow it
  rather than guessing. When a step is the user's (a browser sign-in, a key, a decision), give
  them the command, stop, and wait for them.

## 1. Install

```sh
memrank --version
```

If it prints a version, go to step 2. Otherwise install it:

```sh
curl -fsSL https://memrank.ai/install.sh | sh
```

It supports macOS on Apple Silicon and writes only `~/.local/share/memrank` and
`~/.local/bin/memrank`. Then make sure `memrank` resolves in a new shell. If it does not, add
`~/.local/bin` to PATH in the user's shell startup file (the installer prints the line), and
tell the user the change you made.

## 2. Sign in

```sh
memrank auth status
```

If it does not say `signed in`, run `memrank auth login`. It opens the sign-in page in the
browser, prints its link, and waits. Give that link to the user to open in their browser on this
computer (do not fetch it yourself), and keep the command waiting until it prints `signed in`.
Where there is no browser, it prints a link to open on any device instead; hand that on the same
way.

## 3. Describe the user's agent

Find how their agent is run: a program, an OpenAI-compatible chat server, or its own HTTP API.
Read its code or README; ask the user only if you cannot tell. Then write `memrank-agent.yaml`
in the project root with one of the three connectors in [agent-files.md](agent-files.md). If
none fits without changing their agent, write a small adapter script beside it and use the
command connector; do not change the agent itself unasked.

The agent must keep memory separate per session id: Memrank checks this before every run. If
the agent has one global memory, tell the user it needs keying by the session id rather than
working around the check.

`memrank agents ls` lists the agents Memrank ships, as baselines: `full-context` answers from
the whole history with Claude, and needs `ANTHROPIC_API_KEY` in the environment or saved with
`memrank secrets set ANTHROPIC_API_KEY` (no `--org`).

## 4. Check the connection

```sh
memrank run locomo --agent memrank-agent.yaml --cases 1 --questions 2 --no-judge
```

This spends nothing on judging. Continue when it ends with `✓ Done`; otherwise follow its ending.

## 5. If the user brought their own cases

Put them in an evaluation file. Start from the commented template (it documents every field),
fill it with the user's cases, and validate it until it says it is valid:

```sh
memrank evals new memrank-eval.yaml
memrank evals check memrank-eval.yaml
```

Graders: `exact`, `numeric` and `choice` need no key; `judge` and `rubric` use the judge key; a
program can grade too (`grade: {command: [...]}`). Never invent reference answers: every
`answer` must come from the user or their data. Without their own cases, use a shipped
evaluation: `memrank evals ls` lists them (`locomo`, `longmemeval`, `beam:100k`, ...).

## 6. Run

```sh
memrank run locomo --agent memrank-agent.yaml --cases 10 --questions 10
memrank run memrank-eval.yaml --agent memrank-agent.yaml
```

As it starts, it prints `memrank: Recording to org <org>` and `memrank: View run live at <url>`,
the run's page: tell the user which organisation the run is recorded in, and give them that link.
Judging spends the user's Anthropic credit, so keep `--cases` and `--questions` small for a first
run and tell the user before a larger one. If the organisation has no key saved, the run stops
before starting and prints `memrank secrets set ANTHROPIC_API_KEY --org <org>`: ask the user to
run it themselves (it asks for the key with hidden input; in Claude Code they can type `!`
followed by the command), then run again. A long run may outlast your tool's timeout: run it in
the background, and do not start it twice.

## 7. Read the ending

The ending's first line says where you are. `✓` is done, `!` means a step is needed, `x` means
Memrank itself broke. Below it, **What happened** explains and **What to do** lists the steps,
each command on its own indented line: run those commands exactly as printed.
`Details from <system>:` is another system's raw words, such as the user's agent's error.

| First line | Meaning | What you do |
|---|---|---|
| `✓ Done: ...` | Finished. | Report (step 8). |
| `! The run didn't start` | Something is missing first: a login, a key, a file. | Do the step it prints, or ask the user for it. |
| `! Your agent isn't running` | Nothing answered at `base_url`. | Start the agent, fix `base_url`, or add `start`; run the command it prints. |
| `! Your agent didn't start` | The `start` command exited or never became ready. | Read the log it names, fix the cause, run the command it prints. |
| `! Your agent failed the check before the run` | The agent errored during the separation check. | Read the details, fix the agent or the file, run the command it prints. |
| `! Your agent mixed up two conversations` | The agent answered one session from another's memory. | Tell the user: the agent must key its memory by the session id. |
| `! The run stopped` | The agent stopped answering part-way; progress is saved. | Fix the agent, then run the `--resume` command it prints. |
| `! Judging couldn't run` | Anthropic refused or could not be reached; answers are saved. | Tell the user what it says about their key or credit; once fixed, run the `--resume` command it prints. |
| `! The run couldn't be saved to <org>'s run history` | The upload failed; the run is saved locally. | Run the `--resume` command it prints once Memrank is reachable. |
| `x Something went wrong inside memrank` | A bug in Memrank. | Tell the user, with the details; do not retry in a loop. |

`memrank run --resume <run-id> ...` continues a run without asking answered questions again;
always use the exact command the ending prints, since it carries the options the run needs.
A case whose reset or feed failed, or an answer that failed, stays failed on a resume; once its
cause is fixed, add `--retry-failed` to that command to run those cases again from a fresh reset,
asking only the questions that failed or were never asked.

## 8. Report

Give the user, from the ending: the `Score` and `Likely range` lines, the `By kind` lines if
present, `Not answered`, and the `View results` link. Say in one sentence what the likely range
means: where the score would usually land on other conversations like these, found by redrawing
whole conversations, so a run over few conversations has a wide range and cannot show that one
agent beats another. Offer a next step: a baseline run with `--agent full-context`
on the same `--cases`, `--questions` and `--seed`.
