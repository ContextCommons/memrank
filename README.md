<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/ContextCommons/memrank/main/docs/assets/memrank-logo-dark.svg">
    <img src="https://raw.githubusercontent.com/ContextCommons/memrank/main/docs/assets/memrank-logo-light.svg" alt="memrank" width="360">
  </picture>
</p>

<h1 align="center">Find the memory that works best for your agent.</h1>

<p align="center">
  <a href="https://github.com/ContextCommons/memrank/actions/workflows/public-ci.yml"><img src="https://github.com/ContextCommons/memrank/actions/workflows/public-ci.yml/badge.svg?branch=main" alt="public-ci workflow status on main"></a>
  <a href="https://github.com/ContextCommons/memrank/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="Code license: Apache-2.0"></a>
</p>

Memrank is a command-line tool that evaluates your agent end to end, so you can see which memory
engine, version or setup actually answers its questions better. It is for people building agents:
evaluate your agent on your tasks, put the result beside other versions and baselines, and choose
with evidence. Memrank provides both the evaluation framework and the evaluations.

[Documentation](https://memrank.ai/docs) |
[Quick start](https://memrank.ai/docs/quickstart) |
[Connect your agent](https://memrank.ai/docs/connect-your-agent) |
[Command reference](https://memrank.ai/docs/reference/commands)

<a href="https://memrank.ai">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/ContextCommons/memrank/main/docs/assets/beam-chart-dark.png">
    <img src="https://raw.githubusercontent.com/ContextCommons/memrank/main/docs/assets/beam-chart-light.png" alt="Chart of answer quality against median memory retrieval time for memory systems on BEAM 100K" width="100%">
  </picture>
</a>

*Answer quality against median memory retrieval time (log scale) for the memory systems published
on the BEAM 100K board at [memrank.ai](https://memrank.ai).*

## How it works

Memrank treats your agent as a black box. Your agent needs no Memrank code and keeps running
where it already runs.

1. **Connect your agent.** A small [agent file](https://memrank.ai/docs/reference/agent-file)
   says how Memrank reaches it: a command, an OpenAI-compatible URL, or any HTTP API.
2. **Run an evaluation.** For every case, Memrank starts a fresh conversation, feeds the agent
   the past conversations, asks questions about them, and judges every answer. Before it asks
   anything, it checks that your agent keeps conversations separate.
3. **Compare runs.** Each run reports the score, the failures and the latency. Put it beside an
   earlier version of your agent, another memory setup, or a baseline Memrank ships, such as
   `full-context`, which answers from the whole history with no memory engine. Runs with the
   same evaluation, sample and seed ask the same questions.

Answers are judged by an LLM judge with your organisation's own API key, on your machine. Each
score comes with a 95% interval, so you can tell a real difference from noise. Results land in
the terminal, in a `results/<run-id>/` folder on your machine, and in your organisation's run
history on memrank.ai, where every question, answer and verdict can be read.

## Evaluations included

- [LoCoMo](https://memrank.ai/docs/benchmarks/locomo): long conversations between two people,
  over weeks of dated sessions.
- [LongMemEval](https://memrank.ai/docs/benchmarks/longmemeval): five hundred questions, each with
  its own haystack of dated chat sessions.
- [BEAM](https://memrank.ai/docs/benchmarks/beam): very long conversations from one user, at 100K,
  500K and 1M tokens, testing ten memory abilities.
- [Your own cases](https://memrank.ai/docs/benchmarks/your-own-cases): an evaluation file with the
  conversations, questions and reference answers that matter to you, graded by exact match,
  choice, number, the judge, a rubric, or your own program.

`memrank evals ls` lists every shipped evaluation and its smaller slices.

## Install

The current version runs on macOS with Apple Silicon. You need a [memrank.ai](https://memrank.ai)
account and an API key for the judge.

```bash
curl -fsSL https://memrank.ai/install.sh | sh
memrank --version
```

<details>
<summary>What the installer does</summary>

The installer puts Memrank, with its own Python, under `~/.local/share/memrank` and links
`~/.local/bin/memrank`. It does not touch your Python, your shell startup files, or anything that
needs `sudo`. If it says `~/.local/bin` is not on `PATH`, run the `export` line it prints.

</details>

## First run

Every run is recorded in your organisation's run history, so `memrank run` needs you to sign in
first. Judging spends your own credit; this run asks 15 questions.

```bash
memrank auth login
export ANTHROPIC_API_KEY=...   # the model key full-context answers with
memrank run locomo --agent full-context --cases 3 --questions 5
```

This evaluates `full-context`, a baseline agent that answers from the whole conversation history,
on three LoCoMo conversations. The first judged run asks for the judge's API key and saves it to
your organisation for later runs. While it runs, Memrank shows its progress and a link to follow
the run on memrank.ai. It ends with a `Done` line followed by a summary like this (abridged; your
numbers will differ):

```text
  Score         53.3% correct: 8 of 15 questions, from 3 conversations
  Likely range  27%-80%: on other conversations like these, the score would usually land here
  Not answered  none
Saved locally: results/20260928-182017__locomo__f11cd7/
View results: https://memrank.ai/acme/runs/20260928-182017__locomo__f11cd7
```

Open the `View results` link to see every question with the agent's answer, the reference answer
and the judge's verdict. If the run stops instead, its last block says what happened and the exact
command to run next. Three conversations only check that everything works; comparing two agents
needs more, as [Compare results](https://memrank.ai/docs/results) explains.

To evaluate your own agent, write an [agent file](https://memrank.ai/docs/connect-your-agent) and
pass its path instead of `full-context`.

### Hand the setup to a coding agent

To have a coding agent set Memrank up and evaluate your agent, paste this into Claude Code, Codex
or Cursor:

```text
Evaluate my agent with Memrank, then show me its score and the link to the result. First set
Memrank up:

1. Run: curl -fsSL https://memrank.ai/install.sh | sh
2. Ensure `memrank` resolves in a new shell; if it does not, add ~/.local/bin to PATH in the
   shell startup file and report the change you made.
3. Run: memrank auth login; it opens the sign-in page, prints its link and waits. Give that link
   to me to open in my browser on this laptop (do not fetch it yourself), and keep the command
   waiting until it prints signed in.
4. Run: memrank skills install --agent codex|claude|cursor, choosing the one that names your own
   harness, then reload yourself so the Memrank skill loads (or read the SKILL.md it prints), and
   follow it.
```

See [For coding agents](https://memrank.ai/docs/coding-agents) for the full steps.

## Common tasks

| Task | Guide |
|---|---|
| Connect your agent to Memrank | [Connect your agent](https://memrank.ai/docs/connect-your-agent) |
| Evaluate on your own questions | [Your own cases](https://memrank.ai/docs/benchmarks/your-own-cases) |
| Compare two runs | [Compare results](https://memrank.ai/docs/results) |
| See how answers are judged | [Evaluation methodology](https://memrank.ai/docs/methodology/evaluation-methodology) |
| See what a run cannot tell you | [Limitations](https://memrank.ai/docs/methodology/limitations) |
| Look up a command | [Command reference](https://memrank.ai/docs/reference/commands) |

## Contributing

Memrank aims to be a neutral evaluation tool for agent memory that the whole industry can stand
behind, built together with its ecosystem. This is the beginning of that journey, and anything you think
would make it better is welcome: ideas, issues, discussions or code.

Some places to start: a connector or preset for an agent stack Memrank does not reach easily yet,
a new evaluation, a new baseline, better grading, or a clearer command line and run report. The
[contributing guide](https://github.com/ContextCommons/memrank/blob/main/docs/contributing.md)
covers the development setup and where each of these lives in the code.

## Help

Report a bug or ask a question in [GitHub issues](https://github.com/ContextCommons/memrank/issues).
Include the Memrank version (`memrank --version`), the command you ran and the block the run ended
with. Issues are public, so keep them to what is safe to share: never include API keys or
private data.

## License

Memrank's code is [Apache-2.0](https://github.com/ContextCommons/memrank/blob/main/LICENSE).
Datasets that download on first use keep their own licences.
