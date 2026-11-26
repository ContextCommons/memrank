# Agent files

An agent file is YAML: who the agent is, and how Memrank reaches it. Unknown keys are refused,
so a misspelled field fails before anything runs.

```yaml
name: my-agent             # the person's name for it; runs are recorded under it
version: "2026-09-28"      # optional; change it when the agent changes
description: One sentence. # optional
connector: command         # command, openai-chat, or http
```

Every case is fed and asked under its own `{session_id}`. The agent must keep memory per
session id and start empty for a new one; Memrank checks this before each run and refuses an
agent that recalls another session.

Commands run without a shell, from the directory `memrank run` is started in: use absolute
paths for scripts and data.

## command: a program run per step

```yaml
connector: command
reset: [python3, /abs/path/agent.py, reset, "{session_id}"]   # optional; before each case
feed: [python3, /abs/path/agent.py, feed, "{session_id}"]     # the case's history on stdin
ask: [python3, /abs/path/agent.py, ask, "{session_id}", "{question}"]  # answer on stdout
```

- `feed` receives the history as plain text on stdin: one line per turn (`Speaker: text`, or
  the text alone when the turn names no speaker), each session headed `[Session date: ...]`
  when it has a date.
- `ask` prints the answer on stdout; a non-zero exit is a failed answer, shown with the end of
  stderr. Keep stdout for the answer only.
- Variables: `{session_id}`, `{case_id}`, `{python}`; in `feed` also `{transcript}`; in `ask`
  also `{question}`, `{question_id}`, `{timestamp}`.
- Optional: `ask_json: {answer: answer, retrieval_ms: retrieval_ms}` makes `ask` print one JSON
  object instead, with the answer and the agent's own retrieval time read from it by JMESPath
  (see `ask.retrieval_ms` under http). Without `ask_json`, stdout is the answer as plain text.
  `ask_json` may also name `usage`, and `feed_json: {usage: usage}` makes `feed` print a JSON
  object read only for it: the tokens the agent's paid components were billed on that step
  (see [Reporting token usage](#reporting-token-usage)).

## openai-chat: an OpenAI Chat Completions server

```yaml
connector: openai-chat
base_url: http://127.0.0.1:8100
vars: {model: gpt-4.1-mini}             # required; sent as "model"
auth: {env: OPENAI_API_KEY}             # optional; sent as "Authorization: Bearer $OPENAI_API_KEY"
start:                                  # optional; without it the server must be running already
  argv: [python3, /abs/path/server.py, --port, "8100"]
  ready: http://127.0.0.1:8100/health   # memrank waits for a 2xx here, and stops it after the run
```

Each past session is POSTed to `/v1/chat/completions` as one request with
`"user": "<session_id>"` and `"metadata": {"memrank_op": "feed"}` (the agent may skip replying
to these); each question is a request under the same `user` with `"memrank_op": "ask"`, and the
answer is `choices[0].message.content`. To report how long its memory retrieval took for an
answer, the agent adds a top-level `"memrank_retrieval_ms": <milliseconds>` to that completion
(see `ask.retrieval_ms` under http); without it, none is reported. Any completion, a history
request's or a question's, may likewise carry a top-level `"memrank_usage": [...]` (see
[Reporting token usage](#reporting-token-usage)). `start.argv` may use `{python}` and `{evaluation}` (the
run's eval ref, e.g. `beam:100k`, or its evaluation file's path), for a server that answers
differently per evaluation; no other variable is available there.

## http: any other HTTP API

```yaml
connector: http
base_url: http://127.0.0.1:8000
auth: {env: MY_AGENT_TOKEN, header: Authorization, prefix: "Bearer "}   # optional
reset: {method: DELETE, path: "/sessions/{session_id}"}                  # optional
feed: {per: session, path: /memory, body: {session: "{session_id}", messages: "{messages}"}}
ask: {path: /ask, body: {session: "{session_id}", q: "{question}"}, answer: answer}
```

- `method` defaults to POST. `body` is any JSON; a string that is exactly `"{name}"` keeps the
  variable's type (a list stays a list), and any other `{name}` is replaced as text.
- `feed.per` is `turn`, `session` (default) or `case`, with these variables:
  - turn: `{text}`, `{speaker}`, `{role}`, `{timestamp}`
  - session: `{messages}` (chat messages), `{transcript}`, `{timestamp}`
  - case: `{sessions}` (structured), `{transcript}`
- `feed: chat` instead replays every past turn through the `ask` request, for an agent that only
  chats; the replies are discarded.
- `ask.answer` is a JMESPath expression locating the answer string in the JSON response.
- `ask.retrieval_ms` (optional) is a JMESPath expression locating how long the agent's memory
  retrieval took for this answer, in milliseconds, e.g. `meta.retrieval_ms`. Memrank records it
  beside the answer time it measures, as the agent's own report. An answer where it finds
  nothing counts as not reported; a value that is not a non-negative number fails the answer.
- `ask.usage` and `feed.usage` (optional) are JMESPath expressions locating the step's token
  usage (see [Reporting token usage](#reporting-token-usage)); a feed of several requests
  reports the entries of all of them.
- `start` works as in openai-chat. Every call also has `{session_id}`, `{case_id}` and the file's
  own `vars`.

## Reporting token usage

An agent may report what its paid components were billed on a feed or an ask step, as a list of
entries copied from its provider's response (not estimated):

```json
[{"component": "reader", "provider": "anthropic", "model": "claude-haiku-4-5",
  "input_tokens": 1834, "output_tokens": 12},
 {"component": "embedder", "provider": "openai", "model": "text-embedding-3-small",
  "input_tokens": 9}]
```

- One entry per paid component or per billed call. `component` is the agent's name for the part;
  `reader` is the model that writes the answer, and its input tokens on an ask are the question's
  context tokens. `provider` and `model` are who billed it and the model id as billed.
- `input_tokens` is every input token billed, cached or not; `output_tokens` is omitted for an
  embedding. Counts are whole numbers, at least 0. No other keys are accepted.
- Leave the field out when you do not know; send `[]` when nothing paid was called. A malformed
  list fails the step.
- memrank prices the entries from its price table, which holds only prices providers publish. A
  provider/model with no price gets no cost at all (the tokens are still shown), never $0, and
  the run says how to give one.

To price a model the table lacks (a private router, a self-hosted model, a negotiated rate), or
to change a price it holds, add `prices:` to the agent file, keyed by `provider` then `model`
exactly as the entries report them, in US dollars per million tokens:

```yaml
prices:
  my-router:
    claude-haiku-4-5: {input_per_mtok: 1.10, output_per_mtok: 5.50}
    my-embedder: {input_per_mtok: 0, output_per_mtok: 0}   # self-hosted: say 0 to mean free
```

A price given here wins over the table's, and the result's cost basis lists every one it used.

## `{python}`

`{python}` is the Python that runs Memrank. In the standalone install (`curl ... | sh`) that is
Memrank's own bundled interpreter: it has the standard library but none of the person's
packages. A script that imports anything else must name the person's interpreter instead:
`python3`, or the absolute path of their virtualenv's `bin/python`.
