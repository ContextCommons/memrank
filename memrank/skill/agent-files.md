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
answer is `choices[0].message.content`. `start.argv` may use `{python}` and `{evaluation}` (the
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
- `start` works as in openai-chat. Every call also has `{session_id}`, `{case_id}` and the file's
  own `vars`.

## `{python}`

`{python}` is the Python that runs Memrank. In the standalone install (`curl ... | sh`) that is
Memrank's own bundled interpreter: it has the standard library but none of the person's
packages. A script that imports anything else must name the person's interpreter instead:
`python3`, or the absolute path of their virtualenv's `bin/python`.
