"""A model-free agent for zero-cost end-to-end runs: it answers with the remembered line that
shares the most words with the question. Deterministic, stdlib only, and not memrank code --
it knows nothing of memrank beyond the two public ways memrank reaches an agent.

As a command agent (the `command` connector), memory is one file per session id:

    python keyword_agent.py DIR reset SESSION
    python keyword_agent.py DIR feed SESSION < transcript
    python keyword_agent.py DIR ask SESSION "question"

As an OpenAI-compatible server (the `openai-chat` preset), memory is per `user`, in process:

    python keyword_agent.py serve PORT
"""

from __future__ import annotations

import json
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

WORD = re.compile(r"[a-z0-9]+")
STOP = frozenset("a an and are as at be by did do does for from had has have he her his how i "
                 "in is it me my of on or she that the their they this to was were what when "
                 "where which who why with you your".split())
UNKNOWN = "I don't know."


def words(text: str) -> set[str]:
    return {w for w in WORD.findall(text.lower()) if w not in STOP}


def answer(memory: list[str], question: str) -> str:
    """The remembered line sharing the most words with the question; UNKNOWN when none does."""
    asked = words(question)
    best, overlap = UNKNOWN, 0
    for line in memory:
        shared = len(asked & words(line))
        if shared > overlap:
            best, overlap = line.strip(), shared
    return best


def command(store: Path, op: str, session: str, question: str = "") -> None:
    store.mkdir(parents=True, exist_ok=True)
    path = store / f"{session}.txt"
    if op == "reset":
        path.unlink(missing_ok=True)
    elif op == "feed":
        with path.open("a", encoding="utf-8") as memory:
            memory.write(sys.stdin.read() + "\n")
    elif op == "ask":
        lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
        print(answer(lines, question))
    else:
        sys.exit(f"unknown op {op!r}")


SESSIONS: dict[str, list[str]] = {}


class ChatHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - the stdlib's name
        """/health, for the agent file's `start.ready`."""
        self._send(200 if self.path == "/health" else 404, {"status": "ok"})

    def do_POST(self) -> None:  # noqa: N802 - the stdlib's name
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        user = body.get("user")
        if not user:
            return self._send(400, {"error": {"message": "set `user`"}})
        contents = [str(m.get("content", "")) for m in body["messages"]]
        memory = SESSIONS.setdefault(user, [])
        if (body.get("metadata") or {}).get("memrank_op") == "feed":
            memory.extend(contents)
            return self._send(200, _completion(""))
        return self._send(200, _completion(answer(memory, contents[-1])))

    def _send(self, status: int, payload: dict) -> None:
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args: object) -> None:
        """Quiet: the runner reports every step."""


def _completion(content: str) -> dict:
    return {"object": "chat.completion", "model": "keyword",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": content}}]}


if __name__ == "__main__":
    if sys.argv[1] == "serve":
        ThreadingHTTPServer(("127.0.0.1", int(sys.argv[2])), ChatHandler).serve_forever()
    else:
        command(Path(sys.argv[1]), sys.argv[2], sys.argv[3], *sys.argv[4:5])
