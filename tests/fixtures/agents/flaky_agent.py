"""The keyword agent, failing some of its answers the three ways a real agent does: for local
witnesses of how a run reports questions it got no answer to (ATO-2378). Never used by a test.

Which questions fail is decided by a hash of the question, so every run fails the same ones:
about 8% are answered only after the agent file's `timeout_s` has passed (the run records a
read timeout), 3% get an HTTP 500, and 1% have the connection dropped with no answer.

    python flaky_agent.py PORT
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from http.server import ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from keyword_agent import SESSIONS, ChatHandler, _completion, answer  # noqa: E402

#: Longer than flaky-openai.yaml's `timeout_s`, so the runner gives up first.
SLOW_S = 3.0
#: Out of 100 questions: the first this many time out, the next error, the next are dropped.
TIMEOUTS, ERRORS, DROPS = 8, 3, 1
#: Words of the question memrank's separation check asks (memrank/connect/probe.py).
PROBE = "secret code word"


def bucket(question: str) -> int:
    return int(hashlib.sha256(question.encode()).hexdigest(), 16) % 100


class FlakyHandler(ChatHandler):
    def do_POST(self) -> None:  # noqa: N802 - the stdlib's name
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        asked = (body.get("metadata") or {}).get("memrank_op") != "feed"
        question = str(body["messages"][-1].get("content", "")) if body.get("messages") else ""
        # The separation check's question is answered normally: it must pass for a run to start.
        which = bucket(question) if asked and PROBE not in question else 99
        if which < TIMEOUTS:
            time.sleep(SLOW_S)
        elif which < TIMEOUTS + ERRORS:
            return self._send(500, {"error": {"message": "context length exceeded"}})
        elif which < TIMEOUTS + ERRORS + DROPS:
            self.close_connection = True
            return
        memory = SESSIONS.setdefault(body.get("user") or "", [])
        contents = [str(m.get("content", "")) for m in body.get("messages", [])]
        if not asked:
            memory.extend(contents)
            return self._send(200, _completion(""))
        return self._send(200, _completion(answer(memory, contents[-1])))


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), FlakyHandler).serve_forever()
