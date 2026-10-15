"""A stand-in judge for a local stack with no funded key: Anthropic's Messages API, and Chat Completions.

Not memrank code and never shipped. It is registered on the local API's organisation as an
endpoint -- ``memrank endpoints add stub --adapter anthropic --base-url http://127.0.0.1:8199``
for the Messages API (``POST /v1/messages``), or ``--adapter openai --base-url
http://127.0.0.1:8199/v1`` for Chat Completions (``POST /v1/chat/completions``) -- and a run
names it, ``--judge stub:claude-haiku-4-5``, so everything but the provider is real and the record
says the stub judged. Both adapters' models listing (``GET /v1/models``) lists :data:`MODELS`.
It grades a judge prompt deterministically -- passed when the reference answer's text
appears in the candidate answer -- and says in every rationale that it is a stub. BEAM's rubric
prompts get a score on its {0, 0.5, 1} scale from how many of a criterion's words the response
uses; its event prompts get the response's lines as events, and two events match when one
contains the other.

    python tests/fixtures/stub_anthropic.py 8199
"""

from __future__ import annotations

import json
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def _block(prompt: str, label: str) -> str:
    """The text wrapped under ``label:`` in a judge prompt (between its two sentinel lines)."""
    match = re.search(rf"{re.escape(label)}:\n[^\n]+\n(.*?)\n[^\n]+(?:\n|$)", prompt, re.S)
    return match.group(1).strip() if match else ""


#: Words shorter than this carry no meaning worth matching on.
_MIN_WORD = 4


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) >= _MIN_WORD}


def _nugget(prompt: str) -> dict:
    criterion = _words(_block(prompt, "Rubric criterion"))
    response = _words(_block(prompt, "Response to evaluate"))
    share = len(criterion & response) / len(criterion) if criterion else 0.0
    score = 1.0 if share >= 0.5 else 0.5 if share >= 0.2 else 0.0
    return {"score": score, "rationale": f"stub judge: {share:.0%} of the criterion's words used"}


def _events(prompt: str) -> dict:
    lines = [row.strip(" -*0123456789.") for row in _block(prompt, "Response").splitlines()]
    return {"events": [row for row in lines if row]}


def _equivalent(prompt: str) -> dict:
    one, two = _block(prompt, "Event A").lower(), _block(prompt, "Event B").lower()
    same = bool(one) and bool(two) and (one in two or two in one)
    return {"passed": same, "rationale": "stub judge: one event contains the other" if same
            else "stub judge: different events"}


def verdict(prompt: str) -> dict:
    if "Rubric criterion:" in prompt:
        return _nugget(prompt)
    if "Event A:" in prompt:
        return _equivalent(prompt)
    if "Response:" in prompt and "Candidate answer:" not in prompt:
        return _events(prompt)
    reference = _block(prompt, "Reference answer").lower()
    candidate = _block(prompt, "Candidate answer").lower()
    passed = bool(reference) and reference in candidate
    return {"passed": passed,
            "rationale": f"stub judge: reference {'found' if passed else 'not found'} in answer"}


#: The models both listings answer with: the ones a local witness judges with, the router-shaped
#: id included, so a witness names a model the way a run through the company router does.
MODELS = ("claude-haiku-4-5", "claude-sonnet-4-6", "claude-opus-4-8", "bedrock/claude-haiku-4-5")


def _message(model: str, text: str) -> dict:
    return {"id": "msg_stub", "type": "message", "role": "assistant", "model": model,
            "stop_reason": "end_turn", "stop_sequence": None,
            "content": [{"type": "text", "text": text}],
            "usage": {"input_tokens": 0, "output_tokens": 0}}


def _chat_completion(model: str, text: str) -> dict:
    return {"id": "chatcmpl-stub", "object": "chat.completion", "created": 0, "model": model,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": text}}]}


class Messages(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - the stdlib's name
        if not self.path.rstrip("/").endswith("/models"):
            self._send(404, {"error": {"message": f"no route {self.path}"}})
            return
        self._send(200, {"data": [{"id": model, "type": "model"} for model in MODELS]})

    def do_POST(self) -> None:  # noqa: N802 - the stdlib's name
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        prompt = "\n".join(str(m.get("content", "")) for m in body.get("messages", [])
                           if m.get("role") != "system")
        text = json.dumps(verdict(prompt))
        chat = self.path.rstrip("/").endswith("/chat/completions")
        self._send(200, _chat_completion(body.get("model"), text) if chat
                   else _message(body.get("model"), text))

    def _send(self, status: int, reply: dict) -> None:
        data = json.dumps(reply).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args: object) -> None:
        """Quiet."""


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), Messages).serve_forever()
