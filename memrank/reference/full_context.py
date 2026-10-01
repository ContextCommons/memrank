# Copyright 2026 AtomicStrata
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.
"""The full-context reference agent: no memory system, the whole history in the prompt.

The baseline every memory agent is compared against. It speaks the OpenAI Chat Completions API
(so the ``openai-chat`` preset reaches it), keeps each session's messages in process memory
keyed by the request's ``user``, and answers a question by sending that session's entire
history plus the question to the model -- through the evaluation's own reader prompt
(``Benchmark.answer_prompt``: the benchmark's official one where it publishes one) and memrank's
Anthropic client, the same ones the in-process harness answers with. ``memrank agents serve
full-context --evaluation REF`` fixes the prompt for the process; the shipped spec passes the
run's ``{evaluation}``.

History requests (``metadata.memrank_op == "feed"``) are stored and answered with an empty
message, without calling the model. A request without ``user`` is refused: an agent that cannot
tell sessions apart must not guess.
"""

from __future__ import annotations

import threading
import time
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

from memrank.connect.template import CURRENT_DATE, SESSION_DATE
from memrank.judging.judge import Completer, generate_answer
from memrank.judging.prompts import AnswerPrompt
from memrank.reference import KEY_NAME, PROVIDER, PROVIDER_FAILED, PROVIDER_REFUSED

FEED_OP = "feed"


class ChatRequest(BaseModel):
    """The fields of a Chat Completions request this agent reads; others are accepted and ignored."""

    model_config = ConfigDict(extra="allow")

    messages: list[dict[str, Any]]
    user: str | None = None
    metadata: dict[str, Any] | None = None


def _history_text(messages: list[dict[str, Any]]) -> str:
    """The stored history as the reader's context: dated session headers, then the turns."""
    lines = []
    for message in messages:
        content = str(message.get("content") or "")
        if message.get("role") == "system" and content.startswith(SESSION_DATE):
            lines.append(f"\n[{content}]")
        elif message.get("role") != "system":
            lines.append(content)
    return "\n".join(lines).strip()


def _question(messages: list[dict[str, Any]]) -> tuple[str, str | None]:
    """The question (the last user message) and the date it is asked on, when stated."""
    users = [m for m in messages if m.get("role") == "user"]
    if not users:
        raise HTTPException(400, "a question request needs a user message")
    date = next((str(m["content"])[len(CURRENT_DATE):] for m in messages
                 if m.get("role") == "system" and str(m.get("content", "")).startswith(CURRENT_DATE)),
                None)
    return str(users[-1].get("content") or ""), date


def provider_failure(exc: Exception) -> dict[str, Any]:
    """Why the model call failed, as the runner reads it: who refused, with what, and why.

    ``refused`` when the provider answered with a status -- a spent balance, a bad key, a
    rejected request -- and ``failed`` when it never answered at all.
    """
    status = getattr(exc, "status_code", None)
    return {"code": PROVIDER_REFUSED if isinstance(status, int) else PROVIDER_FAILED,
            "provider": PROVIDER, "status": status, "key_name": KEY_NAME,
            "message": f"{type(exc).__name__}: {exc}"}


def _completion(model: str, content: str) -> dict[str, Any]:
    return {"id": f"chatcmpl-{uuid.uuid4().hex}", "object": "chat.completion",
            "created": int(time.time()), "model": model,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": content}}]}


def create_app(complete: Completer, model: str, prompt: AnswerPrompt) -> FastAPI:
    """The agent as an ASGI app, answering with ``model`` through ``complete`` under ``prompt``."""
    app = FastAPI(title="memrank full-context reference agent")
    sessions: dict[str, list[dict[str, Any]]] = {}
    lock = threading.Lock()

    @app.get("/health")
    def health() -> dict[str, str]:
        """What `start.ready` polls: answers once the server is accepting requests."""
        return {"status": "ok"}

    @app.post("/v1/chat/completions")
    def chat(request: ChatRequest) -> dict[str, Any]:
        if not request.user:
            raise HTTPException(400, "set `user` to the session id; this agent keeps one memory "
                                     "per user and will not guess which one a request belongs to")
        with lock:
            history = sessions.setdefault(request.user, [])
            if (request.metadata or {}).get("memrank_op") == FEED_OP:
                history.extend(request.messages)
                return _completion(model, "")
            context = _history_text(history)
        question, date = _question(request.messages)
        try:
            answer = generate_answer(complete, question=question, context=context, model=model,
                                     prompt=prompt, query_date=date)
        except Exception as exc:  # noqa: BLE001 - reported to the caller, never swallowed
            # A bare 500 would hide why. The provider's own words, and its status when it
            # answered, are what the runner records and explains (memrank.loop.explain).
            raise HTTPException(502, detail=provider_failure(exc)) from exc
        return _completion(model, answer)

    return app
