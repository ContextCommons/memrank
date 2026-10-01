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
"""Template variables for a step, and the one substitution rule every connector uses.

A template is any JSON-shaped value. A string that is exactly ``"{name}"`` becomes the
variable's value with its type intact -- a list of chat messages stays a list. Any other string
has each ``{name}`` replaced by the value as text. A name no scope defines is refused, so a
typo in an agent spec fails before the run rather than sending ``{questoin}`` to the agent.

Variables, by where they are available:

- every call: ``case_id``, ``session_id``, ``python`` (this interpreter, for agents that are
  Python scripts), plus the spec's own ``vars``
- feed, per turn: ``text``, ``speaker``, ``role``, ``timestamp``
- feed, per session: ``messages`` (chat form), ``transcript``, ``timestamp``
- feed, per case: ``sessions`` (structured), ``transcript``
- ask: ``question``, ``question_id``, ``timestamp``, ``messages`` (chat form)
- ``start.argv`` (:mod:`memrank.connect.process`): ``python`` and ``evaluation`` only
"""

from __future__ import annotations

import re
import sys
from typing import Any

from memrank.connect.base import ConnectorConfigError
from memrank.service.protocol import Session, Step, Turn

_WHOLE = re.compile(r"\{(\w+)\}")
_CHAT_ROLES = ("user", "assistant")

#: The system line the chat forms carry a question's date on. The reference agent reads it.
CURRENT_DATE = "Current date: "
SESSION_DATE = "Session date: "


def render(template: Any, variables: dict[str, Any]) -> Any:
    """``template`` with every ``{name}`` substituted from ``variables``."""
    if isinstance(template, dict):
        return {key: render(value, variables) for key, value in template.items()}
    if isinstance(template, list):
        return [render(item, variables) for item in template]
    if not isinstance(template, str):
        return template
    whole = _WHOLE.fullmatch(template)
    if whole:
        return _lookup(whole.group(1), variables)
    return _WHOLE.sub(lambda m: _text(_lookup(m.group(1), variables)), template)


def _lookup(name: str, variables: dict[str, Any]) -> Any:
    if name not in variables:
        raise ConnectorConfigError(
            f"template names {{{name}}}, which is not available here; available: "
            f"{', '.join(sorted(variables))}")
    return variables[name]


def _text(value: Any) -> str:
    return "" if value is None else str(value)


def _turn_line(turn: Turn) -> str:
    return turn.text if turn.speaker == turn.role else f"{turn.speaker}: {turn.text}"


def transcript(sessions: list[Session]) -> str:
    """Plain text: one line per turn, each session headed by its date when it has one."""
    blocks = []
    for session in sessions:
        head = [f"[{SESSION_DATE}{session.timestamp}]"] if session.timestamp else []
        blocks.append("\n".join(head + [_turn_line(t) for t in session.turns]))
    return "\n\n".join(blocks)


def session_messages(session: Session) -> list[dict[str, str]]:
    """A session as chat messages, dated by a leading system message when it has a date."""
    head = ([{"role": "system", "content": f"{SESSION_DATE}{session.timestamp}"}]
            if session.timestamp else [])
    return head + [{"role": t.role if t.role in _CHAT_ROLES else "user", "content": _turn_line(t)}
                   for t in session.turns]


def base_vars(step: Step, extra: dict[str, Any]) -> dict[str, Any]:
    return {**extra, "case_id": step.case_id, "session_id": step.session_id,
            "python": sys.executable}


def turn_vars(step: Step, turn: Turn, extra: dict[str, Any]) -> dict[str, Any]:
    return {**base_vars(step, extra), "text": turn.text, "speaker": turn.speaker,
            "role": turn.role, "timestamp": turn.timestamp}


def session_vars(step: Step, session: Session, extra: dict[str, Any]) -> dict[str, Any]:
    return {**base_vars(step, extra), "messages": session_messages(session),
            "transcript": transcript([session]), "timestamp": session.timestamp}


def case_vars(step: Step, extra: dict[str, Any]) -> dict[str, Any]:
    sessions = step.sessions or []
    return {**base_vars(step, extra), "transcript": transcript(sessions),
            "sessions": [s.model_dump() for s in sessions]}


def ask_vars(step: Step, extra: dict[str, Any]) -> dict[str, Any]:
    question = step.question
    if question is None:
        raise ValueError("an ask step carries a question")
    head = ([{"role": "system", "content": f"{CURRENT_DATE}{question.timestamp}"}]
            if question.timestamp else [])
    return {**base_vars(step, extra), "question": question.text, "question_id": question.id,
            "timestamp": question.timestamp,
            "messages": head + [{"role": "user", "content": question.text}]}
