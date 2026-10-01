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
"""The leak probe: does the agent keep one session's memory out of another's?

Every case is fed and asked under its own session id. An agent that ignores that scoping would
answer later cases from earlier cases' history, and every score it earns would be invalid. So
before a run, the probe feeds a random canary under one session id and asks for it under a
fresh one. If the canary comes back, the run is refused. No timing is involved: the probe is
two resets, one feed and one ask, in that order.
"""

from __future__ import annotations

import secrets

from memrank.connect.base import Connector
from memrank.errors import MemrankError
from memrank.service.protocol import Op, Question, Session, Step, Turn

PROBE_QUESTION = ("What is the secret code word mentioned earlier in our conversation? "
                  "If none was mentioned, say you don't know.")
#: The date the probe's question is asked on. A question carries one wherever its evaluation
#: dates questions, and a reader prompt that states the current date (LongMemEval's official one)
#: refuses a question without one, so an undated probe failed every such agent before its run.
PROBE_DATE = "2023-05-30T00:00:00+00:00"


class LeakDetected(MemrankError):
    """The agent recalled another session's history. Its results would be invalid."""


def _steps(token: str) -> tuple[Step, Step, Step, Step]:
    planted, asked = f"memrank-probe-{token}-a", f"memrank-probe-{token}-b"
    line = f"Please remember this: the secret code word is {canary(token)}."
    sessions = [Session(id="probe", turns=[Turn(role="user", speaker="user", text=line)])]
    return (Step(op=Op.RESET, case_id=planted, session_id=planted),
            Step(op=Op.FEED, case_id=planted, session_id=planted, sessions=sessions),
            Step(op=Op.RESET, case_id=asked, session_id=asked),
            Step(op=Op.ASK, case_id=asked, session_id=asked,
                 question=Question(id="probe", text=PROBE_QUESTION,
                                   timestamp=PROBE_DATE)))


def canary(token: str) -> str:
    return f"CANARY{token.upper()}"


def leak_probe(connector: Connector, token: str | None = None) -> None:
    """Raise :class:`LeakDetected` when the agent answers one session from another's memory.

    Connector failures propagate: an agent that cannot complete four calls cannot be run.
    """
    token = token or secrets.token_hex(6)
    reset_a, feed_a, reset_b, ask_b = _steps(token)
    connector.reset(reset_a)
    connector.feed(feed_a)
    connector.reset(reset_b)
    answer = connector.ask(ask_b)
    if canary(token).lower() in answer.lower():
        raise LeakDetected(
            "Your agent was told a code word in one conversation and repeated it when asked in a "
            f"different one (it answered: {answer[:200]!r}). It does not keep "
            "conversations apart, so it would be scored on other cases' history, so no question was "
            "asked. Keep its memory separate per conversation, using the session id memrank "
            "sends (the `session_id` template variable).")
