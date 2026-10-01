"""The HTTP mapping against a recording fake agent, and the openai-chat preset over it."""

from __future__ import annotations

import json

import httpx
import pytest

from memrank.connect.base import (
    AgentError,
    AgentRateLimited,
    AgentUnreachable,
    ConnectorConfigError,
)
from memrank.connect.http import RATE_LIMIT_ATTEMPTS, HttpConnector, HttpSpec
from memrank.connect.presets import preset_mapping
from memrank.rate_limit import RateLimitGate
from memrank.service.protocol import Op, Question, Session, Step, Turn

SESSIONS = [Session(id=f"s{i}", turns=[Turn(role="user", speaker="Ann", text=f"t{i}{j}")
                                       for j in range(2)]) for i in range(2)]
FEED = Step(op=Op.FEED, case_id="c1", session_id="k1", sessions=SESSIONS)
ASK = Step(op=Op.ASK, case_id="c1", session_id="k1", question=Question(id="q", text="Q?"))


def recording(reply: dict | None = None, status: int = 200):
    seen: list[dict] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append({"path": request.url.path, "body": json.loads(request.content or b"null"),
                     "auth": request.headers.get("authorization")})
        return httpx.Response(status, json=reply or {"answer": "A"})

    return seen, httpx.MockTransport(handle)


def connector(transport, **overrides) -> HttpConnector:
    fields = {"base_url": "http://agent", "ask": {"path": "/ask", "answer": "answer",
                                                  "body": {"q": "{question}"}},
              "feed": {"per": "turn", "path": "/m", "body": {"t": "{text}"}}, **overrides}
    return HttpConnector(HttpSpec.model_validate(fields), transport=transport)


@pytest.mark.parametrize(("per", "requests"), [("turn", 4), ("session", 2), ("case", 1)])
def test_feed_sends_one_request_per_unit_of_history(per, requests):
    seen, transport = recording()
    body = {"turn": {"t": "{text}"}, "session": "{messages}", "case": "{transcript}"}[per]
    connector(transport, feed={"per": per, "path": "/m", "body": body}).feed(FEED)
    assert len(seen) == requests


def test_chat_replay_feeds_every_turn_through_the_ask_request():
    seen, transport = recording()
    connector(transport, feed="chat").feed(FEED)
    assert [s["body"]["q"] for s in seen] == ["Ann: t00", "Ann: t01", "Ann: t10", "Ann: t11"]


def test_the_answer_is_extracted_with_jmespath():
    _, transport = recording({"out": [{"text": "blue"}]})
    assert connector(transport, ask={"path": "/a", "answer": "out[0].text"}).ask(ASK) == "blue"


@pytest.mark.parametrize(("reply", "status", "match"), [
    ({"answer": 3}, 200, "not an answer string"), ({"x": 1}, 500, "returned 500")])
def test_a_reached_agent_that_fails_is_an_agent_error(reply, status, match):
    _, transport = recording(reply, status)
    with pytest.raises(AgentError, match=match):
        connector(transport).ask(ASK)


def test_an_error_answer_keeps_its_status_and_whole_body():
    detail = {"detail": {"message": "x" * 1200}}  # far past the old 300-character cut
    _, transport = recording(detail, 502)
    with pytest.raises(AgentError) as failed:
        connector(transport).ask(ASK)
    assert failed.value.status == 502 and failed.value.body == detail
    assert "x" * 1200 in str(failed.value)


class Clock:
    """A gate's clock that only moves when the gate sleeps: every pause is recorded, none waited."""

    def __init__(self) -> None:
        self.now, self.slept = 0.0, []

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def limited_then_ok(refusals: int, retry_after: str | None):
    """An agent that answers 429 ``refusals`` times, then answers; every request counted."""
    sent: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        sent.append(request.url.path)
        if len(sent) <= refusals:
            headers = {"retry-after": retry_after} if retry_after else {}
            return httpx.Response(429, headers=headers, json={"error": "slow down"})
        return httpx.Response(200, json={"answer": "A"})

    return sent, httpx.MockTransport(handle)


@pytest.mark.parametrize(("refusals", "retry_after", "pauses"), [
    (1, "3", [3.0]), (2, None, [1.0, 2.0]),
], ids=["waits-what-it-asked", "backs-off-when-it-names-no-wait"])
def test_a_429_is_waited_out_on_the_shared_gate_then_sent_again(refusals, retry_after, pauses):
    clock = Clock()
    sent, transport = limited_then_ok(refusals, retry_after)
    gate = RateLimitGate(now=lambda: clock.now, sleep=clock.sleep, jitter=lambda: 0.0)
    fields = {"base_url": "http://agent", "feed": "chat",
              "ask": {"path": "/ask", "answer": "answer", "body": {"q": "{question}"}}}
    agent = HttpConnector(HttpSpec.model_validate(fields), transport=transport, gate=gate)
    assert agent.ask(ASK) == "A" and len(sent) == refusals + 1
    assert clock.slept == pauses


def test_a_limit_that_never_lifts_is_the_steps_failure():
    clock = Clock()
    sent, transport = limited_then_ok(99, "1")
    gate = RateLimitGate(now=lambda: clock.now, sleep=clock.sleep, jitter=lambda: 0.0)
    agent = connector(transport)
    agent.gate = gate
    with pytest.raises(AgentRateLimited) as limit:
        agent.ask(ASK)
    assert len(sent) == RATE_LIMIT_ATTEMPTS and limit.value.retry_after_s == 1.0


def test_a_refused_connection_is_unreachable():
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(AgentUnreachable):
        connector(httpx.MockTransport(refuse)).ask(ASK)


def test_auth_is_read_from_the_named_variable(monkeypatch):
    seen, transport = recording()
    monkeypatch.setenv("AGENT_KEY", "k-123")
    connector(transport, auth={"env": "AGENT_KEY"}).ask(ASK)
    assert seen[0]["auth"] == "Bearer k-123"
    monkeypatch.delenv("AGENT_KEY")
    with pytest.raises(ConnectorConfigError, match="AGENT_KEY"):
        connector(transport, auth={"env": "AGENT_KEY"})


def test_the_openai_preset_scopes_every_request_by_session_id():
    seen, transport = recording({"choices": [{"message": {"content": "blue"}}]})
    mapping = preset_mapping("openai-chat", {"model": "m"})
    chat = connector(transport, **mapping, vars={"model": "m"})
    chat.feed(FEED)
    assert chat.ask(ASK) == "blue"
    assert {s["body"]["user"] for s in seen} == {"k1"}
    assert [s["body"]["metadata"]["memrank_op"] for s in seen] == ["feed", "feed", "ask"]


def test_a_preset_names_the_variables_it_needs():
    with pytest.raises(ConnectorConfigError, match="needs model"):
        preset_mapping("openai-chat", {})
