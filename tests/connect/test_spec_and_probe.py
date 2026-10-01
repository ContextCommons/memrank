"""Agent specs load into connectors, loudly; the leak probe refuses an agent that shares memory."""

from __future__ import annotations

import hashlib

import pytest

from memrank.connect.base import ConnectorConfigError
from memrank.connect.http import HttpConnector
from memrank.connect.probe import LeakDetected, leak_probe
from memrank.connect.spec import load_agent
from memrank.service.protocol import Step


def write(tmp_path, text: str):
    path = tmp_path / "agent.yaml"
    path.write_text(text)
    return path


def test_a_preset_spec_loads_as_an_http_connector(tmp_path):
    path = write(tmp_path, ("name: full-context\nversion: '1'\nconnector: openai-chat\n"
                            "base_url: http://127.0.0.1:8100\nvars: {model: claude-haiku-4-5}\n"))
    spec = load_agent(path)
    assert (spec.ref.name, spec.ref.version, spec.ref.spec_path) == ("full-context", "1", str(path))
    assert spec.ref.spec_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert isinstance(spec.connector, HttpConnector) and spec.start is None
    assert spec.connector.spec.ask.answer == "choices[0].message.content"


@pytest.mark.parametrize(("text", "match"), [
    ("connector: http\n", "missing 'name'"),
    ("name: a\nconnector: grpc\n", "unknown connector 'grpc'"),
    ("name: a\nconnector: openai-chat\nbase_url: x\nvars: {model: m}\nretries: 3\n", "invalid"),
    ("name: a\nconnector: openai-chat\nbase_url: x\n", "needs model"),
    ("name: a\nconnector: command\nfeed: [x]\nask: [x]\nstart: {argv: [x], ready: u}\n",
     "`start` is for HTTP agents"),
    ("name: a\nconnector: openai-chat\nbase_url: x\nvars: {model: m}\nstart: {argv: [x]}\n",
     "invalid"),
], ids=["no-name", "unknown-kind", "unknown-key", "preset-var", "start-on-command",
        "start-without-ready"])
def test_a_spec_that_cannot_work_is_refused(tmp_path, text, match):
    with pytest.raises(ConnectorConfigError, match=match):
        load_agent(write(tmp_path, text))


class SharedMemory:
    """An agent that ignores session ids: everything it is fed, it remembers for everyone."""

    def __init__(self) -> None:
        self.memory: list[str] = []

    def reset(self, step: Step) -> None:
        pass

    def feed(self, step: Step) -> None:
        self.memory += [t.text for s in step.sessions or [] for t in s.turns]

    def ask(self, step: Step) -> str:
        return " ".join(self.memory)

    def close(self) -> None:
        pass


class Scoped(SharedMemory):
    def ask(self, step: Step) -> str:
        return "I don't know."


def test_the_probe_refuses_an_agent_that_leaks_across_sessions():
    with pytest.raises(LeakDetected, match="does not keep conversations apart"):
        leak_probe(SharedMemory(), token="abc123")


def test_the_probe_passes_an_agent_that_keeps_sessions_apart():
    leak_probe(Scoped(), token="abc123")


def test_the_probe_passes_an_agent_whose_reader_states_the_current_date():
    """LongMemEval's official reader refuses an undated question; the probe's is dated."""
    from memrank.benchmarks.answer_prompts import LONGMEMEVAL_READER

    class DatedReader(Scoped):
        def ask(self, step):
            LONGMEMEVAL_READER.render(question=step.question.text, context="",
                                      query_date=step.question.timestamp)
            return super().ask(step)

    leak_probe(DatedReader(), token="abc123")
