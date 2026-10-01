"""Every path that generates an answer uses the evaluation's own reader prompt.

The in-process harness (``run_cell`` -> judge stage -> shape), BEAM's rubric shape, and the
full-context reference agent. The engine agent's path is covered in
tests/internal/leaderboard/test_engine_agent.py.
"""

from __future__ import annotations

from dataclasses import replace

from fastapi.testclient import TestClient

from memrank import runner
from memrank.benchmarks.answer_prompts import BEAM_READER, LONGMEMEVAL_READER
from memrank.core import Document
from memrank.judging.judge import JudgeConfig
from memrank.judging.prompts import AnswerPrompt
from memrank.judging.shape import BeamJudgeShape
from memrank.reference.full_context import create_app
from tests.fakes import FakeAdapter, JudgeFakeBenchmark, make_fake_completer

MARKER = "OFFICIAL READER:"
CUSTOM = AnswerPrompt(name="custom", source="test", system="",
                      template=MARKER + " {}\nQ: {}", fields=("context", "question"))


class Recorder:
    """A completer that answers reader calls itself and hands the judge's to the fake."""

    def __init__(self) -> None:
        self.reader_calls: list[tuple[str, str]] = []
        self._judge = make_fake_completer()

    def __call__(self, model: str, system: str, user: str) -> str:
        if not system:
            self.reader_calls.append((system, user))
            return "green tea"
        return self._judge(model, system, user)


class CustomPromptBenchmark(JudgeFakeBenchmark):
    answer_prompt = CUSTOM


def test_the_in_process_harness_answers_with_the_benchmark_s_prompt_and_records_it() -> None:
    bench = CustomPromptBenchmark()
    adapter = FakeAdapter("fake", {q["text"]: [Document(id="d1", content="green tea",
                                                        user_id="u1")]
                                   for q in bench.load()[0].queries})
    recorder = Recorder()
    cell = runner.run_cell(adapter, bench, k=5, repeats=1, run_id_prefix="t", model="gpt-4o-mini",
                           token_budget=5000,
                           judge=JudgeConfig(cache=False, completer=recorder)).to_dict()
    assert recorder.reader_calls, "no answer was generated"
    assert all(user.startswith(MARKER) for _, user in recorder.reader_calls)
    config = cell["receipt"]["config"]
    assert config["answer_prompt"] == CUSTOM.identity
    assert config["answer_prompt_official"] is True


def test_beam_s_rubric_shape_answers_with_the_configured_prompt() -> None:
    recorder = Recorder()
    recorder._judge = lambda model, system, user: (
        '{"score": 1, "rationale": "r"}' if "ONE rubric criterion" in system
        else '{"passed": true, "rationale": "r"}')
    cfg = replace(JudgeConfig(no_context_control=True), answer_prompt=BEAM_READER)
    query = {"id": "q", "text": "What is it?", "category": "information_extraction",
             "rubric": ["names it"]}
    BeamJudgeShape().grade(recorder, cfg, query=query, context="CTX")
    users = [user for _, user in recorder.reader_calls]
    assert len(users) == 2  # the no-context control, then the real answer
    assert all(u.startswith("\nYou are an assistant that MUST answer") for u in users)
    assert "CONTEXT:\nCTX\n" in users[1]


def test_the_full_context_agent_answers_with_its_evaluation_s_prompt() -> None:
    recorder = Recorder()
    client = TestClient(create_app(recorder, "m", LONGMEMEVAL_READER))
    feed = {"model": "m", "user": "s", "metadata": {"memrank_op": "feed"},
            "messages": [{"role": "user", "content": "My dog is Biscuit."}]}
    ask = {"model": "m", "user": "s", "metadata": {"memrank_op": "ask"},
           "messages": [{"role": "system", "content": "Current date: 2023/05/30"},
                        {"role": "user", "content": "Dog's name?"}]}
    assert client.post("/v1/chat/completions", json=feed).status_code == 200
    reply = client.post("/v1/chat/completions", json=ask).json()
    assert reply["choices"][0]["message"]["content"] == "green tea"
    [(system, user)] = recorder.reader_calls
    assert system == ""
    assert user.startswith("I will give you several history chats")
    assert "My dog is Biscuit." in user
    assert user.endswith("Current Date: 2023/05/30\nQuestion: Dog's name?\nAnswer:")
