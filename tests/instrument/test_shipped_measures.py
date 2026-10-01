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
"""Each shipped measure, on traces built by hand.

Built by hand rather than run, because a measure is a rule from traces to values and that is
exactly what has to be checked: the same traces must always give the same values. No timing is
asserted on -- latency is reported, and the assertion here is about which values exist and
what they say they are, never about how fast anything was.
"""
from __future__ import annotations

import pytest

from memrank.contract import Document, Recall
from memrank.instrument.measure import Decider, Scope
from memrank.instrument.measures import FailureRate, Judge, Latency, WordMatch
from memrank.instrument.task import Expected, Task
from memrank.instrument.trace import Answered, Failure, Trace

WHEN = "2026-09-21T10:00:00Z"


def trace(task_id="t1", *, spans=("marine biologist",), polarity="positive", texts=(),
          answered=None, error=None, timings=None, answers=("marine biologist",)) -> Trace:
    return Trace(
        task=Task(id=task_id, prompt="What is Alex?",
                  expected=Expected(answers=answers, required_spans=spans,
                                    forbidden_spans=spans if polarity == "negative" else (),
                                    polarity=polarity)),
        recalled=None if error else Recall(
            documents=[Document(id=f"d{i}", content=text)
                       for i, text in enumerate(texts, start=1)]),
        answered=Answered(text=answered, produced_by="a-writer") if answered else None,
        error=Failure(step="retrieve", message=error) if error else None,
        timings_ms=dict(timings or {}), started=WHEN, finished=WHEN)


def test_word_match_marks_a_span_that_appears_in_a_recalled_document():
    marked = WordMatch().measure([trace(texts=("Alex is a marine biologist.",))], ())

    assert [(v.measure, v.value, v.decider) for v in marked] == [
        ("word-match", 1.0, Decider.RULE)]
    assert "not answer correctness" in marked[0].why


def test_word_match_marks_a_span_that_is_nowhere_as_a_miss_not_as_missing():
    marked = WordMatch().measure([trace(texts=("Alex likes rain.",))], ())

    assert marked[0].value == 0.0 and marked[0].task_id == "t1"


def test_word_match_on_a_negative_task_is_a_hit_when_the_wrong_memory_stayed_away():
    kept_away = WordMatch().measure([trace(polarity="negative", texts=("Alex likes rain.",))], ())
    surfaced = WordMatch().measure(
        [trace(polarity="negative", texts=("Alex is a marine biologist.",))], ())

    assert kept_away[0].value == 1.0 and surfaced[0].value == 0.0


def test_a_task_that_broke_is_measured_as_none_and_never_as_zero():
    marked = WordMatch().measure([trace(error="ConnectionError: refused")], ())

    assert marked[0].value is None
    assert "broke at retrieve" in marked[0].why


def test_failure_rate_counts_the_traces_that_carry_an_error():
    traces = [trace("t1", texts=("x",)), trace("t2", error="boom"), trace("t3", error="boom")]

    measured = FailureRate().measure(traces, ())

    assert measured[0].value == pytest.approx(2 / 3)
    assert measured[0].decider is Decider.MEMRANK and measured[0].task_id is None
    assert "2 of 3" in measured[0].why and "retrieve" in measured[0].why


def test_latency_reports_a_percentile_per_step_with_the_sample_count_it_had():
    traces = [trace("t1", texts=("x",), timings={"retrieve": 10.0, "ingest": 100.0}),
              trace("t2", texts=("x",), timings={"retrieve": 20.0})]

    measured = {v.measure: v for v in Latency().measure(traces, ())}

    assert set(measured) == {"latency.ingest.p50", "latency.ingest.p95",
                             "latency.retrieve.p50", "latency.retrieve.p95"}
    assert measured["latency.retrieve.p95"].value == 20.0
    assert measured["latency.ingest.p50"].value == 100.0
    assert "over 2 sample(s)" in measured["latency.retrieve.p50"].why
    assert Latency().scope is Scope.RUN


def test_a_judge_is_constructible_without_a_key_and_refuses_to_run_without_one(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr("memrank.config.secret", lambda *a, **k: None)
    judge = Judge()

    assert judge.decider is Decider.MODEL and judge.reads == ("answered",)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        judge.measure([trace(answered="a marine biologist")], ())


def test_a_judge_decides_with_a_model_and_records_what_it_said():
    from memrank.judging.judge import JudgeConfig

    asked = []

    def completer(model, system, user):
        asked.append(model)
        return '{"passed": true, "rationale": "the answer names the profession"}'

    judge = Judge(JudgeConfig(completer=completer, cache=False, no_context_control=False))

    values = judge.measure([trace(answered="a marine biologist")], ())

    assert values[0].value is True and values[0].decider is Decider.MODEL
    assert values[0].why == "the answer names the profession"
    assert asked, "the model was actually asked"


def test_a_judge_says_nothing_rather_than_zero_when_nothing_answered():
    from memrank.judging.judge import JudgeConfig

    judge = Judge(JudgeConfig(completer=lambda *a: "{}", cache=False))

    values = judge.measure([trace()], ())

    assert values[0].value is None and values[0].why == "nothing answered this task"
