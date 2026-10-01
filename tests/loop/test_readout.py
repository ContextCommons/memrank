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
"""A finished run in plain words (ATO-2378): no "95% CI", no "n=", times a person reads.

``web-gui/tests/agent-run.test.ts`` holds the run page to the same cases, so the page and the
terminal cannot word one result two ways.
"""
from __future__ import annotations

import pytest

from memrank.loop import readout
from memrank.service.protocol import Interval, QuestionOutcome
from memrank.service.scoring import apply_verdicts
from tests.service.test_scoring import FAIL, PASS, recorded


def _question(**over: object) -> QuestionOutcome:
    fields: dict = {"case_id": "c1", "question_id": "q1", "question": "?", "reference": ["x"],
                    "category": None, "status": "judged", "score": 1.0}
    return QuestionOutcome(**{**fields, **over})


@pytest.mark.parametrize(("error", "cause"), [
    ("POST /ask failed after sending: ReadTimeout('timed out')", "timed_out"),
    ("python did not finish in 30.0s", "timed_out"),
    ('POST /ask returned 500: {"detail": "context length exceeded"}', "agent_error"),
    ("POST /ask returned 504: gateway timeout", "agent_error"),
    ("python exited 1: Traceback ...", "agent_error"),
    ("POST /ask failed after sending: RemoteProtocolError('Server disconnected')",
     "unreachable"),
    ("could not connect to http://localhost:8000: [Errno 61] Connection refused",
     "unreachable"),
    ("ask response is not JSON: <html>", "unreadable"),
    ("'answer' found None in the ask response, not a string", "unreadable"),
    ("something else entirely", "other"),
])
def test_each_error_message_falls_in_one_cause(error, cause):
    assert readout.cause_of(error) == cause


def test_failures_are_grouped_by_cause_most_first_quoting_the_commonest_message():
    failed = [_question(status="failed", error=e) for e in
              ["POST /ask failed after sending: ReadTimeout('timed out')"] * 3
              + ["POST /ask returned 500: boom", "POST /ask returned 502: bad gateway",
                 "POST /ask returned 500: boom"]
              + ["could not connect to http://a: refused"]]
    groups = readout.failure_groups([*failed, _question()])
    assert [(g.cause, g.count) for g in groups] == [("agent_error", 3), ("timed_out", 3),
                                                    ("unreachable", 1)]
    assert groups[0].excerpt == "HTTP 500: boom"
    assert [g.words() for g in groups] == ["3 agent errors", "3 timed out",
                                           "1 couldn't reach the agent"]


@pytest.mark.parametrize(("error", "short"), [
    ("POST /v1/chat/completions failed after sending: ReadTimeout('timed out')",
     "ReadTimeout: timed out"),
    ("POST /v1/chat/completions returned 500: {'error': {'message': 'context length "
     "exceeded'}}", "HTTP 500: context length exceeded"),
    ("POST /ask returned 502: bad gateway", "HTTP 502: bad gateway"),
    ("POST /ask failed after sending: RemoteProtocolError('Server disconnected without "
     "sending a response.')", "RemoteProtocolError: Server disconnected without sending ..."),
    ("/usr/local/bin/python3 did not finish in 1.0s", "python3 did not finish in 1.0s"),
    ("something else entirely", "something else entirely"),
])
def test_a_quoted_error_is_the_part_that_says_what_happened(error, short):
    assert readout.short_error(error) == short


@pytest.mark.parametrize(("questions", "kind"), [
    ([_question(grading={"gold_answers": ["x"]}), _question(score=0.0)], readout.RIGHT_OR_WRONG),
    ([_question(grading={"rubric": ["a", "b"]}, score=1.0)], readout.PARTIAL_CREDIT),
    ([_question(grading={"grader": {"kind": "rubric"}})], readout.PARTIAL_CREDIT),
    ([_question(grading={"grader": {"command": ["./grade"]}})], readout.PARTIAL_CREDIT),
    ([_question(grading={"grader": {"kind": "exact"}})], readout.RIGHT_OR_WRONG),
    ([_question(score=0.5)], readout.PARTIAL_CREDIT),
    ([], None),
], ids=["gold", "rubric", "rubric-grader", "program", "exact", "half-score", "no-questions"])
def test_how_a_run_was_graded_is_read_from_its_grading_data(questions, kind):
    assert readout.scoring(questions) == kind


@pytest.mark.parametrize(("ms", "text"), [
    (None, "-"), (850.4, "850 ms"), (7300, "7.3 s"), (11_840, "11.8 s"),
    (209_000, "3 min 29 s"), (3_720_000, "1 h 2 min"),
])
def test_times_read_in_a_persons_units(ms, text):
    assert readout.duration(ms) == text


@pytest.mark.parametrize(("value", "digits", "text"), [
    (0.225, 0, "23%"), (0.395, 1, "39.5%"), (0.0, 0, "0%"), (None, 0, "-"), (0.1135, 1, "11.3%"),
])
def test_percentages_round_half_up_as_the_run_page_does(value, digits, text):
    assert readout.percent(value, digits) == text


def _judged():
    return apply_verdicts(recorded(), {"q1": PASS, "q2": FAIL, "q3": PASS},
                          judge_model="j", judge_samples=1)


def test_the_headline_counts_correct_answers_when_judged_right_or_wrong():
    result = _judged()
    assert readout.headline(result, readout.RIGHT_OR_WRONG) == \
        "50.0% correct: 2 of 4 questions, from 2 conversations"
    assert readout.headline(result, readout.PARTIAL_CREDIT) == \
        "50.0% average score across 4 questions, from 2 conversations"


def test_few_conversations_make_the_range_rough():
    result = _judged().model_copy(update={"score": Interval(mean=0.4, ci95=(0.36, 0.43), n=4)})
    assert readout.range_lines(result) == [
        "36%-43%: on other conversations like these, the score would usually land here",
        "Only 2 conversations: treat this range as rough."]
    many = result.model_copy(update={"cases": [
        c.model_copy(update={"case_id": f"c{i}"}) for i in range(20) for c in result.cases[:1]]})
    assert len(readout.range_lines(many)) == 1


def test_kinds_are_listed_highest_score_first():
    result = _judged().model_copy(update={"per_category": {
        "event_ordering": Interval(mean=0.004, ci95=(0.0, 0.013), n=40),
        "abstention": Interval(mean=0.9, ci95=(0.8, 0.975), n=40)}})
    assert readout.by_kind_lines(result, readout.PARTIAL_CREDIT) == [
        "abstention       90%  40 questions, likely 80%-98%",
        "event ordering    0%  40 questions, likely 0%-1%"]


def test_missing_answers_say_why_or_only_how_many():
    result = _judged()
    groups = readout.failure_groups(q for c in result.cases for q in c.questions)
    assert readout.not_answered(result, groups) == "Agent didn't answer 1 of 4 (1 timed out)"
    assert readout.not_answered(result, None) == "Agent didn't answer 1 of 4"
    assert readout.not_answered(result.model_copy(update={"failed": 0}), None) == "none"


def test_speed_is_typical_and_slow_in_human_units():
    result = _judged().model_copy(update={"latency_ms": {
        "ask_p50": 7300.0, "ask_p95": 11_800.0, "feed_p50": 209_000.0, "feed_p95": None}})
    assert readout.speed(result) == ("answers 7.3 s typical, 11.8 s slow (1 in 20); "
                                     "loading the history 3 min 29 s typical")
    assert readout.speed(result.model_copy(update={"latency_ms": {}})) == \
        "not measured: no step was timed"
