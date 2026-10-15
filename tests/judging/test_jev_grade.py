"""Grading each benchmark's answers with Jev: the questions sent and the verdict recorded."""

from __future__ import annotations

import json

import httpx
import pytest

from memrank.benchmarks.beam import BEAMBenchmark
from memrank.benchmarks.locomo import LoCoMoBenchmark
from memrank.benchmarks.longmemeval import LongMemEvalBenchmark
from memrank.judging.jev import JevClient, JevJudge
from memrank.judging.jev.grade import EXPERIMENTAL_ORDERING
from memrank.judging.providers import resolve
from memrank.judging.shape import BinaryJudgeShape


def _judge(answer_for):
    """A JevJudge whose server answers each question by ``answer_for(key, question)``."""
    sent: list[dict] = []

    def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        sent.append(body)
        answers = {key: answer_for(key, q) for key, q in body["questions"].items()}
        return httpx.Response(200, json={"model": body["model"], "answers": answers,
                                         "usage": {"input_tokens": 100, "output_tokens": 0}})

    client = JevClient(resolve("typesafe:jev-1.14.0"), "ts", cache_dir=None,
                       transport=httpx.MockTransport(handle))
    return JevJudge(client), sent


def _noul(value):
    return {"type": "noul", "noul": value}


def _correct(key, question):
    return _noul({"states_reference": 0.92, "contradicts_reference": 0.03,
                  "declines": 0.02}[key])


def test_locomo_asks_three_independent_nouls_over_one_state():
    judge, sent = _judge(_correct)
    query = {"text": "When did Ana move?", "category": "single-hop",
             "gold_answers": ["May 2023"]}
    verdict = judge.verdict(LoCoMoBenchmark().judge_shape(), query, "In May 2023.")
    [body] = sent
    assert body["model"] == "jev-1.14.0"
    assert body["state"] == {"question": "When did Ana move?", "reference_answer": "May 2023",
                             "candidate_answer": "In May 2023."}
    assert set(body["questions"]) == {"states_reference", "contradicts_reference", "declines"}
    assert all(q["type"] == "noul" and set(q["criteria"]) == {"true", "false"}
               for q in body["questions"].values())
    assert (verdict["status"], verdict["passed"], verdict["score"]) == ("judged", True, 1.0)
    assert verdict["probability"] == pytest.approx(0.92)


def test_the_verdict_record_stands_in_for_the_missing_rationale():
    judge, _ = _judge(_correct)
    query = {"text": "q", "category": "single-hop", "gold_answers": ["a"]}
    verdict = judge.verdict(LoCoMoBenchmark().judge_shape(), query, "a")
    record = verdict["decision"]
    assert verdict["rationale"].startswith("Jev gives no rationale")
    assert record["rationale"].startswith("none:")
    assert record["model_requested"] == "jev-1.14.0" and record["model"] == "jev-1.14.0"
    assert record["question_set"] and record["questions_sha256"].startswith("sha256:")
    assert record["threshold"] == 0.5 and record["state"]["candidate_answer"] == "a"
    assert record["answers"]["declines"] == {"type": "noul", "noul": 0.02}
    assert record["passing"]["contradicts_reference"] == pytest.approx(0.97)


def test_longmemeval_carries_each_types_rule_into_the_criteria():
    judge, sent = _judge(_correct)
    shape = LongMemEvalBenchmark().judge_shape()
    query = {"text": "How many days?", "category": "temporal-reasoning",
             "judge_prompt_key": "temporal-reasoning", "gold_answers": ["18 days"]}
    judge.verdict(shape, query, "19 days")
    states = sent[0]["questions"]["states_reference"]["criteria"]["true"]
    assert "off-by-one" in states
    preference = {"text": "Recommend a show", "category": "single-session-preference",
                  "judge_prompt_key": "single-session-preference", "gold_answers": ["likes sci-fi"]}
    judge.verdict(shape, preference, "Try The Expanse")
    assert "rubric" in sent[1]["state"] and "reference_answer" not in sent[1]["state"]


@pytest.mark.parametrize(("declines", "passed"), [(0.85, True), (0.15, False)])
def test_a_longmemeval_abstention_passes_by_declining(declines, passed):
    judge, sent = _judge(lambda key, q: _noul(declines))
    query = {"text": "What is my dog's name?", "category": "single-session-user",
             "judge_prompt_key": "abstention", "kind": "negative",
             "gold_answers": ["The user never mentioned a dog."]}
    verdict = judge.verdict(LongMemEvalBenchmark().judge_shape(), query, "I don't know.")
    assert set(sent[0]["questions"]) == {"declines"}
    assert "explanation" in sent[0]["state"]
    assert verdict["passed"] is passed and verdict["probability"] == declines


def test_beam_rubric_is_one_score_per_criterion_mapped_and_averaged():
    levels = {"criterion_0": 1.8, "criterion_1": 1.2, "criterion_2": 0.1}
    judge, sent = _judge(lambda key, q: {"type": "score", "score": levels[key],
                                         "probabilities": {"0": 0.2, "1": 0.3, "2": 0.5}})
    query = {"text": "q", "category": "information_extraction", "kind": "positive",
             "rubric": ["names X", "dates Y", "explains Z"]}
    verdict = judge.verdict(BEAMBenchmark().judge_shape(), query, "an answer")
    assert all(q["type"] == "score" and len(q["criteria"]) == 3
               for q in sent[0]["questions"].values())
    assert sent[0]["state"] == {"question": "q", "response": "an answer"}
    assert verdict["score"] == pytest.approx((1.0 + 0.5 + 0.0) / 3)
    assert verdict["passed"] is False and verdict["probability"] is None
    assert [n["score"] for n in verdict["decision"]["nuggets"]] == [1.0, 0.5, 0.0]
    assert verdict["decision"]["nuggets"][0]["probabilities"]["2"] == 0.5


def _in_order(key, question):
    if key.startswith("order_"):
        first = int(key.split("_")[1])
        shown_first = question["instructions"]["event_a"]
        return {"type": "choice", "choice": "a_first" if shown_first == f"e{first}" else "b_first"}
    return _noul(0.1 if key == "lists_other_events" else 0.9)


def test_beam_event_ordering_is_scored_from_judgements_and_marked_experimental():
    judge, sent = _judge(_in_order)
    query = {"text": "List the events in order", "category": "event_ordering",
             "kind": "positive", "rubric": ["e0", "e1", "e2"]}
    verdict = judge.verdict(BEAMBenchmark().judge_shape(), query, "e0, then e1, then e2")
    questions = sent[0]["questions"]
    assert {k for k in questions if k.startswith("order_")} == {"order_0_1", "order_0_2",
                                                                "order_1_2"}
    assert questions["order_0_1"]["type"] == "choice"
    assert verdict["score"] == 1.0 and verdict["passed"] is True
    assert verdict["decision"]["experimental"] == EXPERIMENTAL_ORDERING
    assert "experimental" in verdict["rationale"]
    assert verdict["decision"]["ordering"]["relations"] == {"0,1": "same", "0,2": "same",
                                                            "1,2": "same"}


def test_an_answer_jev_did_not_give_leaves_the_question_unjudged():
    judge, _ = _judge(lambda key, q: _noul(0.5) if key == "declines" else {"type": "x"})
    query = {"text": "q", "category": "single-hop", "gold_answers": ["a"]}
    verdict = judge.verdict(LoCoMoBenchmark().judge_shape(), query, "a")
    assert verdict["status"] == "unjudged" and "states_reference" in verdict["error"]


def test_a_negative_question_without_per_type_rules_is_refused():
    judge, sent = _judge(_correct)
    shape = BinaryJudgeShape(frozenset({"abstention"}))
    with pytest.raises(ValueError, match="negative"):
        judge.verdict(shape, {"text": "q", "kind": "negative", "gold_answers": ["a"]}, "a")
    assert sent == []
