"""Each grader, table-driven: what it passes, what it fails, and what it says why."""

from __future__ import annotations

import pytest

from memrank.definitions.base import ModelJudge
from memrank.definitions.graders import GraderFailed, choice_index, grader_of, normalise
from memrank.definitions.schema import GraderSpec
from memrank.judging.judge import JudgeConfig
from memrank.service.protocol import QuestionOutcome
from tests.definitions.conftest import FIXTURES


def outcome(answer: str, gold: list[str], **grading) -> QuestionOutcome:
    return QuestionOutcome(case_id="c", question_id="q", question="Which?", reference=gold,
                           category=None, grading={"gold_answers": gold, **grading},
                           status="not_judged", answer=answer)


@pytest.mark.parametrize("text, normalised", [
    ("The Lisbon!", "lisbon"), ("  an   apple ", "apple"), ("New-York", "newyork"),
])
def test_normalising_drops_case_punctuation_articles_and_spacing(text, normalised):
    assert normalise(text) == normalised


GRADED = [
    ("exact", {}, "lisbon.", ["Lisbon"], 1.0),
    ("exact", {}, "It is Lisbon", ["Lisbon"], 0.0),
    ("exact", {}, "Porto", ["Lisbon", "Porto"], 1.0),
    ("numeric", {}, "About 7 nights", ["7"], 1.0),
    ("numeric", {}, "from 1 to 1,234.5", ["1234.5"], 1.0),
    ("numeric", {"tolerance": 0.5}, "7.4", ["7"], 1.0),
    ("numeric", {"tolerance": 0.5}, "7.6", ["7"], 0.0),
    ("numeric", {}, "no idea", ["7"], 0.0),
]


@pytest.mark.parametrize("kind, options, answer, gold, score", GRADED)
def test_deterministic_graders(kind, options, answer, gold, score, tmp_path):
    grader = grader_of(GraderSpec(kind=kind, **options), tmp_path)
    verdict = grader.grade(outcome(answer, gold), None)
    assert (verdict["score"], verdict["passed"]) == (score, score == 1.0) and verdict["rationale"]
    assert not grader.uses_model


CHOICES = ["Sydney", "Canberra", "Melbourne"]


@pytest.mark.parametrize("answer, picked", [
    ("B", 1), ("(b)", 1), ("B. Canberra", 1), ("canberra", 1), ("It's Canberra.", 1),
    ("A dog", None), ("Sydney or Melbourne", None), ("D", None),
])
def test_a_choice_is_read_by_letter_or_by_text(answer, picked):
    assert choice_index(answer, CHOICES) == picked


def test_the_choice_grader_scores_the_pick(tmp_path):
    grader = grader_of(GraderSpec(kind="choice"), tmp_path)
    right = grader.grade(outcome("B", ["Canberra"], choices=CHOICES), None)
    wrong = grader.grade(outcome("Sydney", ["Canberra"], choices=CHOICES), None)
    none = grader.grade(outcome("no idea", ["Canberra"], choices=CHOICES), None)
    assert (right["score"], wrong["score"], none["score"]) == (1.0, 0.0, 0.0)
    assert wrong["rationale"] == "picks A; the answer is B"


def model(reply: str) -> ModelJudge:
    return ModelJudge(complete=lambda model, system, user: reply,
                      cfg=JudgeConfig(cache=False))


@pytest.mark.parametrize("kind, grading, reply, score", [
    ("judge", {}, '{"passed": true, "rationale": "same city"}', 1.0),
    ("judge", {}, '{"passed": false, "rationale": "other city"}', 0.0),
    ("rubric", {"rubric": ["names it", "dates it"]}, '{"score": 0.5, "rationale": "part"}', 0.5),
])
def test_model_graders_ask_the_judge(kind, grading, reply, score, tmp_path):
    grader = grader_of(GraderSpec(kind=kind), tmp_path)
    verdict = grader.grade(outcome("Lisbon", ["Lisbon"], **grading), model(reply))
    assert grader.uses_model and verdict["status"] == "judged" and verdict["score"] == score


def test_an_unreadable_judge_reply_is_recorded_as_unjudged(tmp_path):
    verdict = grader_of(GraderSpec(kind="judge"), tmp_path).grade(outcome("x", ["y"]),
                                                                    model("no json"))
    assert verdict["status"] == "unjudged"


def test_a_command_grader_reads_the_question_on_stdin_and_prints_a_score():
    grader = grader_of(GraderSpec(command=["{python}", "grade_contains.py"]), FIXTURES)
    hit = grader.grade(outcome("I fly to LISBON", ["lisbon"]), None)
    miss = grader.grade(outcome("Porto", ["lisbon"]), None)
    assert (hit["score"], hit["rationale"]) == (1.0, "mentions the reference")
    assert miss["score"] == 0.0 and not grader.uses_model


@pytest.mark.parametrize("program, refusal", [
    ("print('not json')", "it printed 'not json'"),
    ("print('{\"score\": 3}')", "its score 3 is not between 0 and 1"),
])
def test_a_command_grader_that_prints_no_score_stops_judging(tmp_path, program, refusal):
    grader = grader_of(GraderSpec(command=["{python}", "-c", program]), tmp_path)
    with pytest.raises(GraderFailed, match=refusal):
        grader.grade(outcome("x", ["y"]), None)
