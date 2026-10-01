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
"""LongMemEval's six official judge prompts, and the dispatch that picks between them.

12% of this benchmark was graded against the wrong OBJECT: its `answer` field is a RUBRIC for the
30 preference items and an EXPLANATION of a false premise for the 30 abstention items, and both
were being fed to a "does this match the reference answer" prompt. A further 43% was graded
without its type's licensed tolerance -- the temporal off-by-one clause (133 questions) and the
knowledge-update supersession clause (78).

These prompts carry the only published human-agreement figure in the repo (97-98% against human
experts, paper Table 6), which is why they are adopted verbatim rather than paraphrased.

Four of five surveyed vendor harnesses ship an abstention branch that never executes; only Mastra
reaches it. The precedence test below is the one that catches that class of bug.
"""
import pytest

from memrank.benchmarks.longmemeval import LongMemEvalBenchmark
from memrank.judging.prompts import LME_PROMPTS
from memrank.judging.shape import BinaryJudgeShape
from tests.benchmarks.test_longmemeval_methodology import _item, _write_fixture


def _query(bench_units, qid):
    return next(u.queries[0] for u in bench_units if u.unit_id == qid)


def _loaded(tmp_path, monkeypatch, items):
    _write_fixture(tmp_path, monkeypatch, items)
    return LongMemEvalBenchmark().load()


def test_all_six_official_prompts_are_declared():
    assert set(LME_PROMPTS) == {
        "single-session-user", "single-session-assistant", "multi-session",
        "temporal-reasoning", "knowledge-update", "single-session-preference", "abstention"}


def test_the_reference_is_labelled_for_what_it_actually_is():
    """Preference ships a rubric and abstention an explanation; neither is an answer."""
    assert LME_PROMPTS["single-session-preference"][1] == "Rubric"
    assert LME_PROMPTS["abstention"][1] == "Explanation"
    assert LME_PROMPTS["multi-session"][1] == "Correct Answer"


def test_the_type_specific_clauses_are_present():
    assert "off-by-one" in LME_PROMPTS["temporal-reasoning"][0]
    assert "previous information along with an updated answer" in \
        LME_PROMPTS["knowledge-update"][0]
    assert "does not need to reflect all the points" in \
        LME_PROMPTS["single-session-preference"][0]
    assert "unanswerable" in LME_PROMPTS["abstention"][0]


def test_the_three_default_types_share_one_prompt():
    """The protocol gives these three the same text; only the other three add a clause."""
    shared = {LME_PROMPTS[t][0] for t in
              ("single-session-user", "single-session-assistant", "multi-session")}
    assert len(shared) == 1
    assert "off-by-one" not in shared.pop()


def test_abstention_overrides_question_type(tmp_path, monkeypatch):
    """The official dispatch checks `_abs` in the question id BEFORE question_type.

    This is the exact case four surveyed vendor harnesses get wrong.
    """
    units = _loaded(tmp_path, monkeypatch, [
        _item("plain", "temporal-reasoning"),
        _item("tricky_abs", "temporal-reasoning"),
    ])

    assert _query(units, "plain")["judge_prompt_key"] == "temporal-reasoning"
    assert _query(units, "tricky_abs")["judge_prompt_key"] == "abstention"
    # The parent type is PRESERVED: the 30 abstention items are counted inside their type
    # bucket as well as reported separately. Abstention is a slice, not a seventh type.
    assert _query(units, "tricky_abs")["category"] == "temporal-reasoning"


def test_abstention_items_skip_the_sufficiency_check(tmp_path, monkeypatch):
    """A question built to have no evidence can only ever answer "no"."""
    units = _loaded(tmp_path, monkeypatch,
                    [_item("a_abs", "multi-session"), _item("b", "multi-session")])

    assert _query(units, "a_abs")["kind"] == "negative"
    assert "kind" not in _query(units, "b")


def test_the_shape_selects_the_declared_prompt(tmp_path, monkeypatch):
    units = _loaded(tmp_path, monkeypatch, [_item("p", "single-session-preference")])
    shape = LongMemEvalBenchmark().judge_shape()

    system, label = shape._prompt_for(_query(units, "p"))

    assert label == "Rubric"
    assert "rubric for desired personalized response" in system


def test_an_undeclared_prompt_key_raises_rather_than_grading_generically():
    shape = BinaryJudgeShape(frozenset({"a"}), prompts=LME_PROMPTS)

    with pytest.raises(ValueError, match="judge prompt"):
        shape._prompt_for({"category": "a", "judge_prompt_key": "not-a-real-key"})


def test_a_shape_without_prompts_is_unchanged():
    """LoCoMo and the demo benchmark must keep grading with the generic pair, byte-identical."""
    assert BinaryJudgeShape(frozenset({"temporal"}))._prompt_for({"category": "temporal"}) is None
