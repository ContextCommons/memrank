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
"""BEAM's actual metric: one judge call per rubric criterion, {0, 0.5, 1}, averaged.

We graded BEAM with one holistic verdict against a reference answer we synthesized from nine
candidate field names. That threw away partial credit -- an answer satisfying three of four
criteria was 1 or 0, never 0.75 -- and for `instruction_following` and `preference_following` it
graded against `expected_compliance`, a description of the behaviour expected rather than an
answer, because BEAM ships no answer for those two at all.

The judge is injectable, so all of this runs against a scripted completer: no network, no spend.
"""

import pytest

from memrank.judging.judge import JudgeConfig, grade_nugget, parse_nugget_score
from memrank.judging.prompts import NUGGET_SYSTEM
from memrank.judging.shape import NuggetJudgeShape

_RUBRIC = ["states there is contradictory information",
           "mentions you said you never wrote Flask routes",
           "mentions you also mentioned a homepage route",
           "asks which statement is correct"]


def _completer(scores, *, answer="an answer"):
    """Grades nuggets from `scores` in order; anything else returns a plain answer."""
    it = iter(scores)

    def complete(model, system, user):
        if system == NUGGET_SYSTEM:
            return f'{{"score": {next(it)}, "rationale": "r"}}'
        if "grade whether the provided memory" in system.lower():
            return '{"passed": true, "rationale": "s"}'
        return answer
    return complete


def _query(rubric=None, **kw):
    return {"id": "q1", "text": "did I work with Flask routes?", "category": "contradiction_resolution",
            "rubric": list(_RUBRIC if rubric is None else rubric), **kw}


def test_a_partly_satisfied_rubric_scores_the_mean():
    """The whole point. Three of four criteria is 0.75, where a binary verdict had to pick 1 or 0."""
    cfg = JudgeConfig(samples=1, no_context_control=False)
    # 4 nuggets graded on the real answer.
    jq = NuggetJudgeShape().grade(_completer([1, 1, 1, 0]), cfg, query=_query(), context="ctx")
    assert jq.score == 0.75
    assert [n.score for n in jq.nuggets] == [1.0, 1.0, 1.0, 0.0]
    assert jq.correctness.passed is False, "passed means the WHOLE rubric, not most of it"
    assert "3/4" in jq.correctness.rationale


def test_partial_credit_survives_into_the_score():
    """The 0.5 band exists in BEAM's prompt and its paper; their own harness discards it with
    int(). Ours must not -- see decision-beam-targets-the-spec-not-the-harness."""
    cfg = JudgeConfig(samples=1, no_context_control=False)
    jq = NuggetJudgeShape().grade(_completer([1, 0.5, 0.5, 0]), cfg, query=_query(), context="c")
    assert jq.score == 0.5
    assert "1/4 criteria fully satisfied, 2 partially" in jq.correctness.rationale


def test_a_fully_satisfied_rubric_passes():
    cfg = JudgeConfig(samples=1, no_context_control=False)
    jq = NuggetJudgeShape().grade(_completer([1, 1, 1, 1]), cfg, query=_query(), context="c")
    assert jq.score == 1.0 and jq.correctness.passed is True


def test_abstention_skips_sufficiency_but_is_still_rubric_scored():
    """Its rubric is a prohibition ("should abstain"), which the one grader reads directly -- so
    abstention needs no separate negative prompt, only its sufficiency check skipped."""
    cfg = JudgeConfig(samples=1, no_context_control=False)
    query = _query(rubric=["should abstain or state information is unavailable"],
                   category="abstention", kind="negative")
    jq = NuggetJudgeShape().grade(_completer([1]), cfg, query=query, context="c")
    assert jq.sufficiency is None
    assert jq.score == 1.0


def test_the_cost_estimate_matches_what_grade_actually_spends():
    """A preflight that mis-sizes the shape either refuses a run that would have fitted or admits
    one that dies part-way. Counted against a real call, not asserted from the formula."""
    calls = []
    inner = _completer([1, 1, 1, 1] * 4)

    def counting(model, system, user):
        calls.append(system)
        return inner(model, system, user)

    for control in (True, False):
        calls.clear()
        cfg = JudgeConfig(samples=1, no_context_control=control)
        shape, query = NuggetJudgeShape(), _query()
        shape.grade(counting, cfg, query=query, context="ctx")
        assert len(calls) == shape.calls_per_query(cfg, query), f"no_context_control={control}"


def test_the_scale_is_enforced_rather_than_clamped():
    """A judge that ignored the instruction must not contribute a number."""
    assert parse_nugget_score('{"score": 0.5, "rationale": "r"}') == (0.5, "r")
    for bad in ('{"score": 0.7, "rationale": "r"}', '{"score": true, "rationale": "r"}',
                '{"score": 2, "rationale": "r"}', '{"score": "1", "rationale": "r"}',
                '{"rationale": "r"}', 'not json'):
        with pytest.raises(ValueError):
            parse_nugget_score(bad)


def test_multiple_samples_average_rather_than_vote():
    """The scale is ordinal, so two gradings of 1 and 0.5 are honestly 0.75 -- not whichever won."""
    scored = grade_nugget(_completer([1, 0.5]), question="q", answer="a", nugget="n",
                          model="m", samples=2)
    assert scored.score == 0.75 and scored.samples == 2
