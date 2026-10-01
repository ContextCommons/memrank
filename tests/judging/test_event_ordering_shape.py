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
"""BEAM grades nine abilities by rubric average and one by rank correlation.

`event_ordering` asks for events in the order the user raised them, so `BeamJudgeShape` dispatches
it to extraction -> alignment -> Kendall tau-b, and leaves the other nine on the rubric grader M3
shipped. The dispatch is the part worth guarding: a change that quietly sent event_ordering through
the nugget path would still produce a number, and that number would call a reversed answer perfect.
"""

from memrank.judging import prompts as jp
from memrank.judging.judge import JudgeConfig
from memrank.judging.shape import BeamJudgeShape

_GOLD = ["core functionality", "transaction errors", "security review"]


def _blocks(user):
    """The sentinel-wrapped payloads of a prompt, in order."""
    return [p.strip() for p in user.split(jp.DATA_SENTINEL)[1::2]]


def _completer(events, *, nugget_score=1):
    """Extracts `events`, and calls two snippets equivalent when their text matches exactly."""
    def complete(model, system, user):
        if system == jp.EVENT_EXTRACTION_SYSTEM:
            return '{"events": [' + ", ".join(f'"{e}"' for e in events) + ']}'
        if system == jp.EQUIVALENCE_SYSTEM:
            reference, candidate = _blocks(user)
            same = reference == candidate
            return '{"passed": %s, "rationale": "r"}' % ("true" if same else "false")
        if system == jp.NUGGET_SYSTEM:
            return f'{{"score": {nugget_score}, "rationale": "r"}}'
        if "grade whether the provided memory" in system.lower():
            return '{"passed": true, "rationale": "s"}'
        return "an answer"
    return complete


def _eo(**kw):
    return {"id": "q1", "text": "list them in order", "category": "event_ordering",
            "rubric": list(_GOLD), **kw}


def _cfg(**kw):
    return JudgeConfig(samples=1, no_context_control=False, **kw)


def test_a_correctly_ordered_answer_scores_one():
    jq = BeamJudgeShape().grade(_completer(_GOLD), _cfg(), query=_eo(), context="ctx")
    assert jq.score == 1.0
    assert jq.correctness.passed is True
    assert [p["matched_reference"] for p in jq.ordering["alignment"]] == _GOLD


def test_a_reversed_answer_scores_zero_though_it_names_every_event():
    """The reason this ability is not nugget-averaged. Coverage is perfect; order is inverted."""
    jq = BeamJudgeShape().grade(_completer(list(reversed(_GOLD))), _cfg(), query=_eo(), context="c")
    assert jq.ordering["f1"] == 1.0
    assert jq.ordering["tau_norm"] == 0.0
    assert jq.score == 0.0


def test_the_other_nine_abilities_still_take_the_rubric_path():
    """A dispatch bug here would be silent: the nugget grader returns a plausible number for an
    ordering question, and that number cannot tell a reversed answer from a correct one."""
    query = {"id": "q2", "text": "q", "category": "contradiction_resolution", "rubric": ["a", "b"]}
    jq = BeamJudgeShape().grade(_completer([], nugget_score=1), _cfg(), query=query, context="c")
    assert jq.ordering is None, "not an ordering question"
    assert jq.nuggets is not None and len(jq.nuggets) == 2
    assert jq.score == 1.0


def test_unmatched_events_are_recorded_as_unmatched():
    jq = BeamJudgeShape().grade(_completer(["core functionality", "something else"]),
                                _cfg(), query=_eo(), context="c")
    pairs = {p["predicted"]: p["matched_reference"] for p in jq.ordering["alignment"]}
    assert pairs["core functionality"] == "core functionality"
    assert pairs["something else"] is None
    assert jq.ordering["precision"] < 1.0


def test_the_cost_estimate_is_an_upper_bound_on_what_grade_spends():
    """Not an equality, unlike every other shape: how many events the model lists is not knowable
    before it answers, so the budget errs high. Erring high refuses a run that would have fitted;
    erring low admits one that dies part-way, having spent everything."""
    def _spend(events, control):
        """Calls actually made, and the estimate, for one (answer, control) combination."""
        calls: list[str] = []
        inner = _completer(events)

        def counting(model, system, user):
            calls.append(system)
            return inner(model, system, user)

        cfg = JudgeConfig(samples=1, no_context_control=control)
        shape, query = BeamJudgeShape(), _eo()
        shape.grade(counting, cfg, query=query, context="ctx")
        return len(calls), shape.calls_per_query(cfg, query)

    for events in ([], _GOLD, list(reversed(_GOLD)), _GOLD * 3):
        for control in (True, False):
            spent, estimated = _spend(events, control)
            assert spent <= estimated, f"{events=} {control=}: spent {spent} > {estimated}"


def test_an_over_long_answer_cannot_blow_up_the_call_count():
    """Alignment is quadratic, so an answer listing fifty items would spend hundreds of calls on a
    response already destined to score near zero. The cap is deliberate; BEAM gets one by accident
    from splitting on newlines."""
    calls = []

    def counting(model, system, user):
        calls.append(system)
        return _completer([f"event {i}" for i in range(50)])(model, system, user)

    shape, query = BeamJudgeShape(), _eo()
    shape.grade(counting, _cfg(), query=query, context="ctx")
    equivalence = sum(1 for s in calls if s == jp.EQUIVALENCE_SYSTEM)
    assert equivalence <= 2 * len(_GOLD) * len(_GOLD), "capped at twice the reference length"
