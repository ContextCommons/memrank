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
"""`EvalResult.cases` answers the five questions about one case, judged or not.

The target's recognition test (T12): from a result alone and without opening the source, a
person states for one failing case what was asked, what the engine returned, what answer was
produced (or that none was), what the marking decided, and why -- each by a NAMED field.
Half of that was absent on an unjudged run, which is the shape most people run first, so the
unjudged case is tested here at least as hard as the judged one.

The artifact is not touched. `test_the_case_rows_change_nothing_about_the_artifact` is what
holds that, beside `test_golden_run.py`, which is unmodified.
"""
from __future__ import annotations

import json

from memrank.evaluation.case import (
    CORRECTNESS,
    DECIDED_BY_JUDGE,
    DECIDED_BY_MEMRANK,
    EVIDENCE_FOUND,
    NOT_JUDGED,
    SUFFICIENCY,
)
from memrank.judging.judge import JudgeConfig
from memrank.runner import run_cell
from tests.fakes import FakeAdapter, FakeBenchmark, JudgeFakeBenchmark, make_fake_completer


def _unjudged():
    adapter = FakeAdapter(name="fake", responses={"q1": [], "q2": []})
    return run_cell(adapter, FakeBenchmark(), k=10, repeats=1, run_id_prefix="c",
                    model="gpt-4o-mini", token_budget=5000)


def _judged():
    bench = JudgeFakeBenchmark()
    docs = {q["text"]: [] for q in bench.load()[0].queries}
    cfg = JudgeConfig(no_context_control=False, completer=make_fake_completer())
    return run_cell(FakeAdapter("fake", docs), bench, k=5, repeats=1, run_id_prefix="c",
                    model="gpt-4o-mini", token_budget=5000, judge=cfg)


def test_there_is_one_case_row_per_measured_query_in_order():
    result = _unjudged()

    assert [case.case_id for case in result.cases] == [
        row["query_id"] for row in result.per_query]


def test_an_unjudged_case_answers_all_five_questions_by_name():
    """The whole point of the step: none of the five is absent when no judge ran."""
    case = _unjudged().cases[0]

    assert case.asked                                    # what was asked
    assert case.recalled is not None                     # what the engine returned
    assert case.answered.was_produced is False           # what answer was produced: none
    assert case.answered.not_produced_because == NOT_JUDGED   # ...and why none
    assert case.subscores[EVIDENCE_FOUND].decided_by == DECIDED_BY_MEMRANK  # what marking decided
    assert case.subscores[EVIDENCE_FOUND].why                               # ...and why


def test_an_unjudged_case_carries_no_answer_as_a_value_not_an_absence():
    """T13: a reader must never meet an absence they could take for a failure."""
    case = _unjudged().cases[0]

    assert case.answered is not None
    assert case.answered.text is None
    assert "not judged" in case.answered.not_produced_because
    assert case.answered_without_context is None


def test_a_failing_case_names_what_was_missing():
    result = _unjudged()
    failing = [c for c in result.cases if not c.subscores[EVIDENCE_FOUND].value]

    assert failing, "the fake engine returns nothing, so every case must fail its evidence check"
    assert failing[0].subscores[EVIDENCE_FOUND].missing
    assert "blue whale" in failing[0].subscores[EVIDENCE_FOUND].why


def test_a_judged_case_adds_the_answer_and_the_judge_verdicts():
    cases = {c.case_id: c for c in _judged().cases}
    positive = cases["p1"]

    assert positive.answered.was_produced
    assert positive.answered.produced_by == DECIDED_BY_JUDGE
    assert positive.subscores[CORRECTNESS].decided_by == DECIDED_BY_JUDGE
    assert positive.subscores[CORRECTNESS].why is not None
    # memrank's own deterministic verdict survives the judge, so the two are comparable.
    assert positive.subscores[EVIDENCE_FOUND].decided_by == DECIDED_BY_MEMRANK


def test_the_judge_subscores_are_absent_only_where_the_judge_produced_none():
    """`x1` ships no gold answer, so nothing graded it; it still has memrank's verdict."""
    cases = {c.case_id: c for c in _judged().cases}

    assert CORRECTNESS not in cases["x1"].subscores
    assert EVIDENCE_FOUND in cases["x1"].subscores
    assert cases["x1"].answered.not_produced_because == NOT_JUDGED
    # A negative query has no sufficiency to judge; a positive one does. Both keep the
    # deterministic verdict, so which subscores exist says which markings actually ran.
    assert SUFFICIENCY in cases["p1"].subscores
    assert SUFFICIENCY not in cases["n1"].subscores


def test_recalled_material_is_typed_and_ranked():
    docs = [{"id": "d1", "content": "green tea", "score": 0.9},
            {"id": "d2", "content": "coffee", "score": None}]
    from memrank.evaluation.case import CaseRow

    case = CaseRow.from_artifact({"query_id": "q", "text": "drink?", "retrieved": docs,
                                  "hit": True, "matched_span": "green tea",
                                  "matched_doc_id": "d1"})

    assert [(p.id, p.rank, p.score) for p in case.recalled] == [("d1", 0, 0.9), ("d2", 1, None)]
    assert case.recalled[0].text == "green tea"
    assert "green tea" in case.subscores[EVIDENCE_FOUND].why


def test_the_case_rows_change_nothing_about_the_artifact():
    """Derived, never stored: reading the rows must not alter the dict they were read from."""
    result = _unjudged()
    before = json.dumps(result.to_dict(), indent=2, sort_keys=True, default=str)

    _ = result.cases

    assert json.dumps(result.to_dict(), indent=2, sort_keys=True, default=str) == before
    assert "generated_answer" not in result.to_dict()["per_query"][0]
