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
"""A person holding only a result says what kind of number it is and what it is not.

The target's T9 recognition test, and the honesty half of P16: a run over material that
cannot be frozen makes no reproducibility claim, rather than recording a version
indistinguishable from an author who never set one.
"""
from __future__ import annotations

from memrank.evaluation.case import DECIDED_BY_JUDGE, DECIDED_BY_MEMRANK
from memrank.judging.judge import JudgeConfig
from memrank.metrics.headline import WITHHELD
from memrank.quality import UNFREEZABLE
from memrank.runner import run_cell
from tests.fakes import FakeAdapter, FakeBenchmark, JudgeFakeBenchmark, make_fake_completer


class UnfreezableBenchmark(FakeBenchmark):
    """A benchmark whose material cannot be frozen -- a live corpus, say."""

    name = "unfreezable-fake"
    dataset_version = UNFREEZABLE


class WithheldBenchmark(FakeBenchmark):
    """BEAM's shape: the composite is not a self-contained score, so it is withheld."""

    name = "withheld-fake"
    quality_metric = "judged_nugget_rubric"
    composite_rankable = False
    substring_recall_supported = False


def _cell(benchmark=None, **kwargs):
    adapter = FakeAdapter(name="fake", responses={"q1": [], "q2": []})
    return run_cell(adapter, benchmark or FakeBenchmark(), k=10, repeats=1, run_id_prefix="d",
                    model="gpt-4o-mini", token_budget=5000, **kwargs)


def test_the_result_says_what_its_number_is_and_what_it_is_not():
    """Without opening documentation: `substring_recall` is not answer correctness."""
    score = _cell().score

    assert score.of == "retrieval recall (a substring proxy)"
    assert score.not_of == "end-to-end answer correctness"
    assert score.decided_by == DECIDED_BY_MEMRANK
    assert score.value == _cell().composite


def test_a_judged_result_says_a_judge_decided_it():
    bench = JudgeFakeBenchmark()
    docs = {q["text"]: [] for q in bench.load()[0].queries}
    cfg = JudgeConfig(no_context_control=False, completer=make_fake_completer())
    result = run_cell(FakeAdapter("fake", docs), bench, k=5, repeats=1, run_id_prefix="d",
                      model="gpt-4o-mini", token_budget=5000, judge=cfg)

    score = result.score

    assert score.decided_by == DECIDED_BY_JUDGE
    assert "LLM-judged" in score.of
    assert score.coverage is not None


def test_a_withheld_number_says_why_and_still_says_what_it_would_have_been():
    """No number is the honest answer, and it is a different statement from zero."""
    score = _cell(WithheldBenchmark()).score

    assert score.kind == WITHHELD
    assert score.value is None
    assert score.decided_by is None
    assert score.withheld
    assert "rubric" in score.of


def test_a_run_over_unfreezable_material_claims_no_reproducibility():
    result = _cell(UnfreezableBenchmark())

    assert result.reproducible is False
    assert result.receipt["reproducible"] is False
    assert result.receipt["dataset_version"] == UNFREEZABLE


def test_a_run_over_ordinary_material_still_claims_it():
    result = _cell()

    assert result.reproducible is True
    assert result.receipt["reproducible"] is True
