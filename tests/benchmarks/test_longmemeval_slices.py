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
"""LongMemEval slice composition.

`longmemeval_s_cleaned.json` is GROUPED BY QUESTION TYPE -- its run order is
single-session-user, multi-session, single-session-preference, multi-session, temporal-reasoning,
knowledge-update, single-session-assistant. LoCoMo's `raw[:N]` idiom is safe there because its
unit is a conversation carrying every category; here the unit is a question, so `raw[:N]` took
25 consecutive single-session-user items and zero abstention items.

A slice's number is never a score (commit ec852c6), but a smoke slice that exercises one of six
graders does not smoke-test the benchmark -- and after M5 it would not reach the abstention path
at all.
"""
import pytest

from memrank.benchmarks.longmemeval import _QUESTION_TYPES, LongMemEvalBenchmark
from tests.benchmarks.test_longmemeval_methodology import _item, _write_fixture


def _type_grouped_corpus() -> list[dict]:
    """Six types, ten items each, laid out CONSECUTIVELY -- the canonical file's real shape.

    Abstention items go LAST within each type group, which is where they actually sit: on
    `longmemeval_s_cleaned.json` the first `_abs` item of a group is at rank 64 (single-session-
    user), 56 (multi-session), 127 (temporal-reasoning) and 70 (knowledge-update). A fixture that
    put them first would let a type-only stratification pass a test the real data fails.
    """
    items: list[dict] = []
    for qtype in _QUESTION_TYPES:
        for n in range(10):
            qid = f"{qtype[:4]}{n}"
            items.append(_item(f"{qid}_abs" if n >= 8 else qid, qtype))
    return items


@pytest.mark.parametrize("slice_name", ["smoke", "mini"])
def test_slices_cover_every_question_type(tmp_path, monkeypatch, slice_name):
    _write_fixture(tmp_path, monkeypatch, _type_grouped_corpus())

    units = LongMemEvalBenchmark(slice=slice_name).load()
    types = {u.metadata["question_type"] for u in units}

    assert types == set(_QUESTION_TYPES), f"{slice_name} covers only {sorted(types)}"


@pytest.mark.parametrize("slice_name", ["smoke", "mini"])
def test_slices_reach_the_abstention_path(tmp_path, monkeypatch, slice_name):
    """M5 routes `_abs` items to their own judge prompt; a slice with none cannot exercise it.

    This is the test a type-only stratification passes on a flattering fixture and fails on the
    real file, where `_abs` items are the last few of each type group.
    """
    _write_fixture(tmp_path, monkeypatch, _type_grouped_corpus())

    units = LongMemEvalBenchmark(slice=slice_name).load()

    assert any(u.unit_id.endswith("_abs") for u in units)


def test_slice_selection_is_deterministic(tmp_path, monkeypatch):
    _write_fixture(tmp_path, monkeypatch, _type_grouped_corpus())

    first = [u.unit_id for u in LongMemEvalBenchmark(slice="mini").load()]
    second = [u.unit_id for u in LongMemEvalBenchmark(slice="mini").load()]

    assert first == second


def test_a_slice_larger_than_the_corpus_returns_everything(tmp_path, monkeypatch):
    """Fixtures are small; a slice must not raise or pad when it cannot be filled."""
    _write_fixture(tmp_path, monkeypatch, [_item(f"q{t}", t) for t in _QUESTION_TYPES])

    assert len(LongMemEvalBenchmark(slice="mini").load()) == 6
