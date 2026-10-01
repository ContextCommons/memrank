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
"""The caveat prose survives being split into parts a result can carry.

`core.QUALITY_METRIC_LABELS` and `QUALITY_METRIC_DESCRIPTIONS` are now derived from
`memrank.quality.QUALITY_METRICS`. Every terminal surface still reads those two dicts, so
their values are pinned here against the literals they replaced -- character for character.
A reworded caveat is a deliberate act and should fail this test.
"""
from __future__ import annotations

from memrank.core import QUALITY_METRIC_DESCRIPTIONS, QUALITY_METRIC_LABELS
from memrank.quality import (
    DATASET_VERSION_UNSET,
    QUALITY_METRICS,
    UNFREEZABLE,
    declaration_for,
    is_reproducible,
)

#: The literals as they read before the split.
_LABELS = {"substring_recall": "recall", "graph_score": "graph score",
           "judged_answer_correctness": "judged correctness",
           "judged_nugget_rubric": "judged rubric"}
_DESCRIPTIONS = {
    "substring_recall": "retrieval recall (a substring proxy), not end-to-end answer correctness",
    "graph_score": "a relation-graph structural score (graph correctness), not retrieval coverage or answer correctness",  # noqa: E501
    "judged_answer_correctness": "LLM-judged answer correctness against the dataset's reference answer (judged by default; `--no-judge` reports no quality score)",  # noqa: E501
    "judged_nugget_rubric": "LLM-judged rubric coverage, one call per atomic nugget (judged by default; `--no-judge` reports no quality score)",  # noqa: E501
}


def test_the_derived_labels_are_the_literals_they_replaced():
    for kind, literal in _LABELS.items():
        assert QUALITY_METRIC_LABELS[kind] == literal


def test_the_derived_descriptions_are_the_literals_they_replaced():
    for kind, literal in _DESCRIPTIONS.items():
        assert QUALITY_METRIC_DESCRIPTIONS[kind] == literal


def test_the_declared_metric_kinds_are_exactly_these():
    assert set(QUALITY_METRICS) == {
        "substring_recall", "graph_score", "judged_answer_correctness",
        "judged_nugget_rubric", "passage_recall",
    }


def test_a_metric_this_memrank_does_not_know_claims_nothing_about_itself():
    """No fallback to a WRONG declaration: an unknown kind is named and disclaimed."""
    declared = declaration_for("someones_own_metric")

    assert "someones_own_metric" in declared.of
    assert "no declaration" in declared.of
    assert declared.not_of


def test_unset_and_unfreezable_are_different_values():
    """One means nobody said; the other means it cannot honestly be said."""
    assert UNFREEZABLE != DATASET_VERSION_UNSET
    assert is_reproducible(DATASET_VERSION_UNSET)
    assert not is_reproducible(UNFREEZABLE)
