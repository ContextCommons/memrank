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
"""Every benchmark query carries the gold answer its dataset actually recorded.

BEAM does not use one field name -- it chooses per ability -- and three of those names were missing
from `_ANSWER_FIELDS`. Four of ten abilities therefore loaded with no gold, which made them
unjudgeable: 160 of 400 queries, surfacing only as `n_unjudged_no_gold` inside the artifact.

The gold answer is what the JUDGE grades a generated answer against. It is no longer also emitted
as a `required_spans` retrieval target -- that repurposing was the substring metric no benchmark
defines, and it is gone from all three external benchmarks.
"""
from __future__ import annotations

import pytest

from memrank.benchmarks.beam import BEAMBenchmark


@pytest.mark.parametrize("field,ability", [
    ("ideal_answer", "contradiction_resolution"),
    ("ideal_summary", "summarization"),
    ("expected_compliance", "instruction_following"),
    ("expected_compliance", "preference_following"),
])
def test_each_ability_finds_the_field_its_dataset_uses(field: str, ability: str):
    assert BEAMBenchmark._extract_answer({field: f"gold for {ability}"}) == f"gold for {ability}"


def test_an_empty_field_does_not_mask_a_populated_one():
    """The original loop returned on the first field NAME present, so an empty `answer` beat a
    populated `ideal_answer` further down the list and the query read as having no gold at all."""
    assert BEAMBenchmark._extract_answer({"answer": "", "ideal_answer": "1,200"}) == "1,200"
    assert BEAMBenchmark._extract_answer({"answer": None, "ideal_summary": "a summary"}) == (
        "a summary")


def test_a_question_with_no_answer_at_all_extracts_nothing():
    """An empty string, not a fabricated one: `is_judgeable` reads this to decide gradeability."""
    assert BEAMBenchmark._extract_answer({"question": "?"}) == ""
    assert BEAMBenchmark._extract_answer({"answer": "   "}) == ""
