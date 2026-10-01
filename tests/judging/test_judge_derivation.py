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
"""Whether a run judges is the eval's answer, not a flag the user must remember.

`evals show locomo` said `judge: required` while `submit` ran it without one and returned a run
with no quality number at all -- a requirement stated and not enforced, discovered later as a blank
score. On locomo, longmemeval and beam the judge IS the metric, so asking for it was never a
decision; on demo and relation_graph, which score themselves, it still is one.
"""
from __future__ import annotations

import pytest

from memrank.application.types import JudgeSettings
from memrank.benchmarks import judge_required


@pytest.mark.parametrize("name,required", [
    ("locomo", True), ("longmemeval", True), ("beam", True),
    ("demo", False), ("relation_graph", False),
])
def test_the_eval_says_whether_it_needs_a_judge(name, required):
    """Read off the benchmark, so a new one cannot forget to declare it -- `composite_rankable`
    already decides whether an unjudged run of it measures anything."""
    assert judge_required(name) is required


def test_the_catalog_and_the_cli_ask_the_same_function():
    """`evals show` and `submit` disagreeing about whether an eval needs judging is the exact
    defect this replaces: one door stating a requirement the other did not apply."""
    from memrank.application.catalogs import describe_eval

    for name in ("locomo", "demo"):
        stated = describe_eval(name)["judge"] == "required"
        assert stated is judge_required(name), name


def test_an_unset_judge_setting_means_derive_not_off():
    """Omitting the field and sending `false` are different statements. If the default were
    `False`, every browser config that simply did not mention judging would silently opt out."""
    assert JudgeSettings().enabled is None
    assert JudgeSettings(enabled=False).enabled is False
