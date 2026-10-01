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
"""`memrank.system("word-overlap")`: a shipped system, by the name it is printed under.

The first line a person copies should not have to reach past the seven words into
`memrank.adapters`, and the name it uses should be the one they just saw listed -- so the
catalog names checked here are the registry's own, not a second list beside it.
"""
from __future__ import annotations

import pytest

import memrank
from memrank.adapters import REGISTRY, list_adapters
from memrank.instrument.catalog import evaluation, system
from memrank.instrument.system import kind_of

# These tests exercise the deprecated in-process route on purpose; its notice is expected here.
pytestmark = pytest.mark.filterwarnings(
    "ignore:.*is deprecated and will be removed:DeprecationWarning")


@pytest.mark.parametrize("name", list_adapters())
def test_every_name_the_catalog_prints_constructs(name: str):
    """`memrank list-adapters` is an instruction: every name it prints must build."""
    built = system(name)

    assert isinstance(built, REGISTRY[name])
    assert kind_of(built) == "memory"


def test_the_system_is_reached_by_its_catalog_name_not_its_class_name():
    """The name in the instruction is the catalog's, and the system agrees it is that one."""
    assert system("word-overlap").name == "word-overlap"


def test_options_reach_the_constructor():
    """A client for a running service is given its address the ordinary way."""
    built = system("atomicmemory", base_url="http://localhost:9999")

    assert built.base_url == "http://localhost:9999"


def test_an_unknown_name_is_refused_with_the_known_ones_named():
    """Guessing at a spelling is what happens when the refusal does not say what exists."""
    with pytest.raises(ValueError) as refused:
        system("word-overlaps")

    assert "word-overlaps" in str(refused.value)
    for name in list_adapters():
        assert name in str(refused.value)


def test_the_front_page_reaches_the_same_function():
    """`import memrank` is the whole of what the first line needs."""
    assert memrank.system is system
    assert "system" in memrank.__all__


def test_a_shipped_system_runs_the_shipped_evaluation_end_to_end():
    """The first line of the README, as a test: two names, one number each."""
    result = evaluation("demo").run(system=system("word-overlap"))

    assert result.refusal is None
    assert len(result.traces) == 5
    assert [v.measure for v in result.values_of("demo-score")] == ["demo-score"]
