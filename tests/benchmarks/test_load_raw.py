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
"""`load_raw` answers for every registered benchmark, or says why not.

The notebooks in notebooks/ are the caller: they show each dataset as its authors published it,
before normalisation flattens and relabels it. That view is reachable only through the private
loader seam on each benchmark class, and five notebooks each reaching into a private attribute
is the arrangement that breaks quietly. `load_raw` is the one seam instead -- so a benchmark that
grows a raw form incompatible with it must fail here rather than in somebody's notebook.

Dispatch is on the attribute a benchmark defines, never its name, which is what these tests
pin: a rename in REGISTRY must not cost a benchmark its raw view.
"""
from __future__ import annotations

import pytest

from memrank.benchmarks import REGISTRY, load_raw

#: Benchmarks whose raw form needs no download and no digest, so this file stays offline-safe.
#: beam/locomo/longmemeval are exercised by their own loader tests, which own the network.
LOCAL_ONLY = ("demo", "relation_graph")


@pytest.mark.parametrize("name", LOCAL_ONLY)
def test_returns_a_list_of_records(name: str) -> None:
    raw = load_raw(name)
    assert isinstance(raw, list) and raw, f"{name}: expected a non-empty list"
    assert all(isinstance(r, dict) for r in raw), f"{name}: records must be dicts"


def test_demo_returns_its_one_scenario_whole() -> None:
    """A single scenario, wrapped in a list -- so every benchmark answers the same call alike."""
    raw = load_raw("demo")
    assert len(raw) == 1
    assert set(raw[0]) == {"unit_id", "isolation_id", "documents", "queries"}


def test_relation_graph_returns_the_in_repo_fixtures() -> None:
    from memrank.benchmarks.relation_graph_fixtures import FIXTURES

    assert [f["id"] for f in load_raw("relation_graph")] == [f["id"] for f in FIXTURES]


def test_slice_kwargs_reach_the_benchmark() -> None:
    """Constructor knobs pass through, so a notebook can ask for a cheap subset."""
    assert len(load_raw("relation_graph", slice="smoke")) == 1
    assert len(load_raw("relation_graph", slice="mini")) == 2


def test_raw_is_not_the_normalised_shape() -> None:
    """The point of the seam: `load()` would have relabelled these into BenchmarkUnits.

    relation_graph's raw fixture states the specification -- expected memories, absences and
    relation edges -- none of which survives into a BenchmarkUnit's public fields.
    """
    fixture = load_raw("relation_graph")[0]
    assert {"expected_memories", "expected_absences", "seed_memories"} <= set(fixture)


def test_unknown_benchmark_reports_what_is_available() -> None:
    with pytest.raises(ValueError, match="Unknown benchmark"):
        load_raw("no_such_benchmark")


def test_every_registered_benchmark_can_be_asked() -> None:
    """No benchmark may be reachable by `load()` and unreachable by `load_raw()`.

    Constructs each one and checks the dispatch attribute WITHOUT calling it. Constructing a
    benchmark is I/O-free by contract -- only `load()` may download -- so this stays offline
    where calling it would fetch three datasets.
    """
    unreachable = [
        name for name, cls in REGISTRY.items()
        if not any(getattr(cls(), attr, None) for attr in ("_load_raw", "_fixtures", "_path"))
    ]
    assert not unreachable, (
        f"{unreachable} define none of _load_raw/_fixtures/_path, so load_raw() raises for them"
    )
