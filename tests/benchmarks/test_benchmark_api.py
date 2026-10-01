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
"""The benchmark-only library door: load one benchmark and look at it.

`memrank.benchmark("beam:100k-smoke")` is the one-string constructor (same ref grammar
as `run`'s eval parameter); the concrete classes are the keyword form of the same
object. Construction and reprs are I/O-free by contract -- a notebook must be able to
hold and display a BEAM instance without triggering a multi-minute download.
"""
from __future__ import annotations

import pytest

import memrank
from memrank.benchmarks import BEAMBenchmark, DemoBenchmark, from_ref, load_raw
from memrank.core import Benchmark, BenchmarkUnit, Document
from memrank.targets.resolve import RefError


def test_a_ref_constructs_the_configured_class_without_loading():
    beam = memrank.benchmark("beam:100k-smoke")

    assert isinstance(beam, BEAMBenchmark)
    assert (beam.tier, beam.slice) == ("100k", "smoke")


def test_overrides_reach_the_constructor():
    """A knob the ref grammar does not spell (`k`) is still reachable from the ref."""
    assert memrank.benchmark("demo", k=5).k == 5


def test_an_unknown_ref_refuses_and_names_the_variants():
    with pytest.raises(RefError, match="beam:100k-smoke"):
        memrank.benchmark("beam:nope")


def test_the_lazy_export_is_the_registry_factory():
    assert memrank.benchmark is from_ref
    # Off `__all__` since the seven took the front page: `memrank.evaluation("demo")`
    # is what a person reaches for, and this stays the one-string Benchmark constructor.
    assert "benchmark" not in memrank.__all__ and "benchmark" in dir(memrank)


def test_raw_answers_from_the_instance_and_the_name_form_agrees():
    """`bench.raw()` is the primary form; `load_raw(name)` must stay its equal."""
    assert DemoBenchmark().raw() == load_raw("demo")
    assert isinstance(DemoBenchmark().raw()[0], dict)


def test_a_benchmark_with_no_raw_hooks_says_so():
    class Opaque(Benchmark):
        name = "opaque"

        def load(self):  # pragma: no cover - never called
            return []

        def score(self, unit, responses):  # pragma: no cover
            return {}

        def report_template(self):  # pragma: no cover
            return ""

    with pytest.raises(NotImplementedError, match="opaque"):
        Opaque().raw()


# ------------------------------------------------------------------ #
# Reprs: an inventory, never a transcript -- and never a download
# ------------------------------------------------------------------ #

def test_the_benchmark_repr_carries_class_name_and_knobs_without_io():
    class Explosive(BEAMBenchmark):
        def load(self):  # the I/O the repr must never trigger
            raise AssertionError("repr must not load")

        _load_raw = property(lambda self: (_ for _ in ()).throw(AssertionError))

    text = repr(Explosive(tier="500k", slice="mini"))

    assert "Explosive" in text
    assert "'beam'" in text
    assert "tier=500k" in text and "slice=mini" in text


def test_the_unit_repr_is_counts_not_contents():
    secret = "the whole corpus would have been printed here"
    unit = BenchmarkUnit(unit_id="u1", isolation_id="u1",
                         documents=[Document(id="d1", content=secret)],
                         queries=[{"id": "q1"}, {"id": "q2"}])

    text = repr(unit)

    assert text == "BenchmarkUnit('u1', documents=1, queries=2)"
    assert secret not in text


def test_the_document_repr_truncates_and_sizes():
    doc = Document(id="d1", content="x" * 500, user_id="u1")

    text = repr(doc)

    assert "500B" in text and "…" in text
    assert len(text) < 120, "a document repr must fit on a notebook line"


def test_a_short_document_shows_whole_without_ellipsis():
    assert "…" not in repr(Document(id="d1", content="short"))
