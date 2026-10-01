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
"""The one rule that decides a cell's headline score, and the guarantee it stays the only one.

Three surfaces each carried their own copy of the legacy rankability default and one had drifted,
so a legacy BEAM record ranked in the GUI while every other surface withheld it. None of them
promoted the judged score, which on locomo, longmemeval and beam is the only quality number a run
produces at all. Both defects were invisible because nothing asserted the surfaces agreed.
"""
from __future__ import annotations

import json

import pytest

from memrank.metrics.headline import JUDGED, WITHHELD, cell_headline, rankable_composite


def _judged(correctness: float, coverage: float) -> dict:
    return {"judged_metrics": {"answer_correctness": correctness, "judged_coverage": coverage}}


def test_the_judged_score_outranks_a_composite_that_is_also_present():
    """A benchmark can report both; the judged number is the protocol's, so it is the score."""
    head = cell_headline({"composite": 0.11, "composite_rankable": True, **_judged(0.72, 1.0)})

    assert (head.value, head.kind, head.rankable) == (0.72, JUDGED, True)


@pytest.mark.parametrize("coverage,rankable", [(1.0, True), (0.5, True), (0.49, False), (0.1, False)])
def test_sparse_judged_coverage_is_published_but_not_rankable(coverage, rankable):
    """Half the questions is the line. Below it the number is real but measures an unstated
    subset, so it may be shown and must not be compared."""
    head = cell_headline(_judged(0.72, coverage))

    assert head.value == 0.72
    assert head.rankable is rankable
    assert head.coverage == coverage


def test_a_judged_run_that_graded_nothing_falls_through_rather_than_reporting_zero():
    """Zero coverage is "the judge never ran on this", not "the engine scored 0"."""
    head = cell_headline({"composite": 0.4, "judged_metrics": {"answer_correctness": 0.0,
                                                               "judged_coverage": 0.0}})

    assert (head.value, head.kind) == (0.4, "substring_recall")


def test_a_benchmark_that_reports_no_composite_is_withheld_not_zero():
    """locomo, longmemeval and beam report `composite: None` without a judge."""
    head = cell_headline({"composite": None, "composite_rankable": False})

    assert (head.value, head.kind, head.rankable) == (None, WITHHELD, False)


@pytest.mark.parametrize("cell,expected", [
    ({"composite_rankable": True}, True),
    ({"composite_rankable": False}, False),
    # Absent: fall back to substring support, which keeps legacy BEAM withheld and legacy
    # locomo shown. The API used to treat absent as rankable and ranked legacy BEAM alone.
    ({"substring_recall_supported": False}, False),
    ({"substring_recall_supported": True}, True),
    ({}, True),
    # A leaderboard row read before validation carries None, meaning "not stated", not "no".
    ({"composite_rankable": None, "substring_recall_supported": False}, False),
])
def test_the_legacy_rankability_default_is_unified(cell, expected):
    assert rankable_composite(cell) is expected


def test_the_projection_never_raises_on_a_cell_it_cannot_read():
    """A listing that cannot project one row must still render the other two hundred."""
    assert cell_headline({}).value is None
    assert cell_headline({"judged_metrics": "not-a-mapping"}).value is None


def test_every_public_surface_reads_the_score_from_this_module(tmp_path, monkeypatch):
    """The enumeration that makes a fourth private copy of the rule impossible to add quietly.

    Each surface is driven with a sentinel projection; one that computed its own answer would
    return its own number instead. When a new surface starts showing a score, it belongs in this
    list -- and if it forgot to delegate, this test is what says so.

    The two internal surfaces -- the hosted API's run scoring and the leaderboard's aggregate --
    are enumerated the same way in `tests/internal/test_headline_surfaces.py`. Split because the
    rule is public and those two consumers are not.
    """
    from memrank import runner
    from memrank.metrics import headline
    from memrank.runs import registry

    sentinel = headline.Headline(value=0.4242, kind=JUDGED, rankable=True, coverage=1.0)
    monkeypatch.setattr(headline, "cell_headline", lambda cell: sentinel)
    cell = {"adapter": "demo", "benchmark": "demo", "composite": 0.1,
            "latency_metrics": {}, "token_metrics": {}, "receipt": {}}
    (tmp_path / "demo__demo.json").write_text(json.dumps(cell), encoding="utf-8")

    assert registry._read_info("r", tmp_path / "demo__demo.json").headline == 0.4242
    assert "0.424" in runner._summary_cell(cell)["recall_display"]
