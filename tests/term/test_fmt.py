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
"""Numbers in the units people read -- and the ones deliberately left alone.

A real run printed `ingest total sec 275.204`, `corpus bytes 100844`, `corpus tokens 27708`.
That is 4m 35s, 98.5 KB and 27,708 tokens.

The tests that matter most here are the ones asserting what is NOT converted: a benchmark's
scores are its product, and two runs differing in the fifth decimal are two results.
"""
from __future__ import annotations

import pytest

from memrank.term import fmt


@pytest.mark.parametrize("seconds,expected", [
    (45, "45s"), (275.204, "4m 35s"), (7500, "2h 05m"), (0, "0s"), (None, "—"),
])
def test_durations_read_as_durations(seconds, expected):
    assert fmt.duration(seconds) == expected


@pytest.mark.parametrize("num,expected", [
    (342, "342 B"), (100844, "100.8 KB"), (20156767, "20.2 MB"), (None, "—"),
])
def test_sizes_read_as_sizes(num, expected):
    """Decimal units, matching how S3 and `ls -h` report -- the numbers a reader will compare
    these against."""
    assert fmt.size(num) == expected


def test_counts_are_separated():
    """The digit count is what a reader is judging: 27708 and 277080 look alike for a moment."""
    assert fmt.count(27708) == "27,708"
    assert fmt.count(19) == "19"


def test_a_score_is_never_rounded():
    """THE one that must not get tidier. A composite is the run's product; 0.269737 rendered as
    0.2697 makes two different results look like one."""
    assert fmt.measurement(0.269737) == "0.269737"


def test_a_tiny_price_keeps_its_magnitude():
    """6.615e-05 rounded to cents is zero, and the magnitude IS the information."""
    assert fmt.measurement(6.615e-05) == "6.615e-05"


def test_a_large_measurement_does_not_become_scientific():
    """%g keeps 216.0 as 216, so a latency percentile stays readable beside a price."""
    assert fmt.measurement(216.0) == "216"


def test_kind_decides_the_rendering():
    """The renderer's single entry point. The KIND travels with the value from `cell_metrics`;
    matching on a label string instead would put one fact in two places, and a renamed label
    would silently start printing seconds as bytes."""
    assert fmt.value(275.204, "seconds") == "4m 35s"
    assert fmt.value(100844, "bytes") == "100.8 KB"
    assert fmt.value(27708, "count") == "27,708"
    assert fmt.value(0.269737, "score") == "0.269737"
    assert fmt.value(True, "text") == "yes"
    assert fmt.value(None, "count") == "—"


def test_an_unknown_is_never_a_zero():
    """Absent and zero are different claims -- the distinction this codebase keeps having to
    re-establish, most recently in `token_metrics`."""
    assert fmt.value(None, "seconds") == "—"
    assert fmt.duration(0) == "0s", "a real zero still prints as one"
