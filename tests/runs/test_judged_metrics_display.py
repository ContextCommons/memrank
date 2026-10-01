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
"""A judged run shows what the judge measured.

`judged_metrics` had been written into artifacts since judging existed and read by nothing:
`cell_metrics` built its quality group from composite, k, repeats and units alone. A judged BEAM
run therefore displayed its 0.065 substring recall -- beside the line "benchmark ranks unjudged:
no" -- while its 0.275 judged correctness sat in the same file, unrendered, and was read as the
engine scoring 0.065.
"""
from __future__ import annotations

import json
from pathlib import Path

from memrank.runs import registry

JUDGED = {
    "answer_correctness": 0.275,
    "answer_correctness_context_dependent": 0.1125,
    "retrieval_sufficiency": 0.325,
    "judged_coverage": 0.5,
    "n_judged": 200, "n_unjudged": 200,
    "n_unjudged_no_gold": 160,
    "n_unjudged_unsupported_category": 40,
    "n_unjudged_unparseable_verdict": 0,
    "unsupported_categories": ["event_ordering"],
}


def _quality(tmp_path: Path, artifact: dict) -> dict:
    path = tmp_path / "myengine__beam.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    return registry.cell_metrics(path)["quality"]


def test_a_judged_run_shows_its_judged_scores(tmp_path: Path):
    quality = _quality(tmp_path, {"composite": 0.065, "judged_metrics": JUDGED})

    assert "0.275" in quality["answer correctness"][0]
    assert quality["answer correctness (context-dependent)"][0] == 0.1125
    assert quality["retrieval sufficiency"][0] == 0.325


def test_coverage_rides_on_the_number_it_qualifies(tmp_path: Path):
    """0.275 over half the queries is not 0.275 over the benchmark. On its own line, coverage is
    an invitation to quote the score without it."""
    quality = _quality(tmp_path, {"composite": 0.065, "judged_metrics": JUDGED})

    assert "200/400 queries judged" in quality["answer correctness"][0]


def test_what_the_judge_skipped_says_why(tmp_path: Path):
    """The two reasons are fixed in different places -- a missing gold answer is a loader defect,
    an unsupported category a judge one -- so a bare count of 200 would not be actionable."""
    quality = _quality(tmp_path, {"composite": 0.065, "judged_metrics": JUDGED})

    assert quality["unjudged"][0] == (
        "160 no gold answer, 40 unsupported category (event_ordering)")


def test_a_fully_judged_run_reports_no_skips(tmp_path: Path):
    judged = {**JUDGED, "n_judged": 400, "n_unjudged": 0, "n_unjudged_no_gold": 0,
              "n_unjudged_unsupported_category": 0, "unsupported_categories": []}
    quality = _quality(tmp_path, {"composite": 0.065, "judged_metrics": judged})

    assert "400/400 queries judged" in quality["answer correctness"][0]
    assert "unjudged" not in quality          # absent, not "0 skipped"


def test_an_unjudged_run_is_unchanged(tmp_path: Path):
    """Every run predating the judge must render exactly as it did -- absent is not zero."""
    quality = _quality(tmp_path, {"composite": 0.065, "k": 10, "n_units": 20})

    assert set(quality) == {"composite", "k", "units"}
