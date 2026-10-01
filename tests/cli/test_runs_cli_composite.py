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
"""One run must not read two ways.

`runs ls` renders a score from cells on disk, and this is the half of that guarantee a public
tree can state: the CLI projection, driven cell by cell.

The other two paths are the hosted API's -- it projects a score from the synced record, and when
this machine holds no artifact the listing shows the number the API sent.
`tests/internal/test_runs_cli_composite_api.py` drives the SAME table of cells through those,
importing the helpers below so the two halves cannot drift onto different fixtures.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from memrank.cli.runs import RunRow, _composite
from memrank.metrics.headline import DEFAULT_QUALITY_METRIC, cell_headline
from memrank.metrics.headline import JUDGED as JUDGED_KIND
from memrank.runs.registry import RunInfo

#: A judged cell at full coverage, a judged cell too sparse to rank, and the plain composites.
JUDGED = {"composite": None, "composite_rankable": False,
          "judged_metrics": {"answer_correctness": 0.72, "judged_coverage": 0.9}}
THIN_JUDGE = {"composite": None, "composite_rankable": False,
              "judged_metrics": {"answer_correctness": 0.72, "judged_coverage": 0.3}}
LEGACY_BEAM = {"composite": 0.01, "substring_recall_supported": False}


def _info(cell: dict) -> RunInfo:
    """A registry row built the way ``_read_info`` builds one -- through the projection."""
    head = cell_headline(cell)
    return RunInfo(run_id="r", path=Path("c.json"), adapter="demo", benchmark="demo",
                   config_hash=None, composite=cell.get("composite"),
                   composite_rankable=cell.get("composite_rankable") is not False,
                   timestamp="2026-08-10T00:00:00+00:00",
                   headline=head.value, headline_kind=head.kind, headline_coverage=head.coverage)


def _row(cells, **org) -> RunRow:
    return RunRow(run_id="r", targets=["demo"], eval_name="demo", place="local",
                  state="done", synced="yes", age="1h", progress=None,
                  started_at="2026-08-10T00:00:00+00:00", cells=cells, **org)


#: (cells, what `runs ls` prints, what the API's `_score` returns). One table, both halves.
AGREEMENT = [
    ([{"composite": 0.5}], "0.5000", (0.5, 1)),
    ([{"composite": 0.5}, {"composite": 0.9}], "2 cells", (None, 2)),
    ([], "—", (None, 0)),
    ([{"composite": 0.5, "composite_rankable": False}], "—", (None, 1)),
    # The judged score IS the score: a run whose benchmark reports no composite still has one.
    ([JUDGED], "0.7200 j", (0.72, 1)),
    ([THIN_JUDGE], "0.7200 j", (0.72, 1)),
    # Legacy artifact with no `composite_rankable`: the absent key falls back to
    # `substring_recall_supported`, so a legacy BEAM record is withheld -- the API used to rank it.
    ([LEGACY_BEAM], "—", (None, 1)),
]


@pytest.mark.parametrize("cells,expected_cli,expected_api", AGREEMENT)
def test_the_cli_renders_one_record_the_documented_way(cells, expected_cli, expected_api):
    assert _composite(_row([_info(c) for c in cells])) == expected_cli


#: A row the org listed and this machine holds no cells for, as (the org's fields, the column).
#: An agent run from `memrank run` has one result and no cells, so the API sends its score with
#: `cell_count` 0 (ATO-2365); an engine run's count still decides whether its number is one cell's.
ORG_ONLY = [
    ({"org_kind": "agent", "org_composite": 0.5, "org_cell_count": 0,
      "org_score_kind": JUDGED_KIND}, "0.5000 j"),
    ({"org_kind": "agent", "org_composite": None, "org_cell_count": 0,
      "org_score_kind": None}, "—"),
    ({"org_kind": "engine", "org_composite": 0.5, "org_cell_count": 1,
      "org_score_kind": DEFAULT_QUALITY_METRIC}, "0.5000"),
    ({"org_kind": "engine", "org_composite": None, "org_cell_count": 3,
      "org_score_kind": None}, "3 cells"),
    ({"org_kind": "engine", "org_composite": None, "org_cell_count": 0,
      "org_score_kind": None}, "—"),
]


@pytest.mark.parametrize("org,expected", ORG_ONLY,
                         ids=["agent-judged", "agent-unjudged", "engine-one-cell",
                              "engine-sweep", "engine-empty"])
def test_a_run_listed_only_by_the_org_shows_the_orgs_score(org, expected):
    assert _composite(_row([], **org)) == expected
