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
"""`runs show --full` -- what a run actually produced.

A run measures roughly twenty things: quality, latency percentiles, tokens, an estimated cost per
query, and the provenance of the engine that produced them. The listing showed ONE of them, and
the rest were reachable only by opening the artifact by hand -- which is what "I cannot even see
what each result means" meant in practice.

What these pin is not the formatting but the two judgements underneath it: absent is not zero, and
a label must not claim more than its field knows.
"""
from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from memrank.runner import app
from memrank.runs import registry

runner = CliRunner()

ARTIFACT = {
    "adapter": "hindsight", "benchmark": "demo", "composite": 1.0,
    "quality_metric": "substring_recall", "composite_rankable": True,
    "k": 10, "repeats": 3, "n_units": 1,
    "latency_metrics": {"retrieve_p50_ms": 702.7, "retrieve_p95_ms": 1039.6,
                        "retrieve_p99_ms": 1441.2, "ingest_p50_ms": 5525.1,
                        "ingest_p95_ms": 6987.2, "ingest_p99_ms": 7117.1},
    "latency_contended": False,
    "est_dollars_per_query": 2.145e-05, "context_tokens_mean": 143.4,
    "token_metrics": {"tokens_per_ingest_mean": 562.6, "tokens_per_query_mean": 0.0},
    "receipt": {"config_hash": "1eb86e6c", "engine_provenance": {
        "purl": "pkg:oci/hindsight@sha256:abc?arch=amd64",
        "properties": {"memrank:index_digest": "sha256:release",
                       "memrank:platform": "linux/amd64"}}},
}


@pytest.fixture
def run(monkeypatch, tmp_path):
    """One recorded run with one result artifact, in an isolated registry."""
    monkeypatch.setenv("MEMRANK_RUNS_DIR", str(tmp_path))
    directory = tmp_path / "20260804-000000__demo__smoke__aaaaaa"
    directory.mkdir(parents=True)
    (directory / "hindsight__demo.json").write_text(json.dumps(ARTIFACT), encoding="utf-8")
    # Every real run has one; without it a run with an unreadable artifact has nothing left at
    # all, which is a different case from the one under test.
    (directory / "status.json").write_text(json.dumps({
        "run_id": directory.name, "pid": None, "target": "hindsight", "benchmark": "demo",
        "slice": "smoke", "state": "done", "progress": {"pct": 100}, "message": "",
        "started_at": "2026-08-04T00:00:00+00:00", "updated_at": "2026-08-04T00:01:00+00:00",
        "error": None}), encoding="utf-8")
    return directory


def test_the_default_output_is_unchanged(run):
    """A default that shifts is a break for anything reading `runs show` today."""
    plain = runner.invoke(app, ["runs", "show", run.name]).output

    assert "composite=1.0000" in plain
    assert "latency" not in plain and "$/query" not in plain


def test_full_shows_what_the_listing_never_could(run):
    """Quality, latency, cost and provenance -- the four things a result is."""
    output = runner.invoke(app, ["runs", "show", run.name, "--full"]).output

    assert "substring_recall" in output
    assert "703 / 1040 / 1441" in output, "retrieve percentiles, rounded to whole ms"
    assert "5525 / 6987 / 7117" in output, "ingest percentiles"
    assert "2.145e-05" in output, "a cost this small must not be rounded to 0.00"
    assert "pkg:oci/hindsight@sha256:abc" in output


def test_the_platform_and_release_are_shown_where_the_numbers_are(run):
    """The S3 payoff, at the point of use: two runs sharing a release and differing in
    architecture are comparable on quality and not on latency, and a reader can now see which
    case they are in without running compare-versions."""
    output = runner.invoke(app, ["runs", "show", run.name, "--full"]).output

    assert "sha256:release" in output
    assert "linux/amd64" in output


def test_a_metric_the_run_did_not_record_is_omitted_not_zeroed(run):
    """supermemory records no per-query tokens because its API returns no usage. Printing 0.0
    would assert something the artifact does not say."""
    output = runner.invoke(app, ["runs", "show", run.name, "--full"]).output

    assert "tokens/ingest" in output, "this run DID record ingest tokens"
    assert "tokens/query" not in output, "and did not record per-query ones"


def test_the_rankable_label_does_not_overclaim(run):
    """`composite_rankable` is a property of the BENCHMARK -- whether its composite ranks without a
    judge -- not of the number beside it. Labelled "rankable" alone it reads as the latter, which
    is a different question and one this field does not answer."""
    output = runner.invoke(app, ["runs", "show", run.name, "--full"]).output

    assert "benchmark ranks unjudged" in output


def test_an_unparseable_artifact_does_not_take_the_registry_down_with_it(run):
    """It used to raise JSONDecodeError out of every listing that touched the registry -- so ONE
    truncated file made `runs ls`, `runs show` and `ps` fail for every run. A half-written result
    is a normal thing to find on disk; losing the other 235 runs to it is not.
    """
    (run / "hindsight__demo.json").write_text("{ not json", encoding="utf-8")

    shown = runner.invoke(app, ["runs", "show", run.name, "--full"])
    listed = runner.invoke(app, ["runs", "ls"])

    assert shown.exit_code == 0, shown.output
    assert "(none recorded yet)" in shown.output, "the cell is skipped, not shown as an empty row"
    assert listed.exit_code == 0, listed.output


def test_a_missing_artifact_reports_itself_rather_than_the_cell_vanishing(run, monkeypatch):
    """The other direction: a cell the registry knows about whose file is gone by the time --full
    reads it. Reported, so the reader learns which one rather than wondering what is absent."""
    metrics = registry.cell_metrics(run / "nonexistent__demo.json")

    assert "could not be read" in metrics["error"]["artifact"]


def test_reading_a_cell_from_disk_and_from_a_record_give_the_same_reading(run):
    """The terminal opens the artifact; the API reads the synced record out of Postgres. They
    are the same cell and must be the same reading -- the split exists so the browser cannot end
    up with a second, drifting account of a run."""
    from_disk = registry.cell_metrics(run / "hindsight__demo.json")

    assert registry.metrics_from(ARTIFACT) == from_disk


def test_metrics_come_from_the_artifact_not_the_listing():
    """`RunInfo` carries composite and a path, deliberately -- the loader reads little. Everything
    else is read on demand, which is why `--full` is a flag rather than the default."""
    fields = set(registry.RunInfo.__dataclass_fields__)

    assert "latency_metrics" not in fields and "path" in fields
