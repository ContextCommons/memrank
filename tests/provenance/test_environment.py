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
"""The environment stamp: recorded, never hashed, and never identifying.

The model's first principle says quality compares across places and latency compares only within
like environments -- which needs the conditions to be a value a consumer can act on. What these
pin is the three properties that make the record trustworthy rather than the field's spelling:
place is DETECTED (a flag threaded through three layers can be wrong; a marker the runtime sets
cannot), the stamp stays OUT of the identity hash, and nothing in it names a person or a machine.
"""
from __future__ import annotations

import json

import pytest

from memrank.provenance import environment
from memrank.provenance.receipt import Receipt, build_receipt


@pytest.fixture(autouse=True)
def no_inherited_markers(monkeypatch):
    """Neither the detection nor its absence may depend on where the suite happens to run."""
    for marker in environment._ECS_MARKERS:
        monkeypatch.delenv(marker, raising=False)


def test_place_is_local_without_the_runtime_markers():
    assert environment.detect().place == environment.LOCAL


@pytest.mark.parametrize("marker", environment._ECS_MARKERS)
def test_any_ecs_marker_makes_the_place_cloud(marker, monkeypatch):
    """Both are checked: the launch-type variable is the direct answer, the metadata URI is
    present on Fargate regardless and catches a task whose launch type reads differently."""
    monkeypatch.setenv(marker, "AWS_ECS_FARGATE")

    assert environment.detect().place == environment.CLOUD


def test_place_is_detected_not_passed(monkeypatch):
    """A cloud run executes this harness INSIDE the task, and `--on` is deliberately not
    forwarded there -- so nothing tells the container where it is. It has to look."""
    monkeypatch.setenv("ECS_CONTAINER_METADATA_URI_V4", "http://169.254.170.2/v4/abc")

    assert environment.detect().place == environment.CLOUD


def test_one_architecture_gets_one_spelling():
    """`aarch64` and `arm64` are the same machine; a receipt and an image purl must agree."""
    assert environment.normalise_arch("aarch64") == "arm64"
    assert environment.normalise_arch("x86_64") == "amd64"
    assert environment.normalise_arch("riscv64") == "riscv64", "unknown stays as reported"


def test_nothing_in_the_stamp_identifies_a_person_or_a_machine():
    """Hostname is gone rather than carried-and-stripped: an Environment is publishable by
    construction, so no downstream surface has to remember to drop a field."""
    import socket

    recorded = json.dumps(environment.detect().as_dict())

    assert socket.gethostname() not in recorded
    assert "hostname" not in recorded


def test_unknown_memory_is_none_not_zero(monkeypatch):
    """A run that did not learn its memory has not got zero of it, and a consumer comparing
    environments must be able to tell "unknown" from "tiny"."""
    def refuse(name):
        raise ValueError("no such configuration name")

    monkeypatch.setattr(environment.os, "sysconf", refuse)

    assert environment.detect().memory_bytes is None


# --- recorded, never hashed ---------------------------------------------------------------------- #

def _receipt(**over):
    return build_receipt(adapter_name="a", adapter_version="1", engine_version="x",
                         benchmark_name="demo", dataset_version="v1", seed=42,
                         config={"k": 10}, **over)


def test_the_receipt_carries_the_stamp():
    assert _receipt().environment["place"] in (environment.LOCAL, environment.CLOUD)


def test_two_runs_differing_only_in_environment_ask_the_same_question(monkeypatch):
    """THE identity property. ``config_hash`` says two runs asked the same thing; if place or
    architecture reached it, the same question asked on a laptop and in the cloud would be two
    questions and could never be compared -- the opposite of what recording it is for.
    """
    local = _receipt()
    monkeypatch.setenv("ECS_CONTAINER_METADATA_URI_V4", "http://169.254.170.2/v4/abc")
    cloud = _receipt()

    assert cloud.environment["place"] == environment.CLOUD
    assert local.environment["place"] == environment.LOCAL
    assert cloud.config_hash == local.config_hash


def test_the_stamp_is_not_inside_the_hashed_config():
    """Belt to the braces above: the config dict itself must not carry it either."""
    assert "environment" not in _receipt().config


def test_a_receipt_written_before_the_stamp_still_loads():
    """Schema 1 recorded a ``host`` blob. Those artifacts are on disk and in S3 today."""
    legacy = _receipt().to_dict()
    del legacy["environment"]
    legacy["host"] = {"hostname": "old-laptop.local", "platform": "macOS-14", "python": "3.12.0"}
    legacy["schema_version"] = 1

    receipt = Receipt.from_dict(legacy)

    assert receipt.environment == {}, "absent, not invented"
    assert receipt.adapter_name == "a"


# --- and something reads it ---------------------------------------------------------------------- #

CLOUD_ENV = {"place": "cloud", "arch": "amd64", "os": "linux", "os_version": "5.10",
             "python": "3.12.9", "cpu_count": 4, "memory_bytes": 8589934592}


@pytest.fixture
def cloud_run(monkeypatch, tmp_path):
    """A recorded run whose artifact says it was measured on Fargate, not on this laptop."""
    monkeypatch.setenv("MEMRANK_RUNS_DIR", str(tmp_path))
    run_dir = tmp_path / "20260805-000000__demo__smoke__aaaaaa"
    run_dir.mkdir(parents=True)
    (run_dir / "hindsight__demo.json").write_text(json.dumps({
        "adapter": "hindsight", "benchmark": "demo", "composite": 1.0,
        "quality_metric": "substring_recall", "composite_rankable": True,
        "latency_metrics": {"retrieve_p50_ms": 702.7},
        "receipt": {"config_hash": "1eb86e6c", "environment": CLOUD_ENV}}), encoding="utf-8")
    (run_dir / "status.json").write_text(json.dumps({
        "run_id": run_dir.name, "pid": None, "target": "hindsight", "benchmark": "demo",
        "slice": "smoke", "state": "done", "progress": {}, "message": "",
        "started_at": "2026-08-05T00:00:00+00:00", "updated_at": "2026-08-05T00:01:00+00:00",
        "error": None}), encoding="utf-8")
    return run_dir


def test_runs_show_names_the_place_in_the_default_view(cloud_run):
    """Not behind `--full`, unlike the metrics: a reader who sees the number must see the
    condition that decides whether it may be set beside another one."""
    from typer.testing import CliRunner

    from memrank.runner import app

    output = CliRunner().invoke(app, ["runs", "show", cloud_run.name]).stdout

    assert "cloud" in output
    assert "amd64/linux" in output


def test_runs_show_json_carries_the_whole_stamp(cloud_run):
    from typer.testing import CliRunner

    from memrank.runner import app

    payload = json.loads(CliRunner().invoke(app, ["runs", "show", cloud_run.name,
                                                  "--json"]).stdout)

    assert payload["results"][0]["environment"] == CLOUD_ENV


def test_the_stamp_describes_where_it_RAN_not_where_it_is_read(cloud_run):
    """The point of the whole field: this artifact was measured on Fargate and is being read on
    a laptop, and it must keep saying Fargate."""
    from memrank.runs import registry

    stamped = registry.cell_environment(cloud_run / "hindsight__demo.json")

    assert stamped["place"] == "cloud"
    assert stamped["arch"] == "amd64"
    assert environment.detect().place == environment.LOCAL, "the reader is not the runner"


def test_the_stamp_reaches_the_org_when_a_run_syncs(cloud_run):
    """`runs sync` uploads the receipt inside each cell; the conditions must ride along, or the
    org's copy of a number loses what makes it comparable."""
    from memrank.runs import record as run_record

    record = run_record.build(cloud_run)

    cell = record["cells"][0]
    assert cell["receipt"]["environment"] == CLOUD_ENV
