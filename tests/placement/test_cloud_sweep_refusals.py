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
"""A refused target must not cost the rest of a cloud sweep its submission.

The regression is real: an 8-target locomo sweep (2026-08-03) submitted two runs, hit a
transient Fargate capacity shortfall on the third, and abandoned the remaining five -- none of
which had anything to do with the refusal. Runs are independent, so they are submitted
independently and the refusals are named together at the end.
"""
from __future__ import annotations

import pytest
from typer.testing import CliRunner

from memrank.placement import run_api_client
from memrank.runner import app

runner = CliRunner()

_TARGETS = "word-overlap,no-context,full-context,fixed-context"


@pytest.fixture
def cloud(tmp_path, monkeypatch):
    """A cloud submit path where the API's answer per target is scripted, nothing launched."""
    from memrank import runner as runner_mod  # noqa: F401 - some stubs still land here
    from memrank.orchestration import cloud as cloud_mod

    monkeypatch.setenv("MEMRANK_RUNS_DIR", str(tmp_path / "runs"))
    state: dict = {"attempted": [], "refuse": {}}

    class _FakeHttp:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def _submit(http, org, payload):
        ref = payload["config"]["target_ref"]
        state["attempted"].append(ref)
        problem = state["refuse"].get(ref)
        if problem is not None:
            raise problem
        rid = f"20260803-120000__demo__{ref.replace(':', '-')}"
        return {"id": rid, "state": "submitted", "taskdef_arn": "arn:td",
                "task_arn": f"arn:aws:ecs:::task/c/{rid}", "cluster": "c",
                "region": "us-east-1", "log_group": "/ecs/c",
                "artifact": {"bucket": "b", "prefix": f"cloud-runs/{rid}"}}

    monkeypatch.setattr(cloud_mod, "_api_client", lambda: _FakeHttp())
    monkeypatch.setattr(run_api_client, "submit_run", _submit)
    # No image-contract stub: a submission names no harness build, so there is no tag to check a
    # checkout against. The server launches its own pinned image.
    return state


def _sweep(*args):
    return runner.invoke(app, ["submit", _TARGETS, "demo", "--on", "cloud",
                               "--org", "acme", *args])


def test_a_refusal_mid_sweep_does_not_stop_the_targets_after_it(cloud):
    cloud["refuse"]["no-context"] = run_api_client.RunApiError(
        "AWS has no Fargate capacity ...", code="capacity_unavailable")

    result = _sweep()

    assert cloud["attempted"] == ["word-overlap", "no-context", "full-context", "fixed-context"]
    assert result.exit_code == 1, "a refused target still fails the command"


def test_the_ids_of_everything_that_launched_still_reach_stdout(cloud):
    cloud["refuse"]["no-context"] = run_api_client.RunApiError("nope", code="capacity_unavailable")

    result = _sweep()

    ids = [ln for ln in result.stdout.splitlines() if ln.startswith("20260803-")]
    assert len(ids) == 3, "three targets launched, so three handles"


def test_the_summary_names_the_refused_the_running_and_the_retry(cloud):
    cloud["refuse"]["no-context"] = run_api_client.RunApiError("no capacity", code="x")
    cloud["refuse"]["fixed-context"] = run_api_client.RunApiError("no capacity", code="x")

    result = _sweep()

    assert "2 of 4 targets were refused: no-context, fixed-context" in result.output
    assert "still running: word-overlap, full-context" in result.output
    # Copy-pasteable means it reproduces the run it is finishing, not just its names -- so it states
    # the measurement in full rather than leaning on whatever this version happens to default to.
    assert "memrank submit no-context,fixed-context demo " in result.output
    assert "--on cloud --org acme" in result.output


def test_the_retry_line_carries_the_flags_that_defined_the_question(cloud):
    """The bug this renderer ends: the line was rebuilt from benchmark, slice and tier, so a sweep
    submitted WITH `--judge` was offered back without it. Pasting it produced unjudged runs -- a
    different measurement, reported as the same one, with no error anywhere.
    """
    cloud["refuse"]["no-context"] = run_api_client.RunApiError("no capacity", code="x")

    result = _sweep("--judge", "--seed", "7", "--repeats", "1")

    retry = next(ln for ln in result.output.splitlines() if "resubmit those:" in ln)
    assert "--judge" in retry
    assert "--seed 7" in retry and "--repeats 1" in retry


def test_no_switch_can_be_dropped_from_the_retry_line_unnoticed(cloud):
    """Enumerated, because omission is silent: nothing fails when a new flag is forgotten."""
    from memrank.placement.remote_argv import REMOTE_SWITCHES

    for ref in _TARGETS.split(","):
        cloud["refuse"][ref] = run_api_client.RunApiError("no capacity", code="x")
    # --verbose is Typer's own flag on this command and --allow-empty-judge-coverage needs a judge;
    # both are switches all the same, so the assertion is over the table, not a hand-picked subset.
    result = _sweep("--judge", "--verbose", "--no-judge-cache",
                    "--allow-empty-judge-coverage", "--fail-fast")

    retry = next(ln for ln in result.output.splitlines() if "resubmit those:" in ln)
    assert [flag for _, flag in REMOTE_SWITCHES if flag not in retry] == []


def test_the_summary_counts_rather_than_repeating_each_reason(cloud):
    """Every refusal already printed its reason as it happened; saying it twice is noise."""
    cloud["refuse"]["no-context"] = run_api_client.RunApiError("zone is full", code="x")

    result = _sweep()

    assert result.output.count("zone is full") == 1


def test_a_sweep_nothing_refuses_says_nothing_extra(cloud):
    result = _sweep()

    assert result.exit_code == 0, result.output
    assert cloud["attempted"] == ["word-overlap", "no-context", "full-context", "fixed-context"]
    assert "refused" not in result.output


def test_every_target_refused_still_reports_rather_than_crashing(cloud):
    for ref in _TARGETS.split(","):
        cloud["refuse"][ref] = run_api_client.RunApiError("no capacity", code="x")

    result = _sweep()

    assert result.exit_code == 1
    assert "4 of 4 targets were refused" in result.output
    assert "still running: none" in result.output
