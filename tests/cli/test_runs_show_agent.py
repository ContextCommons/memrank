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
"""`runs show` on a run made by `memrank run` reads its result from the org's record (ATO-2365).

What this closes, from a finished `memrank run` run:

    results    in the org, not on this machine -- the org refused them (... (409))

The run's result was in the record the same call had just read; `show` reconciled it as an
engine run's cells instead, which the artifacts route refused for a record with inline answers
and fetched as an unreadable cell for one whose answers are `answers.json`.
"""
from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from memrank import settings
from memrank.cli import runs_show as runs_show_cli
from memrank.placement import run_api_client
from memrank.runner import app
from memrank.service.protocol import ANSWERS_FILE, RunResult, split_answers
from memrank.service.scoring import apply_verdicts
from tests.service.test_scoring import FAIL, PASS, recorded

runner = CliRunner()

RUN_ID = "20260928-010203__locomo__abc123"
PAGE = f"https://gui.example.com/acme/runs/{RUN_ID}"


def _result(judged: bool) -> RunResult:
    """The scoring fixture's run: four questions, one failed, judged or not."""
    result = recorded().model_copy(update={"run_id": RUN_ID})
    if not judged:
        return result
    return apply_verdicts(result, {"q1": PASS, "q2": FAIL, "q3": PASS},
                          judge_model="claude-haiku-4-5", judge_samples=1)


def _platform(record: dict, state: str = "stopped-success") -> dict:
    """The org's single-run read of an agent run, as `GET /orgs/{org}/runs/{id}` answers."""
    return {"id": RUN_ID, "state": state, "kind": "agent", "target_ref": "keyword-agent",
            "benchmark": "locomo", "place": "local", "created_at": "2026-09-28T01:02:03+00:00",
            "artifact": {"bucket": "b", "prefix": f"cloud-runs/{RUN_ID}"}, "url": PAGE,
            "record": {"kind": "agent", **record}}


def _split(result: RunResult) -> tuple[dict, dict]:
    """The record `memrank run` uploads since ATO-2343, and the `answers.json` beside it."""
    summary, cases = split_answers(result)
    return ({"result": summary.model_dump(mode="json"), "answers": ANSWERS_FILE},
            {"cases": [case.model_dump(mode="json") for case in cases]})


class _NullClient:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def downloads(tmp_path, monkeypatch):
    """The names downloaded, from a signed-in org whose artifact listing must never be asked for an agent run's cells."""
    monkeypatch.setenv("MEMRANK_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(settings, "get", lambda key, *a, **k: "acme"
                        if key == "defaults.org" else None)
    monkeypatch.setattr(run_api_client, "authenticated_client", _NullClient)

    def no_cells(*args):
        raise AssertionError("an agent run has no cells to fetch")

    monkeypatch.setattr(run_api_client, "list_artifacts", no_cells)
    names: list[str] = []
    return names


def _show(monkeypatch, platform: dict, *args: str) -> str:
    monkeypatch.setattr(runs_show_cli, "platform_record", lambda run_id: platform)
    result = runner.invoke(app, ["runs", "show", RUN_ID, *args])
    assert result.exit_code == 0, result.output
    return result.output


def _serve(downloads: list[str], monkeypatch, answers: dict) -> None:
    """The org's artifacts route, answering with ``answers`` and noting each name fetched."""
    def download(http, slug, run_id, name, dest, *a, **kw):
        downloads.append(name)
        dest.write_text(json.dumps(answers), encoding="utf-8")
        return dest

    monkeypatch.setattr(run_api_client, "download_artifact", download)


def test_a_judged_run_shows_its_score_categories_failures_judge_and_link(downloads, monkeypatch):
    record, answers = _split(_result(judged=True))
    _serve(downloads, monkeypatch, answers)
    output = _show(monkeypatch, _platform(record))
    assert "50.0% correct: 2 of 4 questions, from 2 conversations" in output
    assert "Only 2 conversations: treat this range as rough." in output
    assert "single-hop   50%  1 of 2 correct" in output and "temporal     50%" in output
    assert "Agent didn't answer 1 of 4 (1 timed out)" in output
    assert "95% CI" not in output and "n=" not in output
    assert "claude-haiku-4-5" in output
    assert PAGE in output
    assert "refused" not in output and "none recorded yet" not in output


def test_an_unjudged_run_says_it_was_not_judged(downloads, monkeypatch):
    record, answers = _split(_result(judged=False))
    _serve(downloads, monkeypatch, answers)
    output = _show(monkeypatch, _platform(record))
    assert "not judged (--no-judge)" in output
    assert "likely range" not in output and "Agent didn't answer 1 of 4" in output
    assert "refused" not in output


def test_full_reads_the_answers_through_the_artifacts_route(downloads, monkeypatch):
    record, answers = _split(_result(judged=True))
    _serve(downloads, monkeypatch, answers)
    output = _show(monkeypatch, _platform(record), "--full")
    assert downloads == [ANSWERS_FILE]
    assert "pass q1" in output and "fail q2" in output
    assert "error  timeout" in output


def test_a_record_with_inline_answers_needs_no_download(downloads, monkeypatch):
    """Every agent run before ATO-2343 carries its questions in the record itself."""
    monkeypatch.setattr(run_api_client, "download_artifact",
                        lambda *a, **kw: pytest.fail("inline answers need no download"))
    inline = {"result": _result(judged=True).model_dump(mode="json")}
    output = _show(monkeypatch, _platform(inline), "--full")
    assert "50.0%" in output and "pass q3" in output


def test_a_run_still_going_has_no_results_yet(downloads, monkeypatch):
    output = _show(monkeypatch, _platform({"progress": {"pct": 40}}, state="running"))
    assert "none recorded yet" in output


def test_json_carries_the_result(downloads, monkeypatch):
    record, _ = _split(_result(judged=True))
    described = json.loads(_show(monkeypatch, _platform(record), "--json"))
    assert described["state"] == "done"
    assert described["agent_result"]["score"]["mean"] == 0.5
    assert described["results"] == []
