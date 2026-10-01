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
"""``--json`` as one convention rather than four accidents.

Three shapes used to coexist -- a bare array without ``sort_keys``, bare objects with it, and no
``--json`` at all on ``runs show``, the record view a script or an agent wants first. What is
pinned here is one convention: detail views emit an object, listings emit an array of objects,
keys are sorted, stdout carries the document alone, and any object may gain keys additively (a
consumer must ignore what it does not know, which is what makes a later ``schema_version``
possible without an envelope now).
"""
from __future__ import annotations

import json

import click
import pytest
from typer.testing import CliRunner

from memrank.runner import app

runner = CliRunner()

ARTIFACT = {
    "adapter": "hindsight", "benchmark": "demo", "composite": 1.0,
    "quality_metric": "substring_recall", "composite_rankable": True,
    "k": 10, "repeats": 3, "n_units": 1,
    "latency_metrics": {"retrieve_p50_ms": 702.7},
    "est_dollars_per_query": 2.145e-05,
    "receipt": {"config_hash": "1eb86e6c"},
}


@pytest.fixture
def run(monkeypatch, tmp_path):
    """One recorded run with one result artifact, in an isolated registry."""
    monkeypatch.setenv("MEMRANK_RUNS_DIR", str(tmp_path))
    directory = tmp_path / "20260804-000000__demo__smoke__aaaaaa"
    directory.mkdir(parents=True)
    (directory / "hindsight__demo.json").write_text(json.dumps(ARTIFACT), encoding="utf-8")
    (directory / "status.json").write_text(json.dumps({
        "run_id": directory.name, "pid": None, "target": "hindsight", "benchmark": "demo",
        "slice": "smoke", "state": "done", "progress": {"pct": 100}, "message": "",
        "started_at": "2026-08-04T00:00:00+00:00", "updated_at": "2026-08-04T00:01:00+00:00",
        "error": None}), encoding="utf-8")
    return directory


def test_runs_show_json_answers_what_the_terminal_form_answers(run):
    """The record view a script reaches for: state, and the cells with their scores."""
    result = runner.invoke(app, ["runs", "show", run.name, "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["run_id"] == run.name
    assert payload["state"] == "done"
    assert payload["results"] == [{"target": "hindsight", "eval": "demo", "composite": 1.0,
                                   "config_hash": "1eb86e6c",
                                   "path": str(run / "hindsight__demo.json"),
                                   # This fixture's receipt predates schema 2, so it recorded no
                                   # environment; `{}` says that, rather than inventing "local".
                                   "environment": {}}]


def test_the_state_word_is_the_one_the_terminal_shows(run):
    """Not the raw store value: the two stores spell one condition differently, and a script
    that had to know which store answered would be reading an implementation detail."""
    shown = runner.invoke(app, ["runs", "show", run.name]).output
    state = json.loads(runner.invoke(app, ["runs", "show", run.name, "--json"]).stdout)["state"]

    assert state in shown


def test_full_carries_each_metric_with_its_kind(run):
    """A consumer that formats seconds as bytes is the bug the kind exists to prevent, so the
    machine form keeps the kind beside the value rather than pre-formatting either."""
    payload = json.loads(
        runner.invoke(app, ["runs", "show", run.name, "--full", "--json"]).stdout)

    metrics = payload["results"][0]["metrics"]
    assert metrics["quality"]["substring_recall"] == {"value": 1.0, "kind": "score"}
    assert metrics["cost (hypothetical prompt cost)"]["$/query (est)"] == {
        "value": 2.145e-05, "kind": "money"}, "a cost this small must not arrive pre-rounded"


def test_without_full_no_metrics_are_read(run):
    """`--full` is a flag because reading every artifact costs; the JSON form honours that."""
    payload = json.loads(runner.invoke(app, ["runs", "show", run.name, "--json"]).stdout)

    assert "metrics" not in payload["results"][0]


def test_an_unknown_run_refuses_rather_than_emitting_an_empty_document(run):
    """An empty object would parse, and a script would read it as "this run has no results"."""
    result = runner.invoke(app, ["runs", "show", "nope", "--json"])

    assert result.exit_code != 0
    assert result.stdout == ""


# --- the convention, across every surface that has --json ---------------------------------------- #

#: Every ``--json`` invocation, with whether it answers with an object (a detail view) or an
#: array (a listing). ENUMERATED so a new one has to state which it is and be checked here.
JSON_SURFACES = (
    (["runs", "ls", "--json"], list),
    (["ps", "--json"], list),
    (["evals", "show", "demo", "--json"], dict),
    (["targets", "show", "word-overlap", "--json"], dict),
)


@pytest.mark.parametrize("argv,shape", JSON_SURFACES, ids=lambda v: str(v))
def test_stdout_carries_the_document_alone(argv, shape, tmp_path, monkeypatch):
    """Notices still narrate -- onto stderr, so they never land inside the parse."""
    monkeypatch.setenv("MEMRANK_RUNS_DIR", str(tmp_path))
    result = runner.invoke(app, argv)

    assert result.exit_code == 0, result.output
    assert isinstance(json.loads(result.stdout), shape)


@pytest.mark.parametrize("argv,shape", JSON_SURFACES, ids=lambda v: str(v))
def test_keys_are_sorted_and_nothing_is_styled(argv, shape, tmp_path, monkeypatch):
    """Sorted so a diff of two documents is a diff of their contents; unstyled because colour
    is presentation and this is data -- even when the caller's terminal says otherwise."""
    monkeypatch.setenv("MEMRANK_RUNS_DIR", str(tmp_path))
    result = runner.invoke(app, argv, color=True)

    assert "\x1b" not in result.stdout
    parsed = json.loads(result.stdout)
    assert result.stdout.strip() == json.dumps(parsed, indent=2, sort_keys=True, default=str)


def test_every_json_flag_is_spelled_and_described_the_same_way():
    """One vocabulary: a surface that invented its own wording for `--json` would read as a
    different feature. Walks the live Typer tree rather than a list of files."""
    helps = {param.help
             for command in _all_commands(app)
             for param in command.params
             if isinstance(param, click.Option) and "--json" in param.opts}

    # A SUBSET, not equality: on the published tree the projection may register no command
    # carrying `--json` at all, and an empty set is that tree agreeing rather than failing.
    # What is pinned is that every `--json` which DOES exist is worded the one way.
    assert helps <= {"machine-readable JSON output"}


def _all_commands(typer_app):
    """Every Click command in the tree, sub-apps included."""
    group = click.Group() if typer_app is None else _as_click(typer_app)
    return list(_walk(group))


def _as_click(typer_app):
    import typer.main

    return typer.main.get_command(typer_app)


def _walk(command):
    if isinstance(command, click.Group):
        for sub in command.commands.values():
            yield from _walk(sub)
    else:
        yield command
