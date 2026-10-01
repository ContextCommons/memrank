"""Bounded local run observation uses durable handles and never follows implicitly."""

from __future__ import annotations

from memrank.application import runs
from memrank.runs import registry
from memrank.runs import status as run_status


def _run(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMRANK_RUNS_DIR", str(tmp_path / "runs"))
    run_dir = registry.new_run_dir("demo", run_id="run-1")
    status = run_status.RunStatus.create(run_dir, target="word-overlap", benchmark="demo",
                                         slice_=None)
    status.update(state="done", message="recorded")
    return run_dir


def test_local_state_logs_and_artifacts_are_bounded(tmp_path, monkeypatch):
    run_dir = _run(tmp_path, monkeypatch)
    (run_dir / "run.log").write_text("one\ntwo\nthree\n", encoding="utf-8")
    (run_dir / "note.txt").write_text("hello", encoding="utf-8")

    observed = runs.get_run("run-1")
    first = runs.get_run_logs("run-1", limit=2)
    artifact = runs.read_run_artifact("run-1", "note.txt", limit=3)

    assert observed.data["status"] == "done"
    assert first.data == {"lines": ["one", "two"], "next_cursor": "2", "complete": True}
    assert artifact.data["content"] == "hel"
    assert artifact.data["next_offset"] == 3


def test_artifact_path_cannot_escape_run(tmp_path, monkeypatch):
    _run(tmp_path, monkeypatch)

    outcome = runs.read_run_artifact("run-1", "../status.json")

    assert outcome.outcome == "refused"
    assert outcome.refusals[0].code == "invalid_artifact"


def test_cancelling_terminal_run_is_idempotent(tmp_path, monkeypatch):
    _run(tmp_path, monkeypatch)

    outcome = runs.cancel_run("run-1")

    assert outcome.outcome == "ok"
    assert outcome.data["cancelled"] is False
