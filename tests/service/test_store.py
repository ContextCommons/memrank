"""A run lives in its folder: a new service over the same folder picks up where it stopped."""

from __future__ import annotations

import json

import pytest

from memrank.service import engine as service_engine
from memrank.service.protocol import AgentRef, RunCreate, StepResult
from memrank.service.store import FolderRunStore, RunNotFound, RunUnreadable
from tests.service.conftest import PLAN

REQUEST = RunCreate(evaluation="locomo", agent=AgentRef(name="a"), judge=False)


def fresh(root, monkeypatch) -> service_engine.EvaluationService:
    monkeypatch.setattr(service_engine, "build_plan", lambda request: PLAN)
    return service_engine.EvaluationService(store=FolderRunStore(root))


def test_progress_survives_a_new_process_over_the_same_folder(tmp_path, monkeypatch):
    first = fresh(tmp_path, monkeypatch)
    run_id = first.create(REQUEST).run_id
    reset = first.next(run_id)
    first.post(run_id, reset.step_id, StepResult(ok=True, elapsed_ms=1))
    again = fresh(tmp_path, monkeypatch)
    assert again.next(run_id).op.value == "feed"
    saved = json.loads((tmp_path / run_id / "run.json").read_text())
    assert saved["request"]["agent"]["name"] == "a"


def test_a_run_id_is_readable_and_names_the_benchmark(tmp_path, monkeypatch):
    run_id = fresh(tmp_path, monkeypatch).create(REQUEST).run_id
    assert "__locomo__" in run_id and (tmp_path / run_id).is_dir()


@pytest.mark.parametrize("bad", ["../x", ".hidden", ""])
def test_an_id_that_would_leave_the_folder_is_refused(tmp_path, bad):
    with pytest.raises(RunNotFound):
        FolderRunStore(tmp_path).load(bad)


def test_a_run_saved_before_lanes_is_refused_with_a_reason(tmp_path, monkeypatch):
    service = fresh(tmp_path, monkeypatch)
    run_id = service.create(REQUEST).run_id
    path = tmp_path / run_id / "run.json"
    saved = json.loads(path.read_text())
    saved["state"] = {"case_ids": ["c1"], "question_counts": [1], "case_index": 0}
    path.write_text(json.dumps(saved))
    with pytest.raises(RunUnreadable, match="start a new run"):
        service.next(run_id)
