"""A service over a two-case plan, judged by a fake that passes answers containing "blue"."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from memrank.service import app as service_app
from memrank.service import engine as service_engine
from memrank.service.cases import Case, Plan
from memrank.service.protocol import Session, Turn
from memrank.service.store import FolderRunStore

PASS = '{"passed": true, "rationale": "matches"}'
FAIL = '{"passed": false, "rationale": "differs"}'


def query(qid: str, category: str) -> dict:
    return {"id": qid, "text": f"question {qid}?", "gold_answers": ["blue"],
            "category": category, "query_timestamp": "2023-05-08T00:00:00+00:00"}


PLAN = Plan(
    evaluation="locomo", dataset_version="test@v1", task_version=2,
    cases=(
        Case(id="c1", sessions=(Session(id="c1_s1", timestamp="2023-05-01", turns=[
            Turn(role="user", speaker="Ann", text="My car is blue.")]),),
             queries=(query("c1_q0", "single-hop"), query("c1_q1", "temporal"))),
        Case(id="c2", sessions=(Session(id="c2_s1", turns=[
            Turn(role="user", speaker="Bo", text="Hi.")]),),
             queries=(query("c2_q0", "single-hop"),)),
    ))


class FakeJudge:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, model: str, system: str, user: str) -> str:
        self.calls += 1
        return PASS if "blue" in user.rsplit("Candidate answer", 1)[-1] else FAIL


@pytest.fixture
def judge() -> FakeJudge:
    return FakeJudge()


@pytest.fixture
def engine(tmp_path, monkeypatch) -> service_engine.EvaluationService:
    """The service in process, over a run folder in tmp_path, with the two-case plan."""
    monkeypatch.setattr(service_engine, "build_plan", lambda request: PLAN)
    return service_engine.EvaluationService(store=FolderRunStore(tmp_path / "results"))


@pytest.fixture
def client(engine) -> TestClient:
    """The same service over HTTP."""
    return TestClient(service_app.create_app(engine))


def create(client: TestClient) -> str:
    response = client.post("/v1/runs", json={"evaluation": "locomo", "agent": {"name": "a"}})
    assert response.status_code == 200, response.text
    return response.json()["run_id"]


def step(client: TestClient, run_id: str) -> dict:
    response = client.post(f"/v1/runs/{run_id}/next")
    assert response.status_code == 200, response.text
    return response.json()


def post(client: TestClient, run_id: str, step_id: str, **result):
    return client.post(f"/v1/runs/{run_id}/steps/{step_id}", json={"elapsed_ms": 5, **result})
