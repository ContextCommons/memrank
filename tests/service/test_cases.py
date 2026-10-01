"""Cases from benchmark units: attributed turns, judgeable questions, seeded sampling."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from memrank.core import BenchmarkUnit, Document
from memrank.definitions import shipped
from memrank.judging.shape import BinaryJudgeShape
from memrank.service import cases
from memrank.service.app import create_app
from memrank.service.engine import EvaluationService
from memrank.service.protocol import AgentRef, RunCreate
from memrank.service.store import FolderRunStore

DOC = Document(id="d1", content="x", timestamp="2023-05-08T13:56:00+00:00",
               messages=[{"role": "user", "content": "Caroline: I went hiking.",
                          "speaker": "Caroline"},
                         {"role": "assistant", "content": "Nice."}])


def test_a_session_keeps_speakers_and_drops_the_rendered_prefix():
    turns = cases.session_of(DOC).turns
    assert [(t.speaker, t.text) for t in turns] == [("Caroline", "I went hiking."),
                                                    ("assistant", "Nice.")]
    assert turns[0].timestamp == DOC.timestamp


class _Bench:
    """A benchmark stand-in: one unit of six judgeable questions and one without gold."""

    name, dataset_version, VERSION = "locomo", "test@v1", 2

    def __init__(self, unit: BenchmarkUnit) -> None:
        self.unit = unit

    def judge_shape(self) -> BinaryJudgeShape:
        return BinaryJudgeShape(frozenset({"a"}))

    def load(self) -> list[BenchmarkUnit]:
        return [self.unit]


def test_only_judgeable_questions_are_asked_and_sampling_is_seeded(monkeypatch):
    queries = [{"id": f"q{i}", "text": "?", "gold_answers": ["x"], "category": "a"}
               for i in range(6)] + [{"id": "nogold", "text": "?", "category": "a"}]
    unit = BenchmarkUnit(unit_id="u", isolation_id="u", documents=[DOC], queries=queries)
    monkeypatch.setattr(shipped, "resolve_eval", lambda ref: (_Bench(unit), "locomo"))
    cases._cached_plan.cache_clear()
    request = RunCreate(evaluation="locomo", agent=AgentRef(name="a"), questions=3, seed=7)
    first = cases.build_plan(request)
    cases._cached_plan.cache_clear()
    again = cases.build_plan(request)
    cases._cached_plan.cache_clear()
    assert first == again and len(first.cases[0].queries) == 3
    assert "nogold" not in [q["id"] for q in first.cases[0].queries]


@pytest.mark.parametrize("evaluation", ["demo", "relation_graph"])
def test_evaluations_that_are_not_answer_judged_are_refused(tmp_path, evaluation):
    service = EvaluationService(store=FolderRunStore(tmp_path))
    response = TestClient(create_app(service)).post(
        "/v1/runs", json={"evaluation": evaluation, "agent": {"name": "a"}})
    assert response.status_code == 422 and "cannot be put to an agent" in response.text
