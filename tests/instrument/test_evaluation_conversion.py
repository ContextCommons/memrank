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
"""`memrank.evaluation("demo")`: an in-tree benchmark, converted into an evaluation.

Every one of the five registered benchmarks must convert. demo and relation_graph are run end
to end here, because they need no download and no credential; locomo, longmemeval and beam are
converted from a fixture unit, which exercises the same code path without fetching a dataset.
"""
from __future__ import annotations

import pytest

from memrank.benchmarks import REGISTRY
from memrank.core import BenchmarkUnit, Document, Recall
from memrank.instrument.catalog import evaluation, from_benchmark
from memrank.instrument.evaluation import Clearing
from memrank.instrument.kinds import Memory
from tests.instrument.fakes import TinyMemory

# These tests exercise the deprecated in-process route on purpose; its notice is expected here.
pytestmark = pytest.mark.filterwarnings(
    "ignore:.*is deprecated and will be removed:DeprecationWarning")


def test_the_demo_benchmark_converts_to_tasks_measures_and_a_clearing_rule():
    converted = evaluation("demo")

    assert converted.name == "demo"
    assert converted.version.startswith("memrank-demo@v1")
    assert converted.clearing is Clearing.PER_GROUP
    assert [m.name for m in converted.measures] == [
        "demo-score", "word-match", "latency", "failure-rate"]
    assert len(converted.tasks) == 5
    assert converted.groups() == ["demo_alex"], "one group per unit"


def test_a_converted_task_carries_the_question_the_expectation_and_the_group_s_documents():
    task = evaluation("demo").tasks[0]

    assert task.prompt == "What is Alex's profession?"
    assert task.expected.answers == ("marine biologist",)
    assert task.expected.required_spans == ("marine biologist",)
    assert task.expected.evidence_doc_ids == ("sess_1",)
    assert task.category == "single-hop"
    assert [d.id for d in task.context] == ["sess_1", "sess_2", "sess_3"]


def test_the_demo_evaluation_runs_end_to_end_and_the_benchmark_s_own_score_agrees():
    result = evaluation("demo").run(system=TinyMemory())

    assert result.refusal is None
    assert len(result.traces) == 5
    marks = [v.value for v in result.values_of("word-match")]
    scored = result.values_of("demo-score")
    assert len(scored) == 1 and scored[0].group == "demo_alex"
    assert scored[0].value == pytest.approx(sum(marks) / len(marks))
    assert "not answer correctness" in scored[0].why


class GraphMemory(Memory):
    """A memory that publishes the graph snapshot a graph benchmark needs to score at all."""

    name = "graph-memory"
    graph_capable = True

    def __init__(self) -> None:
        self.held: list[Document] = []

    def prepare(self, isolation_unit: str) -> None:
        self.held = []

    def ingest(self, documents: list[Document]) -> None:
        self.held.extend(documents)

    def retrieve(self, query: str, k: int, user_id: str, query_timestamp=None):
        words = set(query.lower().split())
        ranked = [d for d in self.held if words & set(d.content.lower().split())][:k]
        snapshot = {"memories": [
            {"provider_memory_id": d.id, "text": d.content, "relations": [],
             "history": [], "is_inference": False} for d in self.held]}
        return Recall(documents=ranked, declared={"graph_snapshot": snapshot})

    def cleanup(self) -> None:
        self.held = []


def test_relation_graph_converts_and_runs_end_to_end_on_a_graph_capable_memory():
    converted = evaluation("relation_graph", slice="ambiguous-update")

    result = converted.run(system=GraphMemory())

    assert result.refusal is None
    assert [t.task_id for t in result.traces] == ["q_planning_preference", "q_demo_preference"]
    assert "word-match" not in {v.measure for v in result.values}, (
        "relation_graph declares the span proxy meaningless, so no WordMatch is shipped")
    scored = result.values_of("relation_graph-score")
    assert len(scored) == 1 and scored[0].group == "relation-ambiguous-update-001"
    assert isinstance(scored[0].value, float)


def test_a_graph_benchmark_handed_no_snapshot_says_so_rather_than_scoring_zero():
    result = evaluation("relation_graph", slice="ambiguous-update").run(system=TinyMemory())

    scored = result.values_of("relation_graph-score")[0]
    assert scored.value is None and "graph_snapshot" in scored.why


#: One fixture unit per benchmark that would otherwise download, in that loader's own shape.
FIXTURE_QUERIES = {
    "locomo": {"id": "s1_q0", "text": "Where does Alex work?", "user_id": "s1",
               "evidence_doc_ids": ["s1_session_1"], "gold_answers": ["a lab"],
               "category": "single-hop", "query_timestamp": "2026-01-01"},
    "longmemeval": {"id": "q1", "text": "When did Alex move?", "user_id": "q1",
                    "gold_ids": ["s1"], "gold_answers": ["April"], "category": "temporal",
                    "judge_prompt_key": "temporal", "evidence_doc_ids": ["s1"],
                    "retrieval_scoreable": True},
    "beam": {"id": "c1_abstention_0", "text": "What is Alex's shoe size?", "user_id": "c1",
             "evidence_doc_ids": ["c1"], "gold_answers": ["not stated"],
             "category": "abstention", "kind": "negative", "rubric": ["declines to answer"],
             "ordering_tested": []},
}


@pytest.mark.parametrize("name", sorted(FIXTURE_QUERIES))
def test_every_downloading_benchmark_converts_through_the_same_path(name, monkeypatch):
    """The conversion, exercised on a fixture unit. Nothing is fetched."""
    query = FIXTURE_QUERIES[name]
    unit = BenchmarkUnit(unit_id="u1", isolation_id=query["user_id"],
                         documents=[Document(id="s1", content="Alex moved in April.")],
                         queries=[query])
    benchmark = REGISTRY[name]()
    monkeypatch.setattr(type(benchmark), "load", lambda self: [unit])

    converted = from_benchmark(benchmark)

    assert len(converted.tasks) == 1
    task = converted.tasks[0]
    assert (task.id, task.prompt, task.group) == (query["id"], query["text"], query["user_id"])
    assert task.expected.answers == tuple(query["gold_answers"])
    assert task.category == query["category"]
    assert [d.id for d in task.context] == ["s1"]
    assert converted.measures[0].name == f"{name}-score"
    assert converted.clearing is Clearing.PER_GROUP


def test_the_conversion_needs_no_download_when_the_units_are_supplied():
    unit = BenchmarkUnit(unit_id="u1", isolation_id="u1", documents=[],
                         queries=[{"id": "q1", "text": "?", "gold_answers": ["yes"]}])
    benchmark = REGISTRY["demo"]()

    converted = from_benchmark(benchmark, [unit], name="demo-fixture")

    assert converted.name == "demo-fixture" and len(converted.tasks) == 1
    assert converted.measures[0].measure([], []) == [], (
        "no traces for that group, so the benchmark is never asked to score one")
