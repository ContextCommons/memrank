"""Winnability tests for the two relation-graph fixtures that lacked a
correct-graph scoring test: ``same-name-collision`` and ``long-mixed``.

Each test proves the fixture is achievable: a deliberately-wrong graph must
score below 1.0 (so the assertion is meaningful), and the correct graph -- one
that attributes facts to the right entity, supersedes the right history, and
flags the inferred fact with its derives-provenance -- must score a perfect 1.0
with no failures. No scoring logic, thresholds, or expected facts/absences are
touched; correctness lives entirely in the graph/response builders below.
"""

from __future__ import annotations

from memrank.benchmarks.relation_graph import RelationGraphBenchmark
from memrank.core import AdapterResponse, Document


def _doc(
    provider_id: str,
    text: str,
    relations: list[dict[str, str]] | None = None,
    *,
    is_inference: bool = False,
    history: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "provider_memory_id": provider_id,
        "text": text,
        "relations": relations or [],
        "history": history or [],
        "is_inference": is_inference,
        "is_latest": True,
        "is_forgotten": False,
    }


def _collision_graph() -> dict[str, object]:
    """Postgres + Linear attributed to Jordan Lee (Lyra); Rivera stays on Orion."""
    lee = [{"parent_provider_memory_id": "m-lee-seed", "relation": "extends"}]
    return {
        "memories": [
            _doc("m-lee-seed", "Jordan Lee works on Project Lyra."),
            _doc("m-rivera-seed", "Jordan Rivera works on Project Orion."),
            _doc("m-pg", "Jordan Lee uses Postgres as the local cache for Project Lyra.", lee),
            _doc("m-linear", "Jordan Lee tracks Project Lyra stale-chunk bugs in Linear.", lee),
            _doc("m-rivera-only", "Jordan Rivera works only on Project Orion."),
        ]
    }


def _collision_responses(graph: dict[str, object]) -> list[AdapterResponse]:
    ctx = [{"provider_memory_id": "m-lee-seed", "text": "Jordan Lee works on Project Lyra.", "relation": "extends"}]
    return [
        AdapterResponse(query_id="q_lyra_cache", documents=[
            Document(id="m-pg", content="Jordan Lee uses Postgres as the local cache for Project Lyra.",
                     metadata={"relation_context": ctx})], raw={"graph_snapshot": graph}),
        AdapterResponse(query_id="q_rivera_project", documents=[
            Document(id="m-rivera-only", content="Jordan Rivera works only on Project Orion.")],
            raw={"graph_snapshot": graph}),
    ]


def test_same_name_collision_is_winnable_with_correct_attribution():
    unit = RelationGraphBenchmark(slice="same-name-collision").load()[0]
    wrong = {"memories": [_doc("m-pg", "Jordan Rivera uses Postgres as the local cache for Project Lyra.")]}
    wrong_resp = [AdapterResponse(query_id="q_lyra_cache", documents=[], raw={"graph_snapshot": wrong})]
    assert RelationGraphBenchmark().score(unit, wrong_resp)["composite"] < 1.0

    score = RelationGraphBenchmark().score(unit, _collision_responses(_collision_graph()))
    assert score["composite"] == 1.0
    assert score["failures"] == []


def _long_mixed_graph() -> dict[str, object]:
    """Slack live (email superseded in history), two Beacon facts, inferred report."""
    beacon = [{"parent_provider_memory_id": "m-beacon-seed", "relation": "extends"}]
    email_hist = [{"provider_memory_id": "m-email-seed",
                   "text": "Priya prefers email summaries for Project Beacon.", "is_latest": False}]
    return {
        "memories": [
            _doc("m-beacon-seed", "Priya works on Project Beacon."),
            _doc("m-slack", "Priya prefers Slack summaries for Project Beacon.",
                 [{"parent_provider_memory_id": "m-email-seed", "relation": "updates"}], history=email_hist),
            _doc("m-recon", "Project Beacon uses a nightly reconciliation job for stale vector rows.", beacon),
            _doc("m-audit", "Project Beacon stores ingest audit records for each document chunk.", beacon),
            _doc("m-report", "Priya should receive the weekly ingest-audit exception report for Project Beacon.",
                 [{"parent_provider_memory_id": "m-beacon-seed", "relation": "derives"}], is_inference=True),
        ]
    }


def _long_mixed_responses(graph: dict[str, object]) -> list[AdapterResponse]:
    upd = [{"provider_memory_id": "m-email-seed",
            "text": "Priya prefers email summaries for Project Beacon.", "relation": "updates"}]
    der = [{"provider_memory_id": "m-beacon-seed", "text": "Priya works on Project Beacon.", "relation": "derives"}]
    return [
        AdapterResponse(query_id="q_summary_channel", documents=[
            Document(id="m-slack", content="Priya prefers Slack summaries for Project Beacon.",
                     metadata={"relation_context": upd})], raw={"graph_snapshot": graph}),
        AdapterResponse(query_id="q_report_recipient", documents=[
            Document(id="m-report",
                     content="Priya should receive the weekly ingest-audit exception report for Project Beacon.",
                     metadata={"relation_context": der})], raw={"graph_snapshot": graph}),
    ]


def test_long_mixed_is_winnable_with_supersede_and_inference():
    unit = RelationGraphBenchmark(slice="long-mixed").load()[0]
    wrong = {"memories": [_doc("m-email-seed", "Priya prefers email summaries for Project Beacon.")]}
    wrong_resp = [AdapterResponse(query_id="q_summary_channel", documents=[], raw={"graph_snapshot": wrong})]
    assert RelationGraphBenchmark().score(unit, wrong_resp)["composite"] < 1.0

    score = RelationGraphBenchmark().score(unit, _long_mixed_responses(_long_mixed_graph()))
    assert score["composite"] == 1.0
    assert score["failures"] == []
