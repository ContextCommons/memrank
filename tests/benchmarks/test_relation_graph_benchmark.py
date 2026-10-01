from __future__ import annotations

import pytest

from memrank.benchmarks import REGISTRY
from memrank.benchmarks.relation_graph import (
    RelationGraphBenchmark,
    _relation_parent_matches,
    _relation_type_matches,
)
from memrank.core import AdapterResponse, Document


def _doc(provider_id: str, text: str, relations: list[dict[str, str]] | None = None) -> dict[str, object]:
    return {
        "provider_memory_id": provider_id,
        "text": text,
        "relations": relations or [],
        "history": [],
        "is_inference": False,
    }


def test_relation_graph_is_registered_and_loads_fixture_pack():
    assert REGISTRY["relation_graph"] is RelationGraphBenchmark

    units = RelationGraphBenchmark().load()

    assert [unit.unit_id for unit in units] == [
        "relation/ambiguous-update/001",
        "relation/same-name-collision/001",
        "relation/messy-derives/001",
        "relation/long-mixed/001",
    ]
    assert all(unit.metadata["relation_graph_fixture"] for unit in units)


def _ambiguous_graph() -> dict[str, object]:
    """Graph snapshot for the ambiguous-update case: blue updated to green for planning."""
    return {
        "memories": [
            _doc("m-blue-planning", "Avery prefers blue dashboards for planning reviews."),
            _doc("m-blue-demo", "Avery prefers blue dashboards for sprint demos."),
            _doc(
                "m-green-planning",
                "Avery prefers green dashboards for planning reviews.",
                [{"parent_provider_memory_id": "m-blue-planning", "relation": "updates"}],
            ),
        ]
    }


def test_ambiguous_update_scores_perfect_with_correct_graph_and_context():
    unit = RelationGraphBenchmark(slice="ambiguous-update").load()[0]
    graph = _ambiguous_graph()
    responses = [
        AdapterResponse(
            query_id="q_planning_preference",
            documents=[
                Document(
                    id="m-green-planning",
                    content="Avery prefers green dashboards for planning reviews.",
                    metadata={
                        "relation_context": [
                            {
                                "provider_memory_id": "m-blue-planning",
                                "text": "Avery prefers blue dashboards for planning reviews.",
                                "relation": "updates",
                            }
                        ]
                    },
                )
            ],
            raw={"graph_snapshot": graph},
        ),
        AdapterResponse(
            query_id="q_demo_preference",
            documents=[
                Document(id="m-blue-demo", content="Avery prefers blue dashboards for sprint demos.")
            ],
            raw={"graph_snapshot": graph},
        ),
    ]

    score = RelationGraphBenchmark().score(unit, responses)

    assert score["composite"] == 1.0
    assert score["per_category"]["edge_type_correctness"] == 1.0
    assert score["failures"] == []


def test_messy_derives_penalizes_missing_inference_provenance():
    unit = RelationGraphBenchmark(slice="messy-derives").load()[0]
    graph = {
        "memories": [
            _doc("m-mina", "Mina reports to Omar."),
            _doc("m-omar", "Omar owns the incident-response rotation."),
            _doc("m-nora", "Nora reports to Omar."),
            _doc(
                "m-escalation",
                "Incident-response escalation notes for this week should be sent to Mina.",
            ),
        ]
    }
    responses = [
        AdapterResponse(
            query_id="q_escalation_recipient",
            documents=[
                Document(
                    id="m-escalation",
                    content="Incident-response escalation notes for this week should be sent to Mina.",
                )
            ],
            raw={"graph_snapshot": graph},
        )
    ]

    score = RelationGraphBenchmark().score(unit, responses)
    labels = {failure["label"] for failure in score["failures"]}

    assert score["composite"] < 1.0
    assert score["per_category"]["inference_provenance_completeness"] == 0.0
    assert "expected_inference_flag_missing" in labels
    assert "expected_derives_provenance_missing" in labels


def test_slice_smoke_and_mini_select_subsets():
    assert len(RelationGraphBenchmark(slice="smoke").load()) == 1
    assert len(RelationGraphBenchmark(slice="mini").load()) == 2


def test_unknown_slice_fails_loud_instead_of_empty():
    with pytest.raises(ValueError, match="unknown slice"):
        RelationGraphBenchmark(slice="does-not-exist").load()


def test_score_fails_loud_without_graph_snapshot():
    unit = RelationGraphBenchmark(slice="ambiguous-update").load()[0]
    responses = [AdapterResponse(query_id="q_planning_preference", documents=[])]
    with pytest.raises(RuntimeError, match="graph_snapshot"):
        RelationGraphBenchmark().score(unit, responses)


def test_score_fails_loud_on_errored_graph_snapshot():
    unit = RelationGraphBenchmark(slice="ambiguous-update").load()[0]
    responses = [
        AdapterResponse(
            query_id="q_planning_preference",
            documents=[],
            raw={"graph_snapshot": {"memories": [], "error": "audit endpoint 500"}},
        )
    ]
    with pytest.raises(RuntimeError, match="errored"):
        RelationGraphBenchmark().score(unit, responses)


def test_edge_type_and_parent_set_are_independent_dimensions():
    expected = {"relation": "updates", "parent": "fact-blue"}
    provider_to_fact = {"m-blue": "fact-blue"}
    fact_to_providers = {"fact-blue": {"m-blue"}}
    right_parent_wrong_type = {"relations": [{"relation": "mentions", "parent_provider_memory_id": "m-blue"}]}
    right_type_wrong_parent = {"relations": [{"relation": "updates", "parent_provider_memory_id": "m-other"}]}
    assert _relation_parent_matches(right_parent_wrong_type, expected, provider_to_fact, fact_to_providers) is True
    assert _relation_type_matches(right_parent_wrong_type, expected) is False
    assert _relation_type_matches(right_type_wrong_parent, expected) is True
    assert _relation_parent_matches(right_type_wrong_parent, expected, provider_to_fact, fact_to_providers) is False


def test_untested_dimensions_are_excluded_not_scored_perfect():
    from memrank.benchmarks.relation_graph import _score_fixture
    fixture = {
        "id": "relation/x/1",
        "seed_memories": [],
        "expected_memories": [{"id": "f1", "canonical_fact": "Cara likes tea.", "required": True}],
        "expected_absences": [],
        "queries": [],
    }
    result = _score_fixture(fixture, memories=[], search_results=[])
    assert result["scores"]["entity_disambiguation"] == 0.0
    assert "retrieval_exposure" not in result["scores"]
    assert result["weighted_graph_score"] == 0.0


def test_superseded_fact_in_history_is_not_counted_present():
    from memrank.benchmarks.relation_graph import _score_fixture
    fixture = {
        "id": "relation/y/1", "seed_memories": [{"id": "blue", "text": "Avery prefers blue."}],
        "expected_memories": [], "queries": [],
        "expected_absences": [{"id": "blue", "fact": "Avery prefers blue."}],
    }
    green = {
        "provider_memory_id": "m-green", "text": "Avery prefers green.", "is_latest": True,
        "is_forgotten": False, "relations": [{"parent_provider_memory_id": "m-blue", "relation": "updates"}],
        "history": [{"provider_memory_id": "m-blue", "text": "Avery prefers blue.", "is_latest": False}],
    }
    result = _score_fixture(fixture, memories=[green], search_results=[])
    assert result["scores"]["entity_disambiguation"] == 1.0
    assert result["failures"] == []


def test_match_text_rejects_spurious_short_substring():
    from memrank.benchmarks.relation_graph import _match_text
    facts = {
        "lee": {"canonical_fact": "Jordan Lee uses Postgres for Project Lyra.", "aliases": []},
        "rivera": {"canonical_fact": "Jordan Rivera works only on Project Orion.", "aliases": []},
    }
    assert _match_text("Jordan", facts) is None
    # Single unambiguous fact: short substring must still be rejected (the real bug;
    # with two facts above, "Jordan" is dropped only by the ambiguity tie-break).
    assert _match_text("Jordan", {"lee": facts["lee"]}) is None
    assert _match_text("Jordan Lee uses Postgres for Project Lyra.", facts) == "lee"
    facts["lee"]["aliases"] = ["Jordan Lee uses Postgres as the Lyra cache."]
    assert _match_text("Jordan Lee uses Postgres as the Lyra cache.", facts) == "lee"


def test_forbidden_fact_at_rank_two_is_penalized():
    from memrank.benchmarks.relation_graph import RelationGraphBenchmark
    from memrank.core import AdapterResponse, Document
    fixture = {
        "id": "relation/z/1", "seed_memories": [{"id": "old", "text": "Avery prefers blue."}],
        "expected_memories": [], "expected_absences": [],
        "queries": [{"id": "q", "text": "?", "expected_retrieval": {"forbidden_facts": ["old"]}}],
    }
    green = {"provider_memory_id": "m-green", "text": "Avery prefers green.", "is_latest": True,
             "is_forgotten": False, "relations": [], "history": []}
    unit = type("U", (), {"unit_id": "u", "queries": fixture["queries"],
                          "metadata": {"relation_graph_fixture": fixture}})()
    responses = [AdapterResponse(query_id="q", documents=[
        Document(id="m-green", content="Avery prefers green."),
        Document(id="m-old", content="Avery prefers blue."),
    ], raw={"graph_snapshot": {"memories": [green]}})]
    score = RelationGraphBenchmark().score(unit, responses)
    assert score["per_category"]["retrieval_exposure"] == 0.0
    assert any(f["label"] == "forbidden_fact_retrieved" for f in score["failures"])
