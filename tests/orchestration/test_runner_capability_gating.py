"""Capability gating: inapplicable (adapter, benchmark) cells skip, not crash.

A graph-requiring benchmark (relation_graph) run against a non-graph-capable
adapter (baseline) must be SKIPPED with a ``not_applicable`` status BEFORE any
prepare/ingest/retrieve -- never crash in the scorer's ``_raw_graph``.
"""

from __future__ import annotations

from memrank.adapters.word_overlap import WordOverlap
from memrank.benchmarks.relation_graph import RelationGraphBenchmark
from memrank.runner import run_cell


def _minimal_kwargs() -> dict[str, object]:
    """The minimum required run_cell kwargs; no live backend, no judge."""
    return {
        "k": 1,
        "repeats": 1,
        "run_id_prefix": "gating-test",
        "model": "gpt-4o-mini",
        "token_budget": 100,
    }


def test_run_cell_skips_non_graph_adapter_for_graph_benchmark():
    result = run_cell(
        WordOverlap(),
        RelationGraphBenchmark(slice="smoke"),
        **_minimal_kwargs(),
    ).to_dict()
    assert result["status"] == "not_applicable"
    assert result["composite"] is None
    assert result["adapter"] == "word-overlap"
    assert result["benchmark"] == "relation_graph"
