"""Concurrent cross-unit ingest (--workers): same recall as sequential, isolated, flagged."""

from __future__ import annotations

from memrank.adapters.word_overlap import WordOverlap
from memrank.runner import run_cell
from tests.fakes import MultiUnitFakeBenchmark


def _run(workers: int) -> dict:
    return run_cell(
        WordOverlap(), MultiUnitFakeBenchmark(n_units=3),
        k=10, repeats=1, run_id_prefix="run-c", model="gpt-4o-mini",
        token_budget=5000, workers=workers, make_adapter=WordOverlap,
    ).to_dict()


def _query_outcomes(result: dict):
    return sorted((r["query_id"], r["hit"]) for r in result["per_query"])


def test_concurrent_matches_sequential_recall():
    seq, par = _run(1), _run(3)
    assert seq["composite"] == par["composite"] == 1.0
    # completion order must not change the scored outcome
    assert _query_outcomes(seq) == _query_outcomes(par)
    assert seq["n_units"] == par["n_units"] == 3


def test_concurrent_flags_contended_latency_and_records_workers():
    seq, par = _run(1), _run(3)
    assert seq["workers"] == 1 and par["workers"] == 3
    assert seq["latency_contended"] is False
    assert par["latency_contended"] is True


def test_concurrent_units_stay_isolated():
    # each unit's query must retrieve only its own doc -- no cross-unit bleed
    par = _run(3)
    for row in par["per_query"]:
        idx = row["query_id"][1:]  # "q2" -> "2"
        retrieved_ids = {d["id"] for d in row["retrieved"]}
        assert retrieved_ids <= {f"d{idx}"}, f"{row['query_id']} saw {retrieved_ids}"


def test_workers_requires_factory():
    import pytest
    with pytest.raises(ValueError, match="make_adapter"):
        run_cell(WordOverlap(), MultiUnitFakeBenchmark(2), k=10, repeats=1,
                 run_id_prefix="r", model="gpt-4o-mini", token_budget=5000, workers=2).to_dict()
