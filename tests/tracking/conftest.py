"""Shared fixtures for the MLflow-export tests.

Provides ``write_cell``: writes a synthetic, deterministic per-cell result JSON into a run folder
in the shape :mod:`memrank.tracking.export` reads (composite/latency/token stats + a nested
``receipt.config``). Kept here so both the importer tests and the run-hook tests reuse one payload
definition. No network, no live backend, no timing.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest


@pytest.fixture
def write_cell() -> Callable[..., Path]:
    def _write(run_dir: Path, adapter: str = "word-overlap", benchmark: str = "demo") -> Path:
        payload = {
            "adapter": adapter, "benchmark": benchmark, "composite": 0.42,
            "context_tokens_mean": 123.0, "est_dollars_per_query": 0.0007,
            "latency_metrics": {"ingest_p50_ms": 5.0, "retrieve_p50_ms": 9.0},
            "token_metrics": {"tokens_per_query_mean": 200.0},
            "receipt": {"adapter_name": adapter, "benchmark_name": benchmark,
                        "config_hash": "abc123", "started_at": "2026-01-01T00:00:00+00:00",
                        "dataset_version": "ds-1",
                        "config": {"k": 10, "task_version": 3,
                                   "components": {"llm": {"model": "gpt-4o-mini"}}}},
        }
        cell = run_dir / f"{adapter}__{benchmark}.json"
        cell.write_text(json.dumps(payload))
        return cell
    return _write
