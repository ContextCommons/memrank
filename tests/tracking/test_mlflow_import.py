"""Deterministic tests for memrank.tracking.export (run registry -> MLflow importer).

Writes a synthetic registry cell into a tmp ``MEMRANK_RUNS_DIR``, imports it into a tmp
``sqlite://`` MLflow store, and asserts the params/metrics/tags via ``mlflow.search_runs``. The
working directory is redirected to a tmp path so the sqlite DB and artifact dir never touch the
repo. No network, no live backend, no timing.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from memrank.runs import registry

mlflow = pytest.importorskip("mlflow")  # requires: uv sync --extra mlflow

from memrank.tracking import export  # noqa: E402  (imported after the extra gate)

_EXPERIMENT = "memrank-test"


def _import(monkeypatch, tmp_path: Path, write_cell) -> str:
    monkeypatch.setenv("MEMRANK_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.chdir(tmp_path)  # keep the sqlite DB + artifact dir out of the repo
    write_cell(registry.new_run_dir("demo"))
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    export.import_runs(registry.list_runs(), tracking_uri=uri, experiment=_EXPERIMENT)
    return uri


def test_import_logs_params_metrics_and_tags(monkeypatch, tmp_path, write_cell):
    uri = _import(monkeypatch, tmp_path, write_cell)
    mlflow.set_tracking_uri(uri)
    runs = mlflow.search_runs(experiment_names=[_EXPERIMENT])
    assert len(runs) == 1
    row = runs.iloc[0]
    assert row["params.k"] == "10"
    assert row["params.components.llm.model"] == "gpt-4o-mini"
    assert row["params.adapter"] == "word-overlap"
    assert row["metrics.composite"] == pytest.approx(0.42)
    assert row["metrics.retrieve_p50_ms"] == pytest.approx(9.0)
    assert row["tags.memrank_cell"].endswith("baseline__demo")


def test_import_is_idempotent(monkeypatch, tmp_path, write_cell):
    uri = _import(monkeypatch, tmp_path, write_cell)
    second = export.import_runs(
        registry.list_runs(), tracking_uri=uri, experiment=_EXPERIMENT)
    assert second == 0  # nothing new to import on the second pass
    mlflow.set_tracking_uri(uri)
    assert len(mlflow.search_runs(experiment_names=[_EXPERIMENT])) == 1
