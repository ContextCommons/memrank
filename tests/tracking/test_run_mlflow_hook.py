"""Deterministic tests for the ``memrank run`` -> MLflow auto-mirror hook.

Exercises ``runner._mirror_to_mlflow`` directly: it is a no-op unless MEMRANK_MLFLOW_ENABLED=1,
and when enabled it mirrors the just-recorded run's cells into the configured MLflow store. Uses a
tmp registry + tmp sqlite store (chdir'd out of the repo). No network, no live backend, no timing.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from memrank import runner
from memrank.runs import registry

mlflow = pytest.importorskip("mlflow")  # requires: uv sync --extra mlflow

_EXPERIMENT = "memrank-hook-test"


def _prepare(monkeypatch, tmp_path: Path, write_cell) -> Path:
    monkeypatch.setenv("MEMRANK_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("MEMRANK_MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.setenv("MEMRANK_MLFLOW_EXPERIMENT", _EXPERIMENT)
    monkeypatch.chdir(tmp_path)
    run_dir = registry.new_run_dir("demo")
    write_cell(run_dir)
    return run_dir


def test_hook_mirrors_when_enabled(monkeypatch, tmp_path, write_cell):
    monkeypatch.setenv("MEMRANK_MLFLOW_ENABLED", "1")
    run_dir = _prepare(monkeypatch, tmp_path, write_cell)
    runner._mirror_to_mlflow(run_dir)
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    assert len(mlflow.search_runs(experiment_names=[_EXPERIMENT])) == 1


def test_hook_noop_when_disabled(monkeypatch, tmp_path, write_cell):
    monkeypatch.setenv("MEMRANK_MLFLOW_ENABLED", "0")
    run_dir = _prepare(monkeypatch, tmp_path, write_cell)
    runner._mirror_to_mlflow(run_dir)
    assert not (tmp_path / "mlflow.db").exists()  # store never touched when disabled
