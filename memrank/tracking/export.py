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
"""Mirror memrank's local run registry into a local MLflow store.

memrank accumulates every ``memrank submit`` into ``runs/<run-id>/<adapter>__<benchmark>.json``
(see :mod:`memrank.runs.registry`). This module logs one MLflow run per result cell -- config as
params, composite/latency/token stats as metrics, and the raw result JSON as an artifact -- so
``mlflow ui`` gives a browsable, sortable, compare-able view of run history. It is additive and
decoupled: it reads the registry but never mutates memrank state.

Two entry points share the same idempotent core (:func:`import_runs`):

- :func:`mirror_run` -- import just one run folder's cells; called automatically at the end of
  ``memrank submit`` when the integration is enabled (see :func:`memrank.config.mlflow_enabled`).
- the ``scripts/internal/mlflow_import/mlflow_import.py`` CLI -- bulk-import the whole registry on demand.

``mlflow`` is an optional dependency (``uv sync --extra mlflow``) and is lazy-imported via
:func:`_require_mlflow`, which raises loudly (never degrades) if the extra is absent -- so
``--help`` and callers that never mirror work without it installed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from memrank.provenance.engine import derive_origin
from memrank.runs.registry import RunInfo, list_runs

DEFAULT_EXPERIMENT = "memrank"
# MLflow 3.x's recommended local backend; the legacy file:// store is deprecated. Relative to CWD.
DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
_CELL_TAG = "memrank_cell"


def _require_mlflow() -> Any:
    """Return the ``mlflow`` module, or raise loudly if the optional extra is not installed.

    Single chokepoint for the lazy import so every mirroring path fails with the same actionable
    message instead of a bare ``ImportError`` -- mirrors ``judge_client.anthropic_completer``.
    """
    from memrank.errors import optional_import

    return optional_import("mlflow", "mlflow")


def _cell_key(info: RunInfo) -> str:
    """The idempotency key for one registry cell: ``<run-id>/<adapter>__<benchmark>``."""
    return f"{info.run_id}/{info.adapter}__{info.benchmark}"


def flatten(data: dict[str, Any], prefix: str = "") -> dict[str, str]:
    """Flatten a nested config dict into dotted keys for ``log_params``.

    Args:
        data: The (possibly nested) config mapping, e.g. ``receipt["config"]``.
        prefix: Dotted key prefix accumulated during recursion.

    Returns:
        Mapping of dotted key (``components.llm.model``) to stringified leaf value.
    """
    params: dict[str, str] = {}
    for key, value in data.items():
        dotted = f"{prefix}{key}"
        if isinstance(value, dict):
            params.update(flatten(value, f"{dotted}."))
        else:
            params[dotted] = str(value)
    return params


def numeric_metrics(result: dict[str, Any]) -> dict[str, float]:
    """Collect the numeric result fields MLflow can log as metrics.

    Pulls the scalar ``composite``/``context_tokens_mean``/``est_dollars_per_query`` plus every
    numeric value inside ``latency_metrics`` and ``token_metrics``. Non-numeric and ``None``
    values are skipped; booleans are excluded since MLflow metrics must be real numbers.

    Args:
        result: A parsed per-cell result JSON.

    Returns:
        Mapping of metric name to float value.
    """
    candidates: dict[str, Any] = {}
    for scalar in ("composite", "context_tokens_mean", "est_dollars_per_query"):
        candidates[scalar] = result.get(scalar)
    for group in ("latency_metrics", "token_metrics"):
        candidates.update(result.get(group) or {})
    return {
        name: float(value)
        for name, value in candidates.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }


def imported_cell_keys(experiment_name: str, *, tracking_uri: str) -> set[str]:
    """Return the ``memrank_cell`` tags already logged under the experiment.

    Used to make the import idempotent. Returns an empty set when the experiment does not yet
    exist so the first import proceeds.

    ``tracking_uri`` is explicit rather than inherited from MLflow's global state: this reads a
    store to decide what to skip, and answering from the wrong one silently reports work as already
    done. It used to rely on :func:`import_runs` having set the URI first, which held only because
    that was the sole caller.

    Args:
        experiment_name: The MLflow experiment to inspect.
        tracking_uri: The MLflow store to inspect.

    Returns:
        Set of ``"<run-id>/<adapter>__<benchmark>"`` keys already present.
    """
    mlflow = _require_mlflow()
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        return set()
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], output_format="list")
    return {run.data.tags[_CELL_TAG] for run in runs if _CELL_TAG in run.data.tags}


def import_cell(info: RunInfo) -> None:
    """Log one registry cell as a single MLflow run.

    Args:
        info: The registry entry pointing at a per-cell result JSON.
    """
    mlflow = _require_mlflow()
    result = json.loads(info.path.read_text(encoding="utf-8"))
    receipt = result.get("receipt", {})
    config = receipt.get("config", {})
    with mlflow.start_run(run_name=info.run_id):
        mlflow.log_params(flatten(config) | {"adapter": info.adapter, "benchmark": info.benchmark})
        mlflow.set_tags({
            _CELL_TAG: _cell_key(info),
            "config_hash": receipt.get("config_hash"),
            "run_id": info.run_id,
            "dataset_version": receipt.get("dataset_version"),
            "task_version": config.get("task_version"),
            **_provenance_tags(receipt.get("engine_provenance", {})),
        })
        mlflow.log_metrics(numeric_metrics(result))
        mlflow.log_artifact(str(info.path))


def _provenance_tags(prov: dict[str, Any]) -> dict[str, str]:
    """Filterable engine-provenance tags for one cell (empty when no facet present).

    Provenance is NOT in ``config``, so it is surfaced as tags rather than params:
    the purl (pins the exact engine build), the derived origin (fork/mirror/...),
    the manufacturer, the upstream base, deployment model, and the dirty flag.
    """
    if not prov:
        return {}
    props = prov.get("properties") or {}
    ancestors = (prov.get("pedigree") or {}).get("ancestors") or []
    tags = {
        "engine_purl": prov.get("purl"),
        "engine_origin": derive_origin(prov),
        "engine_manufacturer": prov.get("manufacturer"),
        "engine_upstream": ancestors[0] if ancestors else None,
        "engine_distribution": props.get("memrank:distribution"),
        "engine_dirty": props.get("memrank:source_dirty"),
    }
    return {key: str(value) for key, value in tags.items() if value is not None}


def import_runs(infos: list[RunInfo], *, tracking_uri: str, experiment: str) -> int:
    """Import every not-yet-imported cell in ``infos`` into MLflow; return the count imported.

    Idempotent: cells whose ``memrank_cell`` tag already exists under ``experiment`` are skipped.

    Args:
        infos: Registry cells to import (e.g. from :func:`memrank.runs.registry.list_runs`).
        tracking_uri: MLflow tracking URI to log into.
        experiment: MLflow experiment name to log under.

    Returns:
        The number of cells newly imported this call.
    """
    mlflow = _require_mlflow()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    already = imported_cell_keys(experiment, tracking_uri=tracking_uri)
    imported = 0
    for info in infos:
        if _cell_key(info) in already:
            continue
        import_cell(info)
        imported += 1
    return imported


def mirror_run(run_dir: Path, *, tracking_uri: str, experiment: str) -> int:
    """Mirror just one run folder's cells into MLflow; return the count imported.

    Scoped to ``run_dir`` (matched by ``run_dir.name`` against the registry's run-id) so the
    per-run hook does not re-scan the whole registry's contents each invocation.

    Args:
        run_dir: The run folder just written by ``memrank submit``.
        tracking_uri: MLflow tracking URI to log into.
        experiment: MLflow experiment name to log under.

    Returns:
        The number of cells newly imported for this run.
    """
    infos = [info for info in list_runs() if info.run_id == run_dir.name]
    return import_runs(infos, tracking_uri=tracking_uri, experiment=experiment)
