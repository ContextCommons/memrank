# Used by the current path (cli/evals, through get_eval): it sits in a deprecated package but is not
# deprecated itself, and it must not warn.
"""Structured, presentation-neutral target and eval discovery.

Presentation-neutral in both directions: this describes what a deployment can run, and every
consumer decides how to say it. ``memrank evals show`` renders these dicts for a terminal, the
API serves them as JSON, and the browser fills its pickers from them -- one description, three
renderings, so a picker cannot offer an eval the harness does not have.

That is also why :func:`describe_eval` lives here rather than in ``evals_cli``, where it started:
a Typer module is a rendering, and the API cannot import one to answer a question about the
catalog.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from memrank.benchmarks import get_benchmark, judge_required
from memrank.benchmarks.refs import list_eval_refs, parse_eval_ref
from memrank.core import QUALITY_METRIC_DESCRIPTIONS, QUALITY_METRIC_LABELS
from memrank.targets.catalog import list_targets as target_names
from memrank.targets.catalog import resolve_target
from memrank.targets.resolve import RefError


def list_targets() -> list[str]:
    """Return stable target references in catalog order."""
    return target_names()


def get_target(ref: str, overrides: list[str] | None = None) -> dict[str, Any]:
    """Return one fully resolved target without presentation formatting."""
    return asdict(resolve_target(ref, overrides or []))


def list_evals() -> list[str]:
    """Every runnable evaluation, canonically spelled -- `beam:100k-smoke`, not `beam`.

    A tier and a slice select WHICH evaluation runs, not how one runs, so the catalog lists them
    as separate evaluations. Their numbers were never comparable; now their names say so.
    """
    return list_eval_refs()


def get_eval(ref: str) -> dict[str, Any]:
    """Return declared eval metadata without loading its dataset."""
    try:
        name, _kwargs, canonical = parse_eval_ref(ref)
    except RefError as exc:
        raise ValueError(str(exc)) from exc
    return {**describe_eval(name), "ref": canonical}


def describe_eval(name: str) -> dict[str, Any]:
    """One eval as a plain dict -- the single source for every rendering.

    Instantiated (not read off the class) because ``is_synthetic`` and
    ``question_text_public`` are decided per environment: a ``*_DATA_PATH`` override may
    point at private data, and the catalog must describe what a run here would actually do.

    Constructing a benchmark is I/O-free by contract -- only ``load()`` may ever download -- so
    this stays safe to call on a request path.
    """
    bench = get_benchmark(name)
    metric = bench.quality_metric
    # Delegated rather than recomputed: the CLI derives its judge default from the same function,
    # and a catalog that said "optional" while `submit` judged anyway would be two answers.
    needs_judge = judge_required(name)
    return {
        "name": bench.name,
        "dataset_version": bench.dataset_version,
        "task_version": bench.VERSION,
        "unit": bench.info.unit,
        "units_declared": bench.info.units_declared,
        "slices": list(bench.info.slices),
        "tiers": list(bench.info.tiers),
        "quality_metric": metric,
        "quality_label": QUALITY_METRIC_LABELS.get(metric, metric),
        "quality_description": QUALITY_METRIC_DESCRIPTIONS.get(metric, ""),
        "composite_rankable": bench.composite_rankable,
        "substring_recall_supported": bench.substring_recall_supported,
        "judge": "required" if needs_judge else "optional",
        "is_synthetic": bench.is_synthetic,
        "question_text_public": bench.question_text_public,
        "requires_graph": bench.requires_graph,
    }
