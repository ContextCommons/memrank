"""Pure deterministic expansion and validation of Memrank sweep requests."""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import asdict
from typing import Any

from memrank import config
from memrank.application.catalogs import get_eval
from memrank.application.types import (
    Experiment,
    ExperimentSettings,
    ModelRef,
    Notice,
    Requirement,
    SweepPlan,
    SweepRequest,
)
from memrank.metrics import cost
from memrank.secrets.requirements import PROVIDER_KEY_ENV
from memrank.targets.catalog import list_targets as target_names
from memrank.targets.catalog import required_secrets_for, resolve_target
from memrank.targets.manifest import Manifest, ManifestError

MAX_SWEEP_EXPERIMENTS = 100
_AXES = (
    ("engine_llm", "engine_llms"), ("embedder", "embedders"),
    ("reader", "readers"), ("pricing_model", "pricing_models"),
    ("slice", "slices"), ("tier", "tiers"), ("k", "k_values"),
    ("repeats", "repeat_counts"), ("token_budget", "token_budgets"),
    ("seed", "seeds"), ("workers", "worker_counts"),
    ("units", "unit_selections"), ("judge", "judges"),
)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _alternatives(request: SweepRequest) -> list[tuple[str, list[Any]]]:
    out: list[tuple[str, list[Any]]] = []
    for setting_name, axis_name in _AXES:
        alternatives = getattr(request.axes, axis_name)
        out.append((setting_name, alternatives or [getattr(request.settings, setting_name)]))
    return out


def _settings(base: ExperimentSettings, names: list[str], values: tuple[Any, ...]) -> ExperimentSettings:
    updates = dict(zip(names, values, strict=True))
    return base.model_copy(update=updates)


def _component_override(role: str, model: ModelRef | None) -> list[str]:
    if model is None:
        return []
    values = [f"{role}={model.provider}/{model.model}"]
    if role == "embedder" and model.dimensions is not None:
        values.append(f"embedder.dims={model.dimensions}")
    return values


def _overrides(settings: ExperimentSettings) -> list[str]:
    """The manifest override tokens this cell resolves against, free-form ones first.

    Order is what decides a collision: ``parse_overrides`` keeps the last token for a key, so a
    role spelled both as ``settings.overrides`` and as a typed component field resolves to the
    typed one.
    """
    values = list(settings.overrides)
    values.extend(_component_override("llm", settings.engine_llm))
    values.extend(_component_override("embedder", settings.embedder))
    return values


def _manifest_dict(manifest: Manifest) -> dict[str, Any]:
    return asdict(manifest)


def _resolved_settings(settings: ExperimentSettings, manifest: Manifest) -> ExperimentSettings:
    components = manifest.components
    llm = components.get("llm")
    embedder = components.get("embedder")
    return settings.model_copy(update={
        "engine_llm": ModelRef(provider=llm.provider, model=llm.model)
        if llm and llm.provider and llm.model else None,
        "embedder": ModelRef(provider=embedder.provider, model=embedder.model,
                             dimensions=embedder.dims)
        if embedder and embedder.provider and embedder.model else None,
    })


def resolve_experiment(target_ref: str, eval_ref: str, settings: ExperimentSettings,
                       overrides: list[str] | None = None) -> Experiment:
    """Resolve one cell so every presentation computes the same identity."""
    overrides = _overrides(settings) if overrides is None else overrides
    manifest = resolve_target(target_ref, overrides)
    settings = _resolved_settings(settings, manifest)
    material = {
        "target": _manifest_dict(manifest), "eval": get_eval(eval_ref),
        "settings": settings.model_dump(mode="json"),
    }
    experiment_id = _hash(material)
    label = f"{manifest.name}-{experiment_id[:8]}"
    return Experiment(experiment_id=experiment_id, label=label, target_ref=manifest.name,
                      eval_ref=eval_ref, target=material["target"], settings=settings,
                      overrides=overrides)


def _provider_refusals(experiment: Experiment) -> list[Notice]:
    refused: list[Notice] = []
    components = experiment.target.get("components", {})
    for role, component in components.items():
        provider = component.get("provider")
        if provider and provider not in PROVIDER_KEY_ENV:
            refused.append(Notice(code="unsupported_provider",
                                  message=f"{role} provider {provider!r} is unsupported",
                                  experiment_ids=[experiment.experiment_id]))
    return refused


def _requirements(experiment: Experiment, request: SweepRequest) -> list[Requirement]:
    manifest = resolve_target(experiment.target_ref, experiment.overrides)
    names = required_secrets_for(manifest)
    if experiment.settings.judge.enabled:
        names.append("ANTHROPIC_API_KEY")
    cloud = request.routing.placement == "cloud"
    return [Requirement(code="credential", message=f"credential {name} is required",
                        remedy=f"configure {name} in the selected scope",
                        experiment_ids=[experiment.experiment_id],
                        scope="org" if cloud else "local",
                        satisfied=None if cloud else config.secret(name) is not None)
            for name in sorted(set(names))]


def _abstract_refusal(experiment: Experiment) -> list[Notice]:
    """Refuse a base other targets inherit from, naming the variants that are runnable.

    A base leaves a component blank so each variant can fill it, and a blank knob is not neutral:
    the engine chooses. ``mem0`` used to be such a base, and defaulted its unset embedder to
    text-embedding-3-small -- so the run either died on a missing key or, worse, SUCCEEDED, measuring
    an embedder no manifest declared. It is concrete now (it states that embedder), but the shape
    reaches this door through any operator manifest.

    Checked here rather than only in the CLI because every surface that plans is a surface that
    can submit: the browser and the MCP tools reach a task through this function and would
    otherwise pay for one to find out.
    """
    if not experiment.target.get("abstract"):
        return []
    variants = sorted(name for name in target_names()
                      if name != experiment.target_ref
                      and name.startswith(f"{experiment.target_ref}:"))
    return [Notice(code="abstract_target",
                   message=f"{experiment.target_ref!r} is a base other targets inherit from, "
                           f"not something to run: it leaves a component unset",
                   remedy=f"run one of: {', '.join(variants)}" if variants else None,
                   experiment_ids=[experiment.experiment_id])]


def _notices(experiment: Experiment) -> tuple[list[Notice], list[Notice]]:
    warnings: list[Notice] = []
    refusals = _provider_refusals(experiment) + _abstract_refusal(experiment)
    eval_info = get_eval(experiment.eval_ref)
    if experiment.settings.slice and experiment.settings.slice not in eval_info["slices"]:
        refusals.append(Notice(code="unknown_slice",
                               message=f"eval has no slice {experiment.settings.slice!r}",
                               experiment_ids=[experiment.experiment_id]))
    if experiment.settings.tier and experiment.settings.tier not in eval_info["tiers"]:
        refusals.append(Notice(code="unknown_tier",
                               message=f"eval has no tier {experiment.settings.tier!r}",
                               experiment_ids=[experiment.experiment_id]))
    if experiment.settings.reader and not experiment.settings.judge.enabled:
        refusals.append(Notice(code="reader_without_judge",
                               message="reader has no effect unless judging is enabled",
                               experiment_ids=[experiment.experiment_id]))
    reader = experiment.settings.reader
    if reader and (reader.provider != "anthropic" or not reader.model.startswith("claude-")):
        refusals.append(Notice(code="unsupported_reader",
                               message="the current judge runtime requires an Anthropic claude reader",
                               experiment_ids=[experiment.experiment_id]))
    try:
        cost.input_price_per_mtok(experiment.settings.pricing_model)
    except ValueError as exc:
        refusals.append(Notice(code="unknown_pricing_model", message=str(exc),
                               experiment_ids=[experiment.experiment_id]))
    return warnings, refusals


def _plan_hash(request: SweepRequest, experiments: list[Experiment]) -> str:
    material = {
        "contract_version": 1, "request": request.model_dump(mode="json"),
        "experiments": [item.experiment_id for item in experiments],
        "catalog": [{"target": item.target, "eval": get_eval(item.eval_ref)}
                    for item in experiments],
    }
    return _hash(material)


def _expand(request: SweepRequest, alternatives: list[tuple[str, list[Any]]]
            ) -> tuple[list[Experiment], list[Notice]]:
    names = [name for name, _ in alternatives]
    unique: dict[str, Experiment] = {}
    refusals: list[Notice] = []
    values = [items for _, items in alternatives]
    for target in request.targets:
        for eval_ref in request.evals:
            for combination in itertools.product(*values):
                try:
                    settings = _settings(request.settings, names, combination)
                    experiment = resolve_experiment(target, eval_ref, settings)
                except (ManifestError, ValueError) as exc:
                    refusals.append(Notice(code="invalid_experiment", message=str(exc)))
                    continue
                unique.setdefault(experiment.experiment_id, experiment)
    return list(unique.values()), refusals


def plan_sweep(request: SweepRequest) -> SweepPlan:
    """Expand, resolve, validate, and hash a sweep without creating a run."""
    alternatives = _alternatives(request)
    size = len(request.targets) * len(request.evals)
    for _, values in alternatives:
        size *= len(values)
    if size > MAX_SWEEP_EXPERIMENTS:
        refusal = Notice(code="sweep_too_large",
                         message=f"matrix has {size} cells; maximum is {MAX_SWEEP_EXPERIMENTS}")
        return SweepPlan(plan_hash="", request=request, experiments=[], refusals=[refusal])
    experiments, refusals = _expand(request, alternatives)
    requirements = [req for item in experiments for req in _requirements(item, request)]
    notice_pairs = [_notices(item) for item in experiments]
    warnings = [notice for pair in notice_pairs for notice in pair[0]]
    refusals.extend(notice for pair in notice_pairs for notice in pair[1])
    return SweepPlan(plan_hash=_plan_hash(request, experiments), request=request,
                     experiments=experiments, requirements=requirements, warnings=warnings,
                     refusals=refusals, deduplicated=size - len(experiments))
