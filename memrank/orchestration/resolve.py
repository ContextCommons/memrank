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
"""What a submission names, resolved and gated: targets, overrides, credentials, judge.

Typer exceptions still surface from a few gates here -- a scheduled follow-up converts
them to `MemrankError` subclasses translated at the CLI boundary (tech-debt.md).
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any

import typer

from memrank.core import MemoryAdapter
from memrank.evaluation.cell import AdapterFactory, _close_adapter
from memrank.judging.client import JUDGE_MODEL_PREFIX
from memrank.judging.judge import JudgeConfig
from memrank.term import style


def _manifest_factory(ref: str, overrides: list[str]) -> AdapterFactory:
    """A zero-arg constructor for one manifest-configured target."""
    from memrank.targets import resolve_target
    from memrank.targets.factory import build_adapter

    def make() -> MemoryAdapter:
        return build_adapter(resolve_target(ref, overrides))
    return make


def _refuse_abstract(ref: str, overrides: list[str]) -> None:
    """Refuse a base that other targets inherit from, naming what to run instead.

    A base leaves a component blank so each variant can fill it, and a blank knob is not neutral:
    the engine chooses. `memrank submit mem0` once started a task that died on a missing
    OPENAI_API_KEY, because mem0 defaulted its unset embedder to text-embedding-3-small -- and had a
    key been present it would have SUCCEEDED, measuring an embedder no manifest declared. `mem0`
    now declares that embedder and runs; the refusal serves operator manifests on `targets.path`.

    Here rather than deeper, so no image is pulled and no task is paid for. Resolution itself still
    works: the variants inherit from this manifest and would break if it did not resolve.

    Raises:
        typer.Exit: Always, when ``ref`` names an abstract manifest.
    """
    from memrank.targets import ManifestError, TargetNotFound, list_targets, resolve_target

    try:
        target = resolve_target(ref, overrides)
    except (TargetNotFound, ManifestError):
        return      # an unknown or malformed ref is the caller's error to report, with its message
    if not target.abstract:
        return
    variants = sorted(name for name in list_targets()
                      if name != target.name and name.startswith(f"{target.name}:"))
    style.error(
        f"{ref!r} is a base other targets inherit from, not something to run: it leaves a "
        f"component blank for each variant to fill, so the engine would choose one and no receipt "
        f"would say which. Run {' or '.join(variants) if variants else 'one of its variants'}.")
    raise typer.Exit(1)


def _manifest_factories(refs: str, overrides: list[str], *,
                        verify_engine: bool = True) -> list[tuple[str, AdapterFactory]]:
    """Adapter factories for the ``run <target> <benchmark>`` form; a comma sweeps targets.

    Every ref is resolved AND cross-checked against the environment up front, so a bad ref or a
    mislabelled engine fails before any unit is ingested rather than partway through a sweep.

    Args:
        refs: One ref, or several comma-separated.
        overrides: ``key=value`` component overrides.
        verify_engine: Whether to construct each adapter, which cross-checks the manifest against a
            *live* engine. Must be off when SUBMITTING a run elsewhere: the machine doing the
            submitting has no engine to check against, and the cross-check would fail for the
            absence of something it was never going to use. The check still runs -- inside the task,
            where the engine actually is.

    Raises:
        typer.Exit: On a malformed ref or a manifest/engine disagreement.
    """
    from memrank.targets import ManifestError, resolve_target

    out: list[tuple[str, AdapterFactory]] = []
    for ref in [r.strip() for r in refs.split(",") if r.strip()]:
        _refuse_abstract(ref, overrides)
        factory = _manifest_factory(ref, overrides)
        try:
            resolved = resolve_target(ref, overrides)
            if verify_engine and resolved.binding is None:
                # Building once validates the ref, the environment cross-check, AND the engine's own
                # report -- for EVERY target before the first is run, so a sweep cannot spend money on
                # target 1 and then abort on target 2's mislabelled engine.
                _close_adapter(factory())
        except ManifestError as exc:
            style.error(str(exc))
            raise typer.Exit(1) from exc
        out.append((ref, factory))
    return out


# Override keys consumed by the RUN rather than by the target manifest. A target's identity is what
# is under test; the reader is who answers using it -- a different axis, so it is not a manifest field.
_RUN_LEVEL_OVERRIDES: tuple[str, ...] = ("reader",)


def _split_overrides(tokens: list[str]) -> tuple[list[str], dict[str, str]]:
    """Separate component overrides (for the manifest) from run-level ones (e.g. ``reader=``)."""
    from memrank.targets.resolve import parse_overrides

    parsed = parse_overrides(tokens)
    run_level = {key: parsed.pop(key) for key in _RUN_LEVEL_OVERRIDES if key in parsed}
    return [f"{key}={value}" for key, value in parsed.items()], run_level


def _validate_reader(reader: str | None, *, judged: bool) -> None:
    """Reject a reader override that cannot take effect, at parse time rather than mid-run.

    Both failures were previously silent or late: without ``--judge`` no reader ever runs, so the
    override did nothing at all; and the judge client hands the model name straight to the Anthropic
    SDK, so a non-Anthropic name died at the first API call -- after egress had been acknowledged and
    the run had started.
    """

    if reader is None:
        return
    if not judged:
        raise typer.BadParameter(
            f"reader={reader!r} has no effect without --judge: the reader is the model that answers "
            "from retrieved memories, and nothing answers on an unjudged run.")
    if not reader.startswith(JUDGE_MODEL_PREFIX):
        raise typer.BadParameter(
            f"reader={reader!r} is not an Anthropic model. The judge runtime is Anthropic-backed, "
            f"so the reader must start with {JUDGE_MODEL_PREFIX!r} "
            "(e.g. claude-haiku-4-5-20251001).")


def _cli_experiments(factories, eval_name: str, overrides: list[str], params: dict[str, Any]):
    """Resolve CLI-created cells through the same canonical identity operation as MCP.

    The overrides go INSIDE the settings rather than beside them, and ``eval_name`` is the eval's
    bare registry name with tier and slice in the settings. Both are how a planned cell states the
    same facts, and the identity is a hash over exactly those fields -- so stating them the other
    way made ``submit hindsight beam:100k-smoke embedder=voyage/voyage-3`` a different experiment
    id from the identical cell composed through :func:`plan_sweep`. One cell, one id, whichever
    door composed it: that is what lets the CLI describe a cloud submission and have the server
    resolve the SAME experiment from it.

    Args:
        factories: ``(ref, make_adapter)`` for every target in this sweep.
        eval_name: The eval's registry name. NOT a ref carrying a preset -- ``beam``, never
            ``beam:100k-smoke``; the variant travels as ``params["tier"]`` and ``params["slice_"]``.
        overrides: Component overrides for the manifest, run-level ones already split out.
        params: The CLI's parsed parameters.
    """
    from memrank.application.planning import resolve_experiment
    from memrank.application.types import ExperimentSettings, JudgeSettings, ModelRef

    reader = params.get("reader")
    settings = ExperimentSettings(
        reader=ModelRef(provider="anthropic", model=reader) if reader else None,
        overrides=list(overrides),
        pricing_model=params["model"], slice=params["slice_"], tier=params["tier"],
        k=params["k"], repeats=params["repeats"], token_budget=params["token_budget"],
        seed=params["seed"], workers=params["workers"], units=params["unit"],
        judge=JudgeSettings(enabled=params["judge"], samples=params["judge_samples"],
                            cache=not params["no_judge_cache"],
                            allow_empty_coverage=params["allow_empty_judge_coverage"]))
    return {ref: resolve_experiment(ref, eval_name, settings) for ref, _ in factories}


def _experiment_metadata(experiment, plan_hash: str | None) -> dict[str, Any]:
    """Sanitized identity fields persisted with local and cloud run records."""
    if experiment is None:
        return {}
    return {"experiment_id": experiment.experiment_id,
            "experiment_label": experiment.label,
            "plan_hash": plan_hash,
            "experiment_spec": experiment.model_dump(mode="json")}


# Arms whose score is only meaningful once a reader answers. Unjudged, `no-context` is scored purely on
# retrieval, and since it retrieves nothing it can only score on NEGATIVE queries -- the ones a system
# is supposed to decline. That is a real number about the wrong thing.
_JUDGE_ONLY_TARGETS: frozenset[str] = frozenset({"no-context"})


def _warn_if_meaningless_unjudged(ref: str, *, judged: bool) -> None:
    """Say so when an arm is being run in a mode where its score cannot mean anything."""
    if judged or ref.split(":")[0] not in _JUDGE_ONLY_TARGETS:
        return
    style.note(
        f"{ref!r} retrieves nothing, so unjudged it scores only on negative queries "
        "(the ones a system should decline) and its number says nothing about memory. "
        "The no-memory arm tests the hypothesis only with --judge, where a reader answers "
        "from an empty context.")


def run_credentials(targets: list[str], *, judged: bool, provisioning: bool) -> list[str]:
    """Every credential THIS PROCESS needs before the run starts.

    The judge's key is a RUN-level requirement no target declares: ``required_secrets("word-overlap")``
    is empty, yet ``run baseline demo --judge`` cannot work without one. Missing that is why the
    original failure surfaced mid-run instead of before it. The judge is the harness making calls,
    so it is required wherever the harness runs.

    An engine's key is different: it is only memrank's business when memrank LAUNCHES the engine.

    Args:
        targets: Target refs in this run.
        judged: Whether the harness will call the judge.
        provisioning: Whether memrank starts the engine itself (``--on local``), and therefore has
            to hold what the engine needs in order to inject it. With an already-running engine
            (``--on none``) or an ECS task (where the execution role resolves SSM straight into the
            engine's own container), the engine already has its credentials and demanding them here
            asks for a key nobody in this process will use. That mistake killed the first mem0
            cloud run: the task ran with ``--on`` stripped, defaulted to ``none``, and refused to
            start for want of a key ECS had already injected next door.

    Returns:
        De-duplicated env-var names, in a stable order.
    """
    from memrank.judging.client import JUDGE_SECRET
    from memrank.targets import TargetNotFound, resolve_target
    from memrank.targets.catalog import required_secrets_for

    needed: list[str] = [JUDGE_SECRET] if judged else []
    if provisioning:
        for ref in targets:
            try:
                needed.extend(required_secrets_for(resolve_target(ref)))
            except TargetNotFound:
                continue  # a legacy --adapter name with no manifest; nothing to require
    return list(dict.fromkeys(needed))


def _ensure_run_credentials(*, targets: list[str], judged: bool, provisioning: bool) -> None:
    """Resolve every credential this process needs, asking for any that are missing."""
    from memrank.config import ConfigError, ensure_secrets

    try:
        ensure_secrets(run_credentials(targets, judged=judged, provisioning=provisioning))
    except ConfigError as exc:
        style.error(str(exc))
        raise typer.Exit(1) from exc


def _run_targets(*, target: str | None, benchmark_arg: str | None, overrides: list[str],
                 verify_engine: bool = True) -> tuple[str, list[tuple[str, AdapterFactory]]]:
    """Resolve the submission into ``(eval_name, [(label, make_adapter), ...])``.

    One form, since the ``--adapter``/``--benchmark`` pair retired: every target is a manifest
    ref, so what a run measured is described by the catalog rather than by whatever environment
    variables the process happened to hold.

    Raises:
        typer.BadParameter: If either half of the (target, eval) pair is missing.
    """
    if not target:
        raise typer.BadParameter("`submit <target> <eval>` needs a target ref "
                                 "(`memrank targets ls`)")
    if not benchmark_arg:
        raise typer.BadParameter("`submit <target> <eval>` needs an eval (`memrank evals ls`)")
    return benchmark_arg, _manifest_factories(target, overrides, verify_engine=verify_engine)


def _load_org_credentials(org: str) -> None:
    """Load ``org``'s BYOK credentials into this process for the duration of the run.

    Opt-in via ``--org`` rather than automatic on login, deliberately: a signed-in developer
    running an ordinary local benchmark must keep getting the wallet they already had. An
    implicit switch would change which key is billed based on invisible session state.
    """
    from memrank import config as _config
    from memrank.accounts.credentials import CredentialError, CredentialStore
    from memrank.accounts.run_secrets import OrgSecretsError, load_org_secrets
    from memrank.api_client import api_client

    try:
        token = CredentialStore().load()
    except CredentialError as exc:
        # The store refused rather than being empty -- a local permission problem, not a missing
        # login. Sending this user to `auth login` would have them redo work already done.
        raise typer.BadParameter(str(exc)) from exc
    if token is None:
        raise typer.BadParameter("--org needs a signed-in session; run `memrank auth login` first")
    try:
        with api_client(timeout=30, token=token) as http:
            _config.set_org_secrets(load_org_secrets(http, org))
    except OrgSecretsError as exc:
        raise typer.BadParameter(str(exc)) from exc

def _validate_positive(**values: int | None) -> None:
    """Reject non-positive methodology-critical knobs at the CLI boundary.

    ``None`` is not a value to police: an omitted flag whose default is derived from the run has
    nothing to validate here, and the derivation that replaces it cannot produce a bad number.
    """
    for name, value in values.items():
        if value is None:
            continue
        if value < 1:
            raise typer.BadParameter(f"--{name.replace('_', '-')} must be >= 1, got {value}")


def _judge_cfg_or_refuse(*, enabled: bool, samples: int, no_cache: bool,
                         allow_empty_coverage: bool = False, reader: str | None = None):
    """Build a JudgeConfig, or None when this run does not judge.

    It used to gate too: judging a non-synthetic benchmark was refused without `--ack-egress`. That
    consent was ceremony -- `--judge` already names an Anthropic judge, the browser path never asked
    at all (`web-gui/lib/run-config.ts`: "pressing Run ... IS the acknowledgement") and auto-supplied
    the acknowledgement to satisfy the server, and every documented CLI command carried both flags
    together. The DISCLOSURE below is what actually informed anyone, and it stays.
    """
    if not enabled:
        return None
    # What it sends and where responses land. What it will SPEND is said by `question_gates` once
    # the units are loaded, because only then is there arithmetic to report.
    style.say("--judge enabled: sends retrieved content + questions to Anthropic. Responses are "
              "cached locally under MEMRANK_CACHE_DIR / ~/.memrank and may echo retrieved memory "
              "content.")
    cfg = JudgeConfig(samples=samples, cache=not no_cache,
                      allow_empty_coverage=allow_empty_coverage)
    # `reader` selects the model that ANSWERS from retrieved memories. Deliberately independent of
    # --model, which only picks a row in cost.py's price table: coupling them would silently move
    # $/query during a reader ablation, destroying the comparability that yardstick exists for.
    return replace(cfg, answer_model=reader) if reader else cfg
