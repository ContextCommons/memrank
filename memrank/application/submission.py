"""One-cell asynchronous submission from a verified sweep plan."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from memrank.application.planning import plan_sweep
from memrank.application.types import (
    Experiment,
    ModelRef,
    Notice,
    SweepPlan,
    ToolEnvelope,
)


def _refused(code: str, message: str) -> ToolEnvelope:
    return ToolEnvelope(outcome="refused", refusals=[Notice(code=code, message=message)])


def _selected(plan: SweepPlan, experiment_id: str) -> Experiment | None:
    return next((item for item in plan.experiments if item.experiment_id == experiment_id), None)


def _verify(plan: SweepPlan, experiment_id: str, acknowledgements: list[str]) -> ToolEnvelope | Experiment:
    if plan.refusals:
        return ToolEnvelope(outcome="refused", refusals=plan.refusals)
    current = plan_sweep(plan.request)
    if not plan.plan_hash or current.plan_hash != plan.plan_hash:
        return _refused("stale_plan", "the plan no longer matches current catalogs or inputs")
    selected = _selected(current, experiment_id)
    if selected is None:
        return _refused("unknown_experiment", "experiment is not a member of this plan")
    needed = {warning.code for warning in current.warnings
              if experiment_id in warning.experiment_ids}
    missing = sorted(needed - set(acknowledgements))
    if missing:
        return _refused("acknowledgement_required",
                        f"acknowledge these warning codes: {', '.join(missing)}")
    return selected


def remote_params(experiment: Experiment, plan: SweepPlan,
                  acknowledgements: list[str], *, verbose: bool = False,
                  judge_workers: int | None = None,
                  fail_fast: bool = False) -> dict[str, Any]:
    """One planned experiment as the flat parameters every submission path renders from.

    Public because it is the first half of the config-to-container translation, and the half a
    browser needs: the API plans a structured config, calls this, and hands the result to
    :func:`memrank.placement.remote_argv.remote_argv`. The CLI reaches the same dict from Typer's
    parsed flags instead. Two doors, one room -- and the API tests assert both render identical argv.

    Args:
        experiment: The resolved cell.
        plan: The plan it belongs to, for routing.
        acknowledgements: Warning codes the caller accepted.
        verbose: How loudly the run narrates itself. Keyword rather than a settings field
            because it changes what is PRINTED and nothing that is measured -- putting it in
            ``ExperimentSettings`` would make two identical measurements two experiments.
        judge_workers: Judge concurrency, or ``None`` to leave it unstated and let the task keep
            its own default. Same reasoning as ``verbose``: independent calls with an ordered
            reduce, so metrics are identical at any value.
        fail_fast: Whether the task stops at the first unit that fails. Keyword for the same
            reason as ``verbose``: it changes how much of the evaluation survives a failure,
            never what any surviving unit measures.
    """
    from memrank.benchmarks.refs import compose_ref

    settings = experiment.settings
    reader = settings.reader.model if settings.reader else None
    overrides = [*experiment.overrides, *([f"reader={reader}"] if reader else [])]
    # `benchmark_arg` must be the CANONICAL eval ref: the argv renderers no longer forward
    # tier/slice as flags (they are retired), so a bare eval name submitted beside tier/slice
    # settings would otherwise canonicalise to the default variant -- a different evaluation
    # than the one planned. `tier`/`slice` stay in the dict for the submitter's own bookkeeping
    # (bench kwargs, run-id minting, row fields); remote_argv classifies them LOCAL_ONLY.
    return {
        "target": experiment.target_ref,
        "benchmark_arg": compose_ref(experiment.eval_ref,
                                     tier=settings.tier, slice=settings.slice),
        "overrides": overrides, "tier": settings.tier, "slice": settings.slice,
        "k": settings.k, "repeats": settings.repeats, "model": settings.pricing_model,
        "token_budget": settings.token_budget, "seed": settings.seed,
        "workers": settings.workers, "unit": settings.units, "verbose": verbose,
        "judge_workers": judge_workers, "fail_fast": fail_fast,
        # Already resolved by the door that took the request -- never None here. `remote_argv`
        # renders it as `--judge` or `--no-judge`, so the task is told the decision rather than
        # re-deriving it from its own build of the benchmark.
        "judge": settings.judge.enabled, "judge_samples": settings.judge.samples,
        "no_judge_cache": not settings.judge.cache,
        "allow_empty_judge_coverage": settings.judge.allow_empty_coverage,
        "on": plan.request.routing.placement, "output_dir": Path("results"),
        "org": plan.request.routing.org,
    }


def _stated(role: str, ref: ModelRef | None, tokens: list[str]) -> dict[str, Any] | None:
    """``ref`` when the CALLER named this component, ``None`` when the manifest supplied it.

    ``resolve_experiment`` fills ``engine_llm`` and ``embedder`` in from the resolved manifest, so
    the value alone cannot say which of the two happened -- and the difference is the whole of the
    equivalence: a component restated here renders an override token the CLI never rendered, and
    the two commands stop being the same bytes. What does say is the override list, whose typed
    tail is exactly what the caller named.
    """
    if ref is None or not any(token.startswith(f"{role}=") for token in tokens):
        return None
    return ref.model_dump(mode="json")


def remote_config(experiment: Experiment, *, verbose: bool,
                  judge_workers: int | None = None,
                  fail_fast: bool = False,
                  acknowledged: list[str] | None = None) -> dict[str, Any]:
    """One planned experiment as the ``config`` a cloud submission carries.

    The structural twin of :func:`remote_params`: the same cell, DESCRIBED rather than rendered.
    The server re-plans it and renders the command with the same renderer, which is what lets the
    platform's flag vocabulary change without waiting for every installed CLI to catch up.

    Every field is the one the caller stated, never the one resolution supplied -- see
    :func:`_stated` -- so ``plan_sweep`` on the other side of the wire resolves the same
    experiment id and ``remote_argv`` renders byte-identical argv.

    Args:
        experiment: The resolved cell.
        verbose: How loudly the run narrates itself.
        judge_workers: Judge concurrency, or ``None`` to leave the field out and take the API
            model's own default. Unlike argv, an unstated field here is answered by the server
            that renders the command rather than by whatever the task image defaults to.
        acknowledged: Warning codes the caller accepted.
    """
    settings = experiment.settings
    named = experiment.overrides[len(settings.overrides):]
    config = {
        "target_ref": experiment.target_ref, "eval_ref": experiment.eval_ref,
        "engine_llm": _stated("llm", settings.engine_llm, named),
        "embedder": _stated("embedder", settings.embedder, named),
        "reader": settings.reader.model_dump(mode="json") if settings.reader else None,
        "overrides": list(settings.overrides),
        "pricing_model": settings.pricing_model,
        "slice": settings.slice, "tier": settings.tier,
        "k": settings.k, "repeats": settings.repeats,
        "token_budget": settings.token_budget, "seed": settings.seed,
        "workers": settings.workers, "units": list(settings.units),
        "judge": settings.judge.model_dump(mode="json"),
        "verbose": verbose, "fail_fast": fail_fast,
        "acknowledged": list(acknowledged or []),
    }
    if judge_workers is not None:
        config["judge_workers"] = judge_workers
    return config


def _metadata(plan: SweepPlan, experiment: Experiment) -> dict[str, Any]:
    return {"experiment_id": experiment.experiment_id, "experiment_label": experiment.label,
            "plan_hash": plan.plan_hash, "experiment_spec": experiment.model_dump(mode="json")}


def _submit_local(plan: SweepPlan, experiment: Experiment,
                  acknowledgements: list[str]) -> dict[str, Any]:
    # Imported lazily so discovery and planning do not load the execution graph.
    from memrank.orchestration import placement_gate, resolve, sweep

    params = remote_params(experiment, plan, acknowledgements)
    if params["org"]:
        resolve._load_org_credentials(params["org"])
    component, run_level = resolve._split_overrides(params["overrides"])
    resolve._validate_reader(run_level.get("reader"), judged=params["judge"])
    benchmark, factories = resolve._run_targets(target=experiment.target_ref,
                                                benchmark_arg=experiment.eval_ref,
                                                overrides=component,
                                                verify_engine=params["on"] == "none")
    placement_gate._require_placement_ready(params["on"])
    resolve._ensure_run_credentials(targets=[experiment.target_ref], judged=params["judge"],
                                    provisioning=params["on"] == "local")
    bench_kwargs = {"k": params["k"]}
    if benchmark == "beam" and params["tier"]:
        bench_kwargs["tier"] = params["tier"]
    if params["slice"]:
        bench_kwargs["slice"] = params["slice"]
    judge_cfg = resolve._judge_cfg_or_refuse(
        enabled=params["judge"], samples=params["judge_samples"],
        no_cache=params["no_judge_cache"],
        allow_empty_coverage=params["allow_empty_judge_coverage"],
        reader=run_level.get("reader"))
    # One cell, so one judge counter: `targets=1` is the truth here rather than a simplification.
    units, judge_cfg = sweep.question_gates(
        benchmark=benchmark, bench_kwargs=bench_kwargs, slice_=params["slice"],
        judge_cfg=judge_cfg, unit=params["unit"], targets=1)
    opts = sweep._ExecOpts(benchmark=benchmark, eval_ref=experiment.eval_ref,
                            bench_kwargs=bench_kwargs,
                            tier=params["tier"], slice_=params["slice"], on=params["on"],
                            output_dir=params["output_dir"], k=params["k"],
                            repeats=params["repeats"], model=params["model"],
                            token_budget=params["token_budget"], seed=params["seed"],
                            workers=params["workers"], verbose=False, judged=params["judge"],
                            judge_cfg=judge_cfg, units=units,
                            experiments={experiment.target_ref: experiment})
    sweep._sweep_gates(factories, opts)
    runs = sweep._mint_local_runs(factories, benchmark=benchmark, slice_=params["slice"],
                                  tier=params["tier"], run_ids=None,
                                  experiments={experiment.target_ref: experiment})
    runs[0].status.annotate(**_metadata(plan, experiment))
    try:
        sweep._spawn_child(params, runs)
    except BaseException:
        sweep._mark_unstarted(runs, reason="the background spawn failed")
        raise
    return {"run_id": runs[0].run_dir.name, "experiment_id": experiment.experiment_id,
            "state": "queued", "placement": params["on"]}


def _bench_kwargs(experiment: Experiment, benchmark: str, params: dict[str, Any]) -> dict[str, Any]:
    """The benchmark constructor arguments this experiment names."""
    kwargs: dict[str, Any] = {"k": params["k"]}
    if benchmark == "beam" and params["tier"]:
        kwargs["tier"] = params["tier"]
    if params["slice"]:
        kwargs["slice"] = params["slice"]
    return kwargs


def _judge_gate_refusal(experiment: Experiment, params: dict[str, Any]) -> ToolEnvelope | None:
    """Refuse here what the task would otherwise discover after it launched, or None to proceed.

    Judge coverage, not budget: there is no ceiling left to check, but a judged run whose queries
    are all outside the judge's supported categories scores nothing, and finding that out costs a
    Fargate task. Returns an envelope rather than raising because ``typer.BadParameter`` is the
    CLI's shape and this door answers a browser.
    """
    from memrank.benchmarks.refs import parse_eval_ref
    from memrank.orchestration import resolve, sweep

    if not params["judge"]:
        return None
    benchmark, _, _ = parse_eval_ref(experiment.eval_ref)
    kwargs = _bench_kwargs(experiment, benchmark, params)
    cfg = resolve._judge_cfg_or_refuse(
        enabled=True, samples=params["judge_samples"],
        no_cache=params["no_judge_cache"],
        allow_empty_coverage=params["allow_empty_judge_coverage"])
    try:
        # One task per experiment, so one counter: `targets=1` is what the estimate reports.
        sweep.question_gates(benchmark=benchmark, bench_kwargs=kwargs,
                             slice_=params["slice"], judge_cfg=cfg,
                             unit=params["unit"], targets=1)
    except Exception as exc:  # typer.BadParameter, and anything the loader raises
        return _refused("judge_coverage", str(exc))
    return None


def _submit_cloud(plan: SweepPlan, experiment: Experiment,
                  acknowledgements: list[str]) -> dict[str, Any] | ToolEnvelope:
    from memrank.orchestration import cloud
    from memrank.placement import run_api_client
    from memrank.targets.portability import local_only, unportable_message

    # The same guard the CLI applies, at the other cloud door. A target defined only on the
    # submitting machine -- `targets.path`, or the operator's own directory -- resolves here and
    # nowhere else, so the task launches and its container exits on the first line with "unknown
    # target". Refused rather than raised: this path answers in envelopes.
    local = local_only([experiment.target_ref])
    if local:
        return _refused("target_not_portable", unportable_message(local))
    params = remote_params(experiment, plan, acknowledgements)
    # Judged-run refusals that are arithmetic over the benchmark, made here rather than in the task.
    # They are equally true either side of the wire, and asking them only in the task is what cost
    # four Fargate tasks on 2026-08-13.
    refusal = _judge_gate_refusal(experiment, params)
    if refusal is not None:
        return refusal
    # WHAT to measure, not the command that measures it: the server plans this config and renders
    # the argv with the same renderer, so this door speaks no flag vocabulary at all. The
    # positional facts and the experiment identity come off the server's own resolution -- there is
    # nothing for a submitter's copy of them to disagree with.
    #
    # No image is named either: the server launches its own pinned platform harness. Naming one was
    # the harness-developer door and is gone -- it assumed a checkout, a Docker daemon and ECR
    # rights, and it swapped only the image while the API still rendered the task command.
    payload = {"config": remote_config(experiment, verbose=False,
                                       acknowledged=acknowledgements)}
    with cloud._api_client() as http:
        record = run_api_client.submit_run(http, params["org"], payload)
        cloud._record_submission(http, params["org"], experiment.target_ref, record,
                                 benchmark=experiment.eval_ref,
                                 slice_=experiment.settings.slice,
                                 tier=experiment.settings.tier)
    return {"run_id": record["id"], "experiment_id": experiment.experiment_id,
            "state": record["state"], "placement": "cloud"}


def submit_experiment(plan: SweepPlan, experiment_id: str,
                      acknowledgements: list[str] | None = None) -> ToolEnvelope:
    """Revalidate and asynchronously submit exactly one planned experiment."""
    acknowledgements = acknowledgements or []
    selected = _verify(plan, experiment_id, acknowledgements)
    if isinstance(selected, ToolEnvelope):
        return selected
    if plan.request.routing.placement == "cloud":
        data = _submit_cloud(plan, selected, acknowledgements)
        # A refusal travels as itself, the way _verify's does: the cloud door can decline before it
        # submits anything, and wrapping that in `data` would report a refusal as a successful run.
        if isinstance(data, ToolEnvelope):
            return data
    else:
        data = _submit_local(plan, selected, acknowledgements)
    return ToolEnvelope(data=data)
