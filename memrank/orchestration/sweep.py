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
"""The local sweep: mint runs, execute each cell under its own heartbeat, persist,
sync, and spawn the detached child.

Ordering here IS the contract (`tests/orchestration/test_golden_run.py`, `tests/orchestration/test_local_sweep.py`):
checkpoint before judging, writing -> summary -> MLflow -> done -> auto-sync, failure marking
on `Exception`/`SweepKilled` but never `KeyboardInterrupt`, queued siblings marked when a
sweep dies. Typer exceptions still surface from a few gates -- scheduled for conversion to
`MemrankError` subclasses at the CLI boundary (tech-debt.md).
"""
from __future__ import annotations

import json
import shutil
import signal
import subprocess
import sys
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import typer

from memrank import __version__, rate_limit
from memrank.adapters.preflight import PreflightError, preflight
from memrank.benchmarks import get_benchmark
from memrank.benchmarks.refs import canonical_ref
from memrank.core import QUALITY_METRIC_LABELS
from memrank.evaluation import result as result_schema
from memrank.evaluation.cell import (
    AdapterFactory,
    _assert_retrieved_something,
    _close_adapter,
    cell_applicable,
    run_cell,
)
from memrank.evaluation.constants import DEFAULT_JUDGE_WORKERS
from memrank.evaluation.judge_stage import (
    EmptyJudgeCoverage,
    assert_judge_coverage,
    judge_cost_estimate,
)
from memrank.evaluation.observer import NULL_OBSERVER
from memrank.judging.client import build_completer
from memrank.metrics import cost, headline
from memrank.orchestration.observers import RunStatusObserver
from memrank.orchestration.placement_gate import _environment_patch, placement_for
from memrank.orchestration.resolve import _experiment_metadata, _warn_if_meaningless_unjudged
from memrank.placement.remote_argv import (
    REMOTE_REPEATED,
    REMOTE_SWITCHES,
    REMOTE_TRISTATE,
    REMOTE_VALUE_FLAGS,
)
from memrank.rate_limit import RateLimitGate
from memrank.runs import checkpoint, push, registry
from memrank.runs import status as run_status
from memrank.term import style


def _echo_per_query_outcomes(adapter_name: str, benchmark_name: str,
                             per_query: list[dict[str, Any]]) -> None:
    """Verbose: one line per query -- HIT/MISS, what matched, and the query text."""
    for row in per_query:
        status = style.good("HIT ") if row.get("hit") else style.bad("MISS")
        matched = row.get("matched_doc_id") or row.get("matched_span") or "-"
        text = (row.get("text") or "").replace("\n", " ")[:60]
        style.say(f"  [{adapter_name} × {benchmark_name}] {row.get('query_id')} "
                  f"{status} matched={matched}  \"{text}\"")


def _filter_units(units: list[Any], selectors: list[str]) -> list[Any]:
    """Keep units matching ANY selector, in the units' original order (deduped).

    A selector that is all-digits is a 0-based index into ``units``; otherwise it is an
    exact-or-substring match on ``unit_id``. Fails loud (typer.BadParameter) if any
    selector matches nothing -- never silently run zero units."""
    n = len(units)
    keep: set[int] = set()
    for sel in selectors:
        if sel.isdigit():
            idx = int(sel)
            if not 0 <= idx < n:
                raise typer.BadParameter(f"--unit index {idx} out of range (0..{n - 1})")
            keep.add(idx)
            continue
        matched = [i for i, u in enumerate(units) if u.unit_id == sel or sel in u.unit_id]
        if not matched:
            raise typer.BadParameter(f"--unit {sel!r} matched no unit_id")
        keep.update(matched)
    return [u for i, u in enumerate(units) if i in keep]


def _preload_tokenizer_with_notice() -> None:
    """Load the tokenizer before benchmarking starts, announcing the potential
    first-run vocabulary download so a cold run is attributable, not a silent hang."""
    style.say(
        f"preparing tokenizer {cost.ENCODING_NAME!r} "
        "(first run downloads ~4 MB, cached afterwards) ...")
    cost.preload_tokenizer()
    style.say("tokenizer ready")


_IDENTITY_HASH_CHARS = 6


def _safe_label(ref: str) -> str:
    """A target ref as a filesystem-safe stem (HELM does the same for its run directories)."""
    return ref.replace(":", "-").replace("/", "-")


def run_identity(label: str, config_hash: str | None) -> str:
    """A readable name for one configured run: ``hindsight-amb-4c1f9a``.

    Readable stem plus a short content hash, the shape Kustomize uses for generated ConfigMaps --
    no surveyed system presents a bare hash as a primary name. Two runs sharing an identity used the
    same configuration; two that differ did not, whatever their labels say.
    """
    stem = _safe_label(label)
    return f"{stem}-{config_hash[:_IDENTITY_HASH_CHARS]}" if config_hash else stem

@dataclass(frozen=True)
class _ExecOpts:
    """The knobs one target's local execution needs -- bundled so the per-run helpers stay small."""

    benchmark: str
    #: The canonical eval ref (`beam:100k-smoke`). `benchmark` stays the REGISTRY key because the
    #: registry, the receipt and the leaderboard all key on it; this names the evaluation, and is
    #: what artifacts are filed under so two variants cannot overwrite one another.
    eval_ref: str
    bench_kwargs: dict[str, Any]
    tier: str | None
    slice_: str | None
    on: str
    output_dir: Path
    k: int
    repeats: int
    model: str
    token_budget: int
    seed: int
    workers: int
    verbose: bool
    judged: bool
    #: Concurrency for the judge stage. Defaulted rather than positional so every existing
    #: construction of this bundle keeps working, and defaulted to ``DEFAULT_JUDGE_WORKERS`` so a
    #: caller that does not set it gets the same judge concurrency as every other surface. The
    #: value is a throughput knob only: metrics are identical at any worker count.
    judge_workers: int = DEFAULT_JUDGE_WORKERS
    #: Whether one unit's failure ends the cell. Defaulted to False for the same reason
    #: ``run_cell`` defaults it there: a run that dies on unit 12 of 50 throws away the eleven
    #: that worked, and how often a unit fails is itself something the artifact should record.
    fail_fast: bool = False
    judge_cfg: Any = None
    judge_runtime: Any = None
    units: Any = None
    experiments: Any = None
    #: The `rate_limit.GateScope` the sweep's shared judge runtime resolves through, so a
    #: judge pause lands in the CURRENT cell's gate (and its receipt). Set by `_sweep_gates`.
    gate_scope: Any = None


@dataclass(frozen=True)
class _LocalRun:
    """One pre-minted run: a target ref bound to its own run dir and heartbeat."""

    ref: str
    make_adapter: AdapterFactory
    run_dir: Path
    status: run_status.RunStatus


def question_gates(*, benchmark: str, bench_kwargs: dict[str, Any], slice_: str | None,
                   judge_cfg, unit: list[str], targets: int):
    """The refusals that are arithmetic over the benchmark: unit filter, judge coverage, budget.

    Split out from :func:`_sweep_gates` because these are the only ones a SUBMISSION can make.
    Preflighting engines needs engines that are not on this machine and building a completer needs
    credentials the task gets from SSM, but "does this cap fit this benchmark" is a sum over loaded
    units and is equally true wherever it is asked.

    That distinction cost four Fargate tasks on 2026-08-13. `--on cloud` returned before
    ``_sweep_gates`` ran, so the cap check happened inside the task, after image pull and a 274 MB
    dataset download -- four times, for arithmetic that was knowable for free.

    ``targets`` is the fan-out width the cap is shared across, and it differs by placement: a local
    sweep runs every engine under ONE counter, while each cloud target is a separate task with its
    own. Passing ``len(factories)`` for a submission would over-refuse a sweep that is really N
    independent budgets.

    Returns:
        ``(units, judge_cfg)`` -- the units this run will use, filtered, and the judge config
        unchanged. Takes primitives rather than an ``_ExecOpts`` because a submission has no local
        execution to describe, and assembling a bundle of placeholder knobs to reach one field is
        how a fake value ends up somewhere real.
    """
    units = None
    bench = None
    if unit or judge_cfg is not None:
        bench = get_benchmark(benchmark, **bench_kwargs)
        units = bench.load()
    if unit:
        assert units is not None  # loaded above whenever `unit` is set
        units = _filter_units(units, unit)  # fails loud if nothing matches
    if judge_cfg is not None:  # refuse empty coverage AFTER filtering (reflects the run set)
        assert bench is not None  # loaded above whenever `judge_cfg` is set
        try:
            # `bench` is loaded above whenever judge_cfg is set, so its shape is available --
            # and must be passed, exactly as the cost estimate below does. Omitting it was the
            # bug: the gate fell back to the demo benchmark's vocabulary.
            assert_judge_coverage(units, judge_cfg, benchmark, bench.judge_shape())
        except EmptyJudgeCoverage as exc:
            raise typer.BadParameter(str(exc)) from exc
        # Said, not enforced. The estimate is why a caller would choose a smaller slice; it refuses
        # nothing, because bounding the judge would bound the measurement rather than the run.
        style.say(judge_cost_estimate(units, judge_cfg, benchmark, slice_, targets=targets,
                                      shape=bench.judge_shape()))
    return units, judge_cfg


def _sweep_gates(factories: list[tuple[str, AdapterFactory]], opts: _ExecOpts) -> _ExecOpts:
    """The sweep-wide refusals that need THIS machine: engines answering, a tokenizer, a key.

    The other half -- unit filter, judge coverage, judge budget -- is :func:`question_gates`, run by
    the command before it forks on placement, because those are arithmetic over the benchmark and a
    submission can make them too. Keeping them here was the defect: `--on cloud` returned before
    this function, so a cap that could not finish was discovered inside the Fargate task.

    Run by the submitting parent (so a refusal lands on a terminal and nothing is minted) AND by the
    executing child -- the cloud container enters execute mode with no parent to have gated for it,
    so execute mode cannot assume them. One implementation serves both.
    """
    _preflight_engines(factories, opts.benchmark, on=opts.on)
    _preload_tokenizer_with_notice()
    # Built ONCE and shared across the fan-out: the judge call cap + counter deliberately span
    # every target rather than resetting per engine, which is why `question_gates` costed this
    # sweep with `targets=len(factories)`. The scope is how a runtime that outlives any one
    # cell still pauses on the CURRENT cell's gate -- `_execute_local_run` enters it per cell.
    scope = rate_limit.GateScope()
    judge_runtime = (build_completer(opts.judge_cfg, gate=scope.get)
                     if opts.judge_cfg is not None else None)
    return replace(opts, judge_runtime=judge_runtime, gate_scope=scope)


class SweepKilled(BaseException):
    """Raised by the SIGTERM handler: `memrank kill` ends this sweep.

    A ``BaseException`` so no adapter's ``except Exception`` can swallow the kill on its
    way up -- the sweep unwinds through its context managers (placements tear down) and
    every run records its end, exactly as any other abort does.
    """


def _install_kill_handler() -> None:
    """Arm execute mode for `memrank kill`: SIGTERM unwinds the sweep via SweepKilled.

    The disposition is reset before raising, so a SECOND SIGTERM hard-kills a child stuck
    unwinding -- that reset is the whole escalation policy.
    """
    def _on_sigterm(signum: int, frame: Any) -> None:
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        raise SweepKilled()

    signal.signal(signal.SIGTERM, _on_sigterm)


def _run_local_sweep(factories: list[tuple[str, AdapterFactory]], opts: _ExecOpts, *,
                     run_ids: list[str]) -> None:
    """Execute mode: run a sweep in THIS process -- one run per target, in order, blocking.

    Entered only via the hidden ``--run-id``: by the background child a submission spawned, or
    by the cloud container (whose rendered command chains ``&& upload`` and therefore needs a
    submit that blocks).
    """
    _install_kill_handler()
    opts = _sweep_gates(factories, opts)
    runs = _mint_local_runs(factories, benchmark=opts.benchmark, slice_=opts.slice_,
                            tier=opts.tier, run_ids=run_ids, experiments=opts.experiments)
    cells: list[dict[str, Any]] = []
    try:
        for run in runs:
            cells.append(_execute_local_run(run, opts))
    except SweepKilled as exc:  # the active run is marked; exit cleanly, not with a traceback
        _mark_unstarted(runs, reason="sweep killed", sync=True)
        raise typer.Exit(1) from exc
    except BaseException as exc:  # the failed run is already marked; siblings must not lie
        _mark_unstarted(runs, reason=f"sweep aborted after {run.ref!r} failed",
                        sync=isinstance(exc, Exception))
        raise
    # The overwritten "latest" view in --output-dir still covers the whole sweep; each run dir
    # already holds its own one-cell summary.
    _write_summary(opts.output_dir, opts.eval_ref, opts.tier, opts.slice_, opts.k,
                   opts.seed, cells)


def _submit_local_sweep(factories: list[tuple[str, AdapterFactory]], opts: _ExecOpts, *,
                        params: dict[str, Any]) -> None:
    """Submit mode: gate loudly, mint one run per target, hand the sweep to ONE background
    child, print the bare ids on stdout, and return -- the interface model's only lifecycle.

    The gates' results are discarded here; the child rebuilds them (tokenizer and dataset
    caches make the second pass cheap). What must not be discarded is their refusals.
    """
    _sweep_gates(factories, opts)
    runs = _mint_local_runs(factories, benchmark=opts.benchmark, slice_=opts.slice_,
                            tier=opts.tier, run_ids=None, experiments=opts.experiments)
    try:
        _spawn_child(params, runs)
    except BaseException:
        _mark_unstarted(runs, reason="the background spawn failed")
        raise
    for run in runs:
        style.out(run.run_dir.name)  # bare ids on stdout: the machine-readable handle


def _mint_local_runs(factories: list[tuple[str, AdapterFactory]], *, benchmark: str,
                     slice_: str | None, tier: str | None,
                     run_ids: list[str] | None, experiments=None) -> list[_LocalRun]:
    """One run (dir + id + single-target heartbeat) per target, minted before anything executes.

    Printing every id up front is the interface model's contract ("one run ID per target,
    printed immediately"), and the ``queued`` heartbeats are what let `memrank ps` show the
    not-yet-started tail of a sweep. One process executes them in order, so the sweep's log
    belongs to the FIRST run; each sibling records that under ``log_run`` for `memrank logs`.
    """
    if run_ids is not None and len(run_ids) != len(factories):
        raise typer.BadParameter(f"--run-id was given {len(run_ids)} time(s) for "
                                 f"{len(factories)} target(s); a sweep needs one per target")
    runs: list[_LocalRun] = []
    for index, (ref, make_adapter) in enumerate(factories):
        run_dir = registry.new_run_dir(benchmark, slice_, tier,
                                           run_id=run_ids[index] if run_ids else None)
        status = run_status.RunStatus.create(
            run_dir, target=ref, benchmark=benchmark, slice_=slice_,
            eval_ref=canonical_ref(benchmark, {"tier": tier, "slice": slice_}))
        experiment = (experiments or {}).get(ref)
        if experiment is not None:
            status.annotate(**_experiment_metadata(experiment, None))
        status.update(state="queued", message="queued")
        style.say(f"run {run_dir.name}  ({ref} × {benchmark})")
        runs.append(_LocalRun(ref, make_adapter, run_dir, status))
    for sibling in runs[1:]:
        sibling.status.annotate(log_run=runs[0].run_dir.name)
    return runs


def _execute_local_run(run: _LocalRun, opts: _ExecOpts) -> dict[str, Any]:
    """Run one target's cell under its own heartbeat; never leave it non-terminal."""
    # The run's OWN observer and gate, built here and handed in -- one explicit object per
    # run where a process-global "current run" used to be.
    observer = RunStatusObserver(run.status, verbose=opts.verbose)
    cell_gate = RateLimitGate()
    scope = opts.gate_scope if opts.gate_scope is not None else rate_limit.GateScope()
    try:
        _warn_if_meaningless_unjudged(run.ref, judged=opts.judged)
        run.status.update(state="running", message=f"running {run.ref} × {opts.benchmark}")
        bench = get_benchmark(opts.benchmark, **opts.bench_kwargs)
        from memrank.placement.cloud_launch import resolve_target_or_inprocess
        placement_target = resolve_target_or_inprocess(run.ref)
        # Stable per-cell path so `report --result-file` is predictable; re-running
        # replaces it (the run registry keeps the non-overwritten history).
        out_path = opts.output_dir / f"{_safe_label(run.ref)}__{_safe_label(opts.eval_ref)}.json"
        with placement_for(placement_target, opts.on, run_dir=run.run_dir) as placement:
            endpoint = placement.provision(placement_target)
            # The adapter reads its URL from the environment at construction, so the DISCOVERED
            # port must be in place before the factory runs. Never a guessed one.
            with _environment_patch(endpoint.adapter_env), scope.use(cell_gate):
                # Receipt and worker adapters must observe the same placement facts.
                aggregated = _run_and_persist(
                    run.make_adapter(), bench, k=opts.k, repeats=opts.repeats,
                    run_id_prefix=f"{_safe_label(run.ref)}-{uuid.uuid4().hex[:8]}",
                    model=opts.model, token_budget=opts.token_budget, seed=opts.seed,
                    judge=opts.judge_cfg, judge_runtime=opts.judge_runtime,
                    out_path=out_path, units=opts.units, workers=opts.workers,
                    judge_workers=opts.judge_workers, fail_fast=opts.fail_fast,
                    make_adapter=run.make_adapter,
                    observer=observer, gate=cell_gate, target=run.ref)
        _echo_cell_outcome(run.ref, aggregated, opts, out_path)
        _record_local_run(run, out_path, aggregated, opts)
        return aggregated
    except BaseException as exc:  # mark failed (never leave a run "running"), then re-raise
        error = ("killed by user" if isinstance(exc, SweepKilled)
                 else f"{type(exc).__name__}: {exc}")
        run.status.update(state="failed", error=error)
        # The org must learn a run ended even when it ended badly -- but only when the ending
        # IS an outcome. An eval failure (Exception) and a `memrank kill` (SweepKilled, whose
        # placements already unwound) both report; an operator interrupt does not, because an
        # HTTP request inside a KeyboardInterrupt unwind is worse than a pending record.
        if isinstance(exc, (Exception, SweepKilled)):
            push.auto_sync(run.run_dir)
        raise


def _echo_cell_outcome(ref: str, aggregated: dict[str, Any], opts: _ExecOpts,
                       out_path: Path) -> None:
    """Print one cell's outcome line (or why the pairing did not apply)."""
    if aggregated.get("status") == "not_applicable":
        style.say(f"[{ref} × {opts.benchmark}] not applicable: {aggregated['reason']}")
        return
    if opts.verbose:
        _echo_per_query_outcomes(ref, opts.benchmark, aggregated.get("per_query", []))
    # The judged score when there is one: on locomo, longmemeval and beam it is the ONLY quality
    # number the run produced, and echoing the withheld composite instead told a judged run it
    # had measured nothing.
    head = headline.cell_headline(aggregated)
    recall = "n/a (judge required)" if head.value is None else f"{head.value:.3f}"
    label = QUALITY_METRIC_LABELS.get(head.kind, "score")
    identity = run_identity(ref, aggregated.get("receipt", {}).get("config_hash"))
    style.say(f"[{identity} × {opts.benchmark}] {label}={recall} -> {out_path}")


def _record_local_run(run: _LocalRun, out_path: Path, aggregated: dict[str, Any],
                      opts: _ExecOpts) -> None:
    """Snapshot one run's cell and one-cell summary into its dir, and mark it done."""
    if out_path.exists():
        shutil.copy2(out_path, run.run_dir / out_path.name)
    run.status.update(state="writing", message="writing summary")
    _write_summary(run.run_dir, opts.eval_ref, opts.tier, opts.slice_, opts.k, opts.seed,
                   [aggregated], announce=False)
    _mirror_to_mlflow(run.run_dir)
    run.status.update(state="done", message="run recorded", pct=100)
    style.say(f"run recorded -> {run.run_dir}")
    # After `done`, so a crash mid-sync leaves a run that is correctly pending rather than
    # one that never finished. Never raises: see its docstring for why this one degrades
    # where the MLflow mirror above refuses.
    push.auto_sync(run.run_dir)


def _mark_unstarted(runs: list[_LocalRun], *, reason: str, sync: bool = False) -> None:
    """A queued run that lost whatever was going to start it must not stay queued forever.

    ``sync`` mirrors the caller's judgment on whether the ending is an outcome to report --
    false during an operator interrupt, where an HTTP request has no place in the unwind.
    """
    for run in runs:
        if run.status.data.get("state") == "queued":
            run.status.update(state="failed", error=f"not started: {reason}")
            if sync:
                push.auto_sync(run.run_dir)


def _preflight_engines(factories, benchmark_name: str, *, on: str) -> None:
    """Refuse before any cell runs when an engine this run needs is not answering.

    The same gate ``compare`` has had all along (memrank/analysis/compare.py), for the same reason it
    states: a later engine's failure must not leave an earlier one having already ingested and
    spent tokens. Without it a sweep's second target being down is discovered after the first
    has paid, and the user sees a raw connection traceback from deep inside ingest rather than
    the name of the engine that is missing.

    Only for ``--on none``, where nothing has established that an engine exists: ``--on local``
    provisions its own and blocks on readiness, and ``--on cloud`` runs nothing here. Cells that
    would not run are skipped at the same ``cell_applicable`` chokepoint the run loop uses, so an
    inapplicable pairing never touches a live backend to be told it was not going to run.
    """
    if on != "none":
        return
    # Cheap to construct: benchmarks load their data lazily, and applicability reads one
    # attribute. Slices and tiers change what runs, never whether an engine is reachable.
    bench = get_benchmark(benchmark_name)
    for _, factory in factories:
        adapter = factory()
        try:
            if cell_applicable(adapter, bench):
                preflight(adapter)
        except PreflightError as exc:
            style.error(str(exc))
            raise typer.Exit(code=1) from exc
        finally:
            _close_adapter(adapter)


def _child_argv(params: dict[str, Any], run_ids: list[str]) -> list[str]:
    """The argv the background child runs, rebuilt from the parsed parameters.

    Rebuilt for the same reason ``remote_argv`` is: ``sys.argv`` is not reliably this
    command's own (a ``CliRunner`` invocation proved it), and scraping it silently drops any
    flag whose spelling a filter does not anticipate. Unlike a cloud task, the child keeps the
    local-only flags -- the resolved ``--on`` (never re-defaulted, so a config change between
    spawn and start cannot reroute it), ``--output-dir``, ``--org`` -- plus one ``--run-id``
    per minted run, which is what puts it in execute mode.
    """
    argv = ["submit", params["target"], params["benchmark_arg"],
            *(params.get("overrides") or [])]
    for name, flag in REMOTE_VALUE_FLAGS:
        if params.get(name) is not None:
            argv += [flag, str(params[name])]
    for name, flag in REMOTE_SWITCHES:
        if params.get(name):
            argv.append(flag)
    for name, when_true, when_false in REMOTE_TRISTATE:
        argv.append(when_true if params.get(name) else when_false)
    for name, flag in REMOTE_REPEATED:
        for value in params.get(name) or []:
            argv += [flag, str(value)]
    argv += ["--output-dir", str(params["output_dir"])]
    for name, flag in (("on", "--on"), ("org", "--org")):
        if params.get(name) is not None:
            argv += [flag, str(params[name])]
    if params.get("target_digest") is not None:
        argv += ["--target-digest", str(params["target_digest"])]
    for rid in run_ids:
        argv += ["--run-id", rid]
    return argv


def _spawn_child(params: dict[str, Any], runs: list[_LocalRun]) -> None:
    """One background child executes the whole sweep; the submitting parent returns.

    One child has one stdout, so run.log lives in the FIRST run's dir -- minting already
    pointed each sibling's ``log_run`` there. The heartbeats adopt the child's pid right
    after the spawn: the parent's pid dies with the parent, and a queued run attributed to
    a dead process would classify as stale before the child ever re-stamps it.
    """
    argv = _child_argv(params, [run.run_dir.name for run in runs])
    logfile = (runs[0].run_dir / "run.log").open("w", encoding="utf-8")
    proc = subprocess.Popen([sys.executable, "-m", "memrank.runner", *argv],  # noqa: S603 - our own argv
                            stdout=logfile, stderr=subprocess.STDOUT, start_new_session=True)
    for run in runs:
        run.status.annotate(pid=proc.pid)
    ids = " ".join(run.run_dir.name for run in runs)
    style.say(f"track: memrank watch {ids}")
    style.say(f"logs:  memrank logs {runs[0].run_dir.name} -f")


def _mirror_to_mlflow(run_dir: Path) -> None:
    """Auto-mirror the just-recorded run into MLflow when the integration is enabled.

    No-op unless ``MEMRANK_MLFLOW_ENABLED=1`` (see :func:`memrank.config.mlflow_enabled`). The run
    is already durably in the registry before this runs, so a mirror failure never loses data -- a
    later ``scripts/internal/mlflow_import/mlflow_import.py`` still works. Raises loudly (never degrades)
    when MLflow is enabled but the ``mlflow`` extra is not installed.
    """
    from memrank import config
    if not config.mlflow_enabled():
        return
    from memrank.tracking import export
    imported = export.mirror_run(run_dir, tracking_uri=config.mlflow_tracking_uri(),
                                        experiment=config.mlflow_experiment())
    style.say(f"mirrored {imported} cell(s) -> MLflow ({config.mlflow_tracking_uri()})")

def _composite_rankable(cell: dict[str, Any]) -> bool:
    """Whether ``cell``'s composite is a self-contained quality score to display + rank.

    Delegated: the legacy default lives in one place now (``memrank.metrics.headline``), because
    three copies of it had drifted and the one in the API ranked legacy BEAM records the rest
    withheld.
    """
    return headline.rankable_composite(cell)


#: Moved to `metrics.headline` beside the rankability rule it renders; the alias keeps
#: `runner._composite_display` importable for its existing consumers.
_composite_display = headline.composite_display

def _discard_checkpoint(out_path: Path) -> None:
    """Remove the pre-judgment checkpoint once the artifact it protects exists."""
    try:
        out_path.with_name(checkpoint.CHECKPOINT_FILE).unlink(missing_ok=True)
    except OSError:
        pass          # a checkpoint we cannot delete is clutter, not a reason to fail a good run


def _run_and_persist(instance, bench, *, k, repeats, run_id_prefix, model,
                     token_budget, seed, judge, judge_runtime, out_path, units=None,
                     workers=1, judge_workers=DEFAULT_JUDGE_WORKERS, fail_fast=False,
                     make_adapter=None, verbose=False,
                     observer=None, gate=None, target=None):
    """Run one cell, write its JSON, and always close the adapter afterward."""
    try:
        # Guarded HERE, not in run_cell: run_cell is a library primitive that tests drive directly
        # with doubles that legitimately return nothing. The danger is a PERSISTED number -- a
        # result that gets reported, compared and published.
        result = run_cell(instance, bench, k=k, repeats=repeats,
                          run_id_prefix=run_id_prefix, model=model,
                          token_budget=token_budget, seed=seed, judge=judge,
                          judge_runtime=judge_runtime, units=units,
                          workers=workers, judge_workers=judge_workers, fail_fast=fail_fast,
                          checkpoint_path=out_path.with_name(checkpoint.CHECKPOINT_FILE),
                          make_adapter=make_adapter,
                          observer=observer if observer is not None else NULL_OBSERVER,
                          gate=gate)
        # The target REF, which nothing else records: the cell's ``adapter`` field is the adapter
        # name ("mem0"), so without this an artifact cannot say which variant produced it.
        if target is not None:
            result = replace(result, target=target)
        aggregated = result.to_dict()
        _assert_retrieved_something(instance, aggregated.get("per_query") or [])
        out_path.write_text(json.dumps(aggregated, indent=2, sort_keys=True),
                            encoding="utf-8")
        # The artifact exists, so the checkpoint has nothing left to rescue. Left in place would be
        # a second copy of every retrieved document beside the result, and a stale one the moment
        # anything re-judges.
        _discard_checkpoint(out_path)
        return aggregated
    finally:
        _close_adapter(instance)


def _summary_cell(c: dict[str, Any]) -> dict[str, Any]:
    """One summary entry; composite is nulled (not a rankable 0.0) when the
    composite is not a self-contained rankable score (e.g. BEAM, judge required),
    with the status carried.

    A ``not_applicable`` cell (graph benchmark x non-graph adapter) has no
    composite/metrics -- emit a status-only row, never index the absent keys.

    ``judged_metrics`` and ``config_hash`` ride along because this file is what a CLOUD run
    leaves behind for the platform to read: the full cell artifacts stay in S3 (tens to hundreds
    of megabytes), and the server projects a run's score from this summary. Without the judged
    block, every judged cloud run would reach the GUI as a withheld score -- which is exactly the
    bug this carries the fix for.
    """
    if c.get("status") == "not_applicable":
        return {"adapter": result_schema.adapter_of(c), "status": "not_applicable",
                "composite": None, "reason": c.get("reason")}
    rankable = _composite_rankable(c)
    head = headline.cell_headline(c)
    return {
        "adapter": result_schema.adapter_of(c),
        "composite": c["composite"] if rankable else None,
        "substring_recall_supported": c.get("substring_recall_supported", True),
        "composite_rankable": rankable,
        "quality_metric": c.get("quality_metric", "substring_recall"),
        "judged_metrics": c.get("judged_metrics"),
        "config_hash": (c.get("receipt") or {}).get("config_hash"),
        "recall_display": ("n/a (judge required)" if head.value is None
                           else f"{head.value:.3f}"),
        "latency_metrics": c["latency_metrics"],
        "token_metrics": c["token_metrics"],
    }


def _write_summary(output_dir, eval_ref, tier, slice, k, seed, cells, *,
                   announce: bool = True) -> None:
    """Write a summary of ``cells``; ``announce=False`` for the quiet per-run-dir copies.

    The FILE is named by the eval ref so two variants cannot overwrite one another; the `benchmark`
    field inside stays the registry name, because sync, the leaderboard and the API all read it and
    a ref there would be a second, silent schema change.
    """
    benchmark = eval_ref.split(":", 1)[0]
    summary = {
        "memrank_version": __version__,
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": benchmark, "eval_ref": eval_ref,
        "tier": tier, "slice": slice, "k": k, "seed": seed,
        "cells": [_summary_cell(c) for c in cells],
    }
    path = output_dir / f"summary__{_safe_label(eval_ref)}.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if announce:
        style.say(f"Summary written to {path}")


# ---------------------------------------------------------------------------- #
# Shared CLI-boundary validation
# ---------------------------------------------------------------------------- #
