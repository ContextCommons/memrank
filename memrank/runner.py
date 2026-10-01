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
"""Memrank CLI + orchestrator.

Entry point:
    memrank submit mem0 locomo:smoke --seed 42
    memrank submit atomicmemory beam:100k-smoke
    memrank submit word-overlap demo --on cloud --org atomicstrata
    memrank evals ls
    memrank runs ls

The runner is deliberately small. The bulk of the work lives in the
adapter/benchmark contracts; the runner just glues the lifecycle together,
captures latency/token metrics, and writes a reproducibility receipt.

Operator and cross-record-analysis commands (``leaderboard*``, ``arena*``, ``compare``,
``report``, ``compare-versions``, ``mlflow-sync``) used to live here. They now live in
:mod:`memrank.ops`, reached through ``scripts/internal/memrank-ops.py`` -- ``memrank`` shows exactly
one product.
"""

from __future__ import annotations

import time  # noqa: F401 - `runner.time` is a documented patch surface (tests/orchestration/test_ingest_retry.py)
from pathlib import Path
from typing import Any

import typer

from memrank import (
    errors,
    settings,
)
from memrank.benchmarks import (
    get_benchmark,  # noqa: F401 - an established `runner.get_benchmark` import surface
    judge_required,
)
from memrank.benchmarks.refs import parse_eval_ref
from memrank.cli import deprecated as deprecated_cli
from memrank.cli.agents import agents_app
from memrank.cli.agents import run as _run_cmd
from memrank.cli.agents import serve as _serve_cmd
from memrank.cli.auth import app as auth_app
from memrank.cli.config import config_app
from memrank.cli.evals import evals_app
from memrank.cli.monitor import logs as _logs_cmd
from memrank.cli.retired import refuse_retired_flags, register_retired

# Flat aliases for the hot path -- exactly `ps`, `watch`, `logs` and `kill`, and the SAME
# functions their `runs` forms are, not reimplementations. `ps` used to read
# local status.json files only, so it and `runs ls` disagreed about what was running: a cloud
# sweep submitted from another directory was invisible to one and 89% complete according to the
# other.
from memrank.cli.runs import cli_ps as _ps_cmd
from memrank.cli.runs import runs_app
from memrank.cli.secrets import secrets_app
from memrank.cli.skills import skills_app
from memrank.cli.targets import targets_app
from memrank.cli.watch import kill as _kill_cmd
from memrank.cli.watch import watch as _watch_cmd
from memrank.errors import ActionRequired, Concluded, MemrankError

# The eval loop lives in `memrank.evaluation` now; every name it took along is re-imported
# here BY ASSIGNMENT so `memrank.runner.<name>` stays importable and -- crucially -- stays
# the SAME object: tests monkeypatch attributes on this module, and code here must keep
# resolving to what they patched. Trim once in-repo consumers import the real homes.
from memrank.evaluation.aggregate import (  # noqa: F401
    _aggregate_cell,
    _benchmark_params,
    _corpus_size,
    _drill_unit,
    _ingest_throughput,
    _mean_composite,
)
from memrank.evaluation.cell import (  # noqa: F401
    _INGEST_BACKOFF_BASE_S,
    _INGEST_RETRY_DEADLINE_S,
    AdapterFactory,
    EmptyRetrieval,
    RateLimitExhausted,
    _assert_retrieved_something,
    _close_adapter,
    _effective_budget_mode,
    _ingest_one,
    _ingest_with_progress,
    _is_rate_limited,
    _not_applicable_cell,
    _pause_for_rate_limit,
    _retrieve_one,
    _retrieve_with_progress,
    _run_one_unit,
    _run_unit_repeats,
    _run_units_concurrent,
    _run_units_sequential,
    cell_applicable,
    run_cell,
)
from memrank.evaluation.constants import DEFAULT_JUDGE_WORKERS
from memrank.evaluation.judge_stage import (  # noqa: F401
    EmptyJudgeCoverage,
    _apply_judge,
    _context_text,
    _grade_all,
    _judge_cfg_from_receipt,
    _judge_metrics,
    _judge_one_query,
    _judge_receipt_config,
    _truncation_metrics,
    assert_judge_coverage,
    judge_cost_estimate,
)
from memrank.evaluation.observer import (  # noqa: F401
    NULL_OBSERVER,
    EvalObserver,
    EvalPlan,
    _eval_plan,
    _progress_step,
)
from memrank.evaluation.receipt import _build_receipt, _evidence_assessment  # noqa: F401
from memrank.judging.judge import (
    UnparseableVerdict,  # noqa: F401 - `runner.UnparseableVerdict` is a patch surface
)
from memrank.judging.shape import (  # noqa: F401 - `runner.BinaryJudgeShape` etc. are patch surfaces
    GENERIC_BINARY_CATEGORIES,
    BinaryJudgeShape,
    JudgeShape,
)

# The run lifecycle lives in `memrank.orchestration` now, re-imported BY ASSIGNMENT for
# the same reason as the `memrank.evaluation` block above: `submit` calls these as bare
# names (so a test that patches `runner.<name>` still intercepts the command's path), and
# in-repo consumers still import them from here. Trim as consumers move to the real homes.
from memrank.orchestration.cloud import (  # noqa: F401
    _api_client,
    _record_submission,
    _report_refusals,
    _submit_sweep,
)
from memrank.orchestration.observers import (
    _planned_progress,  # noqa: F401 - re-export; tests size plans through `runner._planned_progress`
)
from memrank.orchestration.placement_gate import (  # noqa: F401
    _CLOUD_SIZE_IN_PROCESS,
    _CLOUD_SIZE_STACK,
    PLACEMENTS,
    _environment_patch,
    _require_placement_ready,
    _source_digest_guard,
    _validate_source_target,
    placement_for,
)
from memrank.orchestration.resolve import (  # noqa: F401
    _JUDGE_ONLY_TARGETS,
    _cli_experiments,
    _ensure_run_credentials,
    _experiment_metadata,
    _judge_cfg_or_refuse,
    _load_org_credentials,
    _manifest_factories,
    _manifest_factory,
    _refuse_abstract,
    _run_targets,
    _split_overrides,
    _validate_positive,
    _validate_reader,
    _warn_if_meaningless_unjudged,
    run_credentials,
)
from memrank.orchestration.sweep import (  # noqa: F401
    SweepKilled,
    _child_argv,
    _composite_display,
    _composite_rankable,
    _discard_checkpoint,
    _echo_cell_outcome,
    _echo_per_query_outcomes,
    _ExecOpts,
    _execute_local_run,
    _filter_units,
    _install_kill_handler,
    _LocalRun,
    _mark_unstarted,
    _mint_local_runs,
    _mirror_to_mlflow,
    _preflight_engines,
    _preload_tokenizer_with_notice,
    _record_local_run,
    _run_and_persist,
    _run_local_sweep,
    _safe_label,
    _spawn_child,
    _submit_local_sweep,
    _summary_cell,
    _sweep_gates,
    _write_summary,
    question_gates,
    run_identity,
)
from memrank.provenance.install import version_line
from memrank.targets.resolve import RefError
from memrank.term import outcome, style


def _exit_with(exc: BaseException) -> SystemExit:
    """Turn an exception that reached the boundary into terminal output and an exit code.

    ``MemrankError`` is a promise that ``str(exc)`` is a complete, actionable sentence, so it
    is printed verbatim. Anything else is a bug in memrank rather than a mistake by the user;
    name it as such so it gets reported, and point at the stack without printing it --
    ``submit`` decrypts org credentials into a frame, and locals would carry them out.
    """
    if isinstance(exc, Concluded):
        outcome.render(exc.outcome)
        return SystemExit(exc.exit_code)
    if isinstance(exc, ActionRequired):
        style.step(exc.statement)
        for step in exc.steps:
            style.step(step.text, step.commands)
        return SystemExit(1)
    if isinstance(exc, MemrankError):
        style.error(str(exc))
        return SystemExit(1)
    if isinstance(exc, KeyboardInterrupt):
        style.error("interrupted")
        return SystemExit(130)
    style.error(f"internal error: {type(exc).__name__}: {exc}")
    style.note("this is a bug in memrank; re-run with MEMRANK_DEBUG=1 for a traceback")
    return SystemExit(1)


class _BoundedTyper(typer.Typer):
    """A Typer whose ``__call__`` is the CLI's error boundary.

    The boundary lives here, not in ``main``, because the console script is generated at
    install time: an editable install picks up source changes immediately but keeps whatever
    ``sys.exit(app())`` line it was born with. Hanging the boundary off the entry point name
    meant every already-installed CLI kept bypassing it until someone reinstalled -- which is
    how a traceback survived the commit that was supposed to remove it. Anything that calls
    ``app()`` is bounded now, whatever the entry point happens to be named.
    """

    def __call__(self, *args: object, **kwargs: object) -> object:
        try:
            return super().__call__(*args, **kwargs)
        except MemrankError as exc:
            raise _exit_with(exc) from exc
        except KeyboardInterrupt as exc:
            raise _exit_with(exc) from None
        except Exception as exc:  # noqa: BLE001 - the boundary's whole job is the unexpected
            if errors.wants_traceback():
                raise
            raise _exit_with(exc) from exc


#: Named once because the root callback below would otherwise silently replace it with its
#: own docstring -- Typer prefers the callback's help over the app's.
_APP_HELP = (
    "Evaluate your agent on your tasks. You supply the system -- an agent: `memrank run locomo --agent "
    "full-context` -- and memrank puts the evaluation to it and writes down what produced "
    "every value."
)


app = _BoundedTyper(
    name="memrank",
    help=_APP_HELP,
    add_completion=False,
    no_args_is_help=True,
    # Typer installs its own sys.excepthook that renders a Rich traceback -- with local
    # variables -- for anything uncaught. Off, so the boundary above is the only thing that
    # decides how a failure reaches the terminal.
    pretty_exceptions_enable=False,
)
app.command("run")(_run_cmd)
app.command("serve")(_serve_cmd)
app.add_typer(agents_app, name="agents")
app.add_typer(auth_app, name="auth")
app.add_typer(config_app, name="config")
app.add_typer(evals_app, name="evals")
app.add_typer(runs_app, name="runs")
app.add_typer(secrets_app, name="secrets")
app.add_typer(skills_app, name="skills")
# The engine-hosting path -- `submit`, its catalog of targets, and the commands that follow a
# background run it started -- is hidden rather than removed (ATO-2343): a new user meets only
# `memrank run --agent`, while the code, its tests and anyone's muscle memory keep working.
# Each one is deprecated and says so on stderr when invoked.
app.add_typer(targets_app, name="targets", hidden=True,
              callback=lambda: deprecated_cli.notice("targets"))
app.command("ps", hidden=True)(deprecated_cli.deprecated("ps", _ps_cmd))
app.command("logs", hidden=True)(deprecated_cli.deprecated("logs", _logs_cmd))
app.command("watch", hidden=True)(deprecated_cli.deprecated("watch", _watch_cmd))
app.command("kill", hidden=True)(deprecated_cli.deprecated("kill", _kill_cmd))
register_retired(app)


def _print_version(requested: bool) -> None:
    """Answer ``--version`` before Typer looks for a subcommand, then stop.

    Eager, because ``memrank --version`` names no command and would otherwise be met with the
    help text ``no_args_is_help`` exists to show.
    """
    if requested:
        style.out(version_line())
        raise typer.Exit()


@app.callback(help=_APP_HELP)
def _root(
    version: bool = typer.Option(  # noqa: ARG001 - read by its eager callback, not the body
        False, "--version", "-V", callback=_print_version, is_eager=True,
        help="Show the version and where this build came from, then exit."),
) -> None:
    """The root of the CLI, carrying only what applies to every command.

    ``-V`` rather than ``-v``: ``-v`` is verbose by convention everywhere else, and spending
    it on version would put the two in permanent conflict the first time a global verbosity
    flag is wanted.
    """
    # Library modules narrate through `logging` (dataset downloads, survivable adapter
    # warnings); the terminal is the CLI's concern, so the CLI is what gives them a voice.
    style.install_log_bridge()


# ---------------------------------------------------------------------------- #
# Run
# ---------------------------------------------------------------------------- #


@app.command("submit", hidden=True)  # the engine-hosting path; see the registrations above
def submit(
    target: str | None = typer.Argument(None, help="Target ref, e.g. hindsight:matched (comma-separated to sweep)"),
    benchmark_arg: str | None = typer.Argument(None, metavar="EVAL",
                                           help="Eval ref, e.g. beam:100k-smoke (see `memrank evals ls`)"),
    overrides: list[str] = typer.Argument(None, metavar="[KEY=VALUE]...", help="Component overrides, e.g. embedder=voyage/voyage-4-large"),
    adapter: str | None = typer.Option(None, hidden=True, help="retired: see `--help`'s TARGET"),
    benchmark: str | None = typer.Option(None, hidden=True, help="retired: see `--help`'s EVAL"),
    tier: str | None = typer.Option(None, hidden=True),
    slice: str | None = typer.Option(None, hidden=True),
    ack_egress: bool = typer.Option(False, "--ack-egress", hidden=True,
                                    help="retired: the benchmark gates egress, not this flag"),
    max_judge_calls: int | None = typer.Option(None, "--max-judge-calls", hidden=True,
                                               help="retired: the eval ref bounds the judge"),
    k: int = typer.Option(10, help="Top-k for retrieval"),
    repeats: int = typer.Option(3, help="Retrieval repeats for latency stability"),
    model: str = typer.Option("gpt-4o-mini", help="Model name for cost pricing"),
    token_budget: int = typer.Option(
        5000, help="Retrieval token budget: caps BOTH the $/query estimate and the "
                   "context actually sent to the reader. The fairness control -- every "
                   "target is held to the same budget, except on a benchmark whose own "
                   "protocol declares the reader uncapped (beam)."),
    seed: int = typer.Option(42, help="Random seed"),
    workers: int = typer.Option(1, help="Ingest+retrieve this many UNITS concurrently "
                                        "(each on its own adapter instance; recall is unaffected, "
                                        "latency becomes contended). 1 = sequential."),
    fail_fast: bool = typer.Option(
        False, "--fail-fast", help="Stop the cell at the first unit that fails. The default "
               "attempts every unit, records what happened to each in the artifact "
               "(unit_outcomes, units_failed) and scores the composite over the units that ran."),
    unit: list[str] = typer.Option([], "--unit", help="Run only unit(s) matching this "
                                   "unit_id / substring, or a 0-based index (repeatable). "
                                   "Applied after the eval ref's slice."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Emit every doc/query in the "
                                 "progress bars and print a per-query HIT/MISS table per cell."),
    all_adapters: bool = typer.Option(False, "--all-adapters", hidden=True,
                                      help="retired: sweep with a comma-separated TARGET"),
    output_dir: Path = typer.Option(Path("results"), help="Where to write per-cell JSON"),
    source: Path | None = typer.Option(
        None, "--source", hidden=True, help="retired: declare binding.root in a named target"),
    judge: bool | None = typer.Option(
        None, "--judge/--no-judge",
        help="LLM-judged sufficiency + answer correctness (sends data to Anthropic). Defaults to "
             "what the eval needs: on for locomo/longmemeval/beam, whose only quality metric is "
             "the judge's, off where the benchmark scores itself. `--no-judge` measures latency "
             "and cost without paying for quality; `--judge` adds it where it is optional."),
    judge_samples: int = typer.Option(1, help="Majority-vote samples per judge grade"),
    judge_workers: int = typer.Option(
        DEFAULT_JUDGE_WORKERS, help="Grade this many queries concurrently. Separate from --workers because they bound "
                "different resources: --workers is engine containers, this is ONE provider "
                "account. Quality is unaffected -- the calls are independent and the reduce is "
                "ordered, so metrics are identical at any value."),
    no_judge_cache: bool = typer.Option(False, "--no-judge-cache", help="Disable the judge response cache"),
    allow_empty_judge_coverage: bool = typer.Option(False, "--allow-empty-judge-coverage", help="Permit a judged run that scores 0 queries (no supported categories)"),
    on: str | None = typer.Option(None, "--on", help="Where to run: none (an already-running "
                           "engine) | local (a disposable image or named source target) | cloud "
                           "(submit an ECS task). Every placement returns immediately; block "
                           "with `memrank watch <id>`. Default: `defaults.on` (`memrank config ls`)"),
    org: str | None = typer.Option(None, "--org", help="Spend this org's stored credentials "
                                   "instead of the local wallet (requires `memrank auth login`)"),
    run_id: list[str] = typer.Option(None, "--run-id", hidden=True, help="internal: execute mode -- reuse pre-created run id(s), one per target in order (set by the submitting parent and by the cloud task's rendered command)"),
    target_digest: str | None = typer.Option(None, "--target-digest", hidden=True,
                                             help="internal: source target definition guard"),
) -> None:
    """Submit a (target, benchmark) cell -- or a comma sweep -- and print one id per target."""
    deprecated_cli.notice("submit")
    refuse_retired_flags({"--adapter": adapter, "--benchmark": benchmark,
                          "--all-adapters": all_adapters, "--tier": tier, "--slice": slice,
                          "--ack-egress": ack_egress, "--max-judge-calls": max_judge_calls})
    if source is not None:
        raise typer.BadParameter(
            "--source is retired; put the checkout in a named target's binding.root, then "
            "submit that target by name")
    explicit_on = on
    configured_on = settings.get("defaults.on")
    on = explicit_on or configured_on or "none"
    component_overrides, run_overrides = _split_overrides(overrides or [])
    benchmark, factories = _run_targets(
        target=target, benchmark_arg=benchmark_arg, overrides=component_overrides,
        # The cross-check asks a LIVE engine what it is configured with, so it can only be made
        # where one is already answering. That is `--on none` alone: `--on local` provisions the
        # engine after this point and `--on cloud` runs it in the task. In both of those the check
        # still happens -- at the real construction, once the engine is up -- which is why narrowing
        # it here loses nothing. Left unnarrowed, `--on local` failed in preflight with a connection
        # refused against a container it was about to start.
        verify_engine=on == "none")
    on, source_targets = _validate_source_target(
        explicit_on=explicit_on, on=on, factories=factories, overrides=component_overrides)
    if source_targets:
        target_digest = _source_digest_guard(source_targets, target_digest)
    if not source_targets and explicit_on is None and configured_on is None:
        raise typer.BadParameter("no placement selected; pass --on none, local, or cloud")
    submitting = on == "cloud"
    # `defaults.org` answers "on whose account does the PLATFORM run this" and nothing else. A
    # local run's --org pulls the org's decrypted credentials onto this machine, so letting a
    # configured default reach it would mean signing in silently changed what local runs spend.
    if submitting and not org:
        org = settings.get("defaults.org")
    # A submission never needs decrypted values here; a local run explicitly using an org does.
    if org and not submitting:
        _load_org_credentials(org)
    _validate_positive(k=k, repeats=repeats, token_budget=token_budget,
                       judge_samples=judge_samples, workers=workers)
    # Parsed BEFORE the submission fork, not only on the local path. The eval ref IS the
    # evaluation's identity: `beam:100k-smoke` is a different evaluation from `beam:1m`, not the
    # same one with a knob turned. Deriving here means `tier`/`slice` everywhere below come from
    # the ref rather than beside it -- and it is what the CLOUD payload needs: the server mints
    # run ids and rows from the positional facts (benchmark, tier, slice), and shipping the raw
    # ref as `benchmark` put a `:` into a string that becomes a shell token and an S3 key
    # (refused, correctly, as an unsafe run id on 2026-08-13). `benchmark_arg` becomes the
    # canonical spelling so the rendered argv records what actually runs.
    try:
        eval_name, eval_kwargs, eval_ref = parse_eval_ref(benchmark)
    except RefError as exc:
        raise typer.BadParameter(str(exc)) from exc
    bench_kwargs: dict[str, Any] = {**eval_kwargs, "k": k}
    tier, slice = eval_kwargs.get("tier"), eval_kwargs.get("slice")
    benchmark_arg = eval_ref
    # An omitted flag is not a decision to skip judging: on locomo, longmemeval and beam the judge
    # IS the metric, and a run without one measures latency and nothing else. Resolved here rather
    # than defaulted at the flag, because the answer depends on which eval was named -- and resolved
    # BEFORE `_validate_reader`, which refuses a reader override on an unjudged run and would
    # otherwise refuse one on a run that is about to judge.
    judge = judge_required(eval_name) if judge is None else judge
    _validate_reader(run_overrides.get("reader"), judged=judge)
    judge_cfg = _judge_cfg_or_refuse(
        enabled=judge, samples=judge_samples, no_cache=no_judge_cache,
        allow_empty_coverage=allow_empty_judge_coverage,
        reader=run_overrides.get("reader"))
    # The coverage refusal and the cost estimate happen for EVERY placement, here, before anything
    # is minted or submitted. They are sums over the benchmark's own queries -- equally true on this
    # laptop and inside a Fargate task -- and asking them only in the task is what killed four runs
    # on 2026-08-13, each after an image pull and a 274 MB dataset download.
    # `targets` is placement-dependent and cannot be defaulted: a local sweep runs every engine
    # under ONE shared counter, while each submitted target is its own task with its own. It only
    # changes what the printed estimate says now, but it says it truthfully.
    gated_units, judge_cfg = question_gates(
        benchmark=eval_name, bench_kwargs=bench_kwargs, slice_=slice, judge_cfg=judge_cfg,
        unit=unit, targets=1 if submitting else len(factories))
    # `eval_name`, not the ref as it was typed: the variant is already in `tier`/`slice` below, and
    # a cell's identity must not depend on which of `beam` and `beam:100k` the caller wrote.
    experiments_by_ref = _cli_experiments(
        factories, eval_name, component_overrides,
        {**locals(), "reader": run_overrides.get("reader"), "slice_": slice})
    plan_hash = None
    if submitting:
        # Everything below is LOCAL execution work a submission must not do -- prompting for
        # credentials the task gets from SSM, provisioning engines, building the judge.
        params = dict(locals())
        _submit_sweep([label for label, _ in factories], params, org=org,
                      benchmark=eval_name, slice_=slice, tier=tier)
        return
    # Before a run exists, not inside one. A machine that cannot run here used to be discovered by
    # the placement AFTER minting, so a missing Docker became a run record marked failed with
    # `internal error: FileNotFoundError: 'docker'` -- memrank blaming itself for the machine. The
    # rows come from the placement that needs them, so this cannot drift from what actually stops
    # the run.
    # Every source target in the sweep, not just the first: each launches its own directory, so
    # each has its own markers and launcher to be missing. Naming them one run at a time would be
    # the round-trip a single honest refusal removes.
    for checked in source_targets or [None]:
        _require_placement_ready(on, target=checked)
    # In the PARENT, before anything spawns: the background child has no terminal, so a prompt
    # there would hang forever with nobody to answer it. The secrets land in this process's
    # environment, which the child inherits.
    _ensure_run_credentials(targets=[label for label, _ in factories], judged=judge,
                            # Only --on local has memrank starting the engine and therefore
                            # needing its keys; otherwise the engine already has them.
                            provisioning=(on == "local"))
    output_dir.mkdir(parents=True, exist_ok=True)
    opts = _ExecOpts(benchmark=eval_name, eval_ref=eval_ref,
                     bench_kwargs=bench_kwargs, tier=tier, slice_=slice,
                     on=on, output_dir=output_dir, k=k, repeats=repeats, model=model,
                     token_budget=token_budget, seed=seed, workers=workers, verbose=verbose,
                     judged=judge, judge_workers=judge_workers, fail_fast=fail_fast,
                     judge_cfg=judge_cfg,
                     units=gated_units, experiments=experiments_by_ref)
    if run_id:  # execute mode: the spawned child, or the cloud container -- the blocking path
        _run_local_sweep(factories, opts, run_ids=list(run_id))
        return
    _submit_local_sweep(factories, opts, params=dict(locals()))


@app.command("version")
def cli_version() -> None:
    """Print the Memrank version and where this build came from."""
    style.out(version_line())


def main() -> None:
    """Entry point for ``python -m memrank.runner`` and for ``[project.scripts]``.

    Carries no error handling of its own: the boundary is on ``app.__call__`` so that a
    console script generated before this existed -- one that still calls ``app()`` directly --
    is bounded too. Which name the entry point resolves to no longer changes behavior.
    """
    app()


if __name__ == "__main__":  # pragma: no cover - direct invocation guard
    main()
