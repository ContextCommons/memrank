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
    memrank run locomo --agent full-context
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
from memrank.cli.endpoints import endpoints_app
from memrank.cli.evals import evals_app
from memrank.cli.monitor import logs as _logs_cmd
from memrank.cli.retired import register_retired

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
# the same reason as the `memrank.evaluation` block above: `task-exec` calls these as bare
# names (so a test that patches `runner.<name>` still intercepts the command's path), and
# in-repo consumers still import them from here. Trim as consumers move to the real homes.
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
    ``task-exec`` decrypts org credentials into a frame, and locals would carry them out.
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
app.add_typer(endpoints_app, name="endpoints")
app.add_typer(evals_app, name="evals")
app.add_typer(runs_app, name="runs")
app.add_typer(secrets_app, name="secrets")
app.add_typer(skills_app, name="skills")
# The engine-hosting path's catalog of targets, and the commands that follow a hosted run, are
# hidden rather than removed (ATO-2343): a new user meets only `memrank run --agent`. `submit`
# itself is removed (ATO-2366); `task-exec` below only executes runs its caller already minted.
# Each hidden command is deprecated and says so on stderr when invoked.
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
        False, "--version", "-v", "-V", callback=_print_version, is_eager=True,
        help="Show the version and where this build came from, then exit."),
) -> None:
    """The root of the CLI, carrying only what applies to every command.

    ``-v`` because it is the version flag people try first, and a failing ``memrank -v`` reads
    as a broken install (ATO-2696); ``-V`` stays for anyone already typing it. A root option
    is only parsed before the command, so a subcommand's own ``-v`` (``--verbose``) is untouched.
    """
    # Library modules narrate through `logging` (dataset downloads, survivable adapter
    # warnings); the terminal is the CLI's concern, so the CLI is what gives them a voice.
    style.install_log_bridge()


# ---------------------------------------------------------------------------- #
# Run
# ---------------------------------------------------------------------------- #


#: The placements a task may execute under. ``cloud`` is absent on purpose: a task placing
#: another task is how a hosted run could start without the API's run-shape ceilings (ATO-2366).
_TASK_PLACEMENTS = ("none", "local")


@app.command("task-exec", hidden=True)
def task_exec(
    target: str = typer.Argument(..., help="Target ref, e.g. hindsight:matched (comma-separated to sweep)"),
    benchmark_arg: str = typer.Argument(..., metavar="EVAL",
                                        help="Eval ref, e.g. beam:100k-smoke"),
    overrides: list[str] = typer.Argument(None, metavar="[KEY=VALUE]...", help="Component overrides, e.g. embedder=voyage/voyage-4-large"),
    k: int = typer.Option(10, help="Top-k for retrieval"),
    repeats: int = typer.Option(3, help="Retrieval repeats for latency stability"),
    model: str = typer.Option("gpt-4o-mini", help="Model name for cost pricing"),
    token_budget: int = typer.Option(5000, help="Retrieval token budget"),
    seed: int = typer.Option(42, help="Random seed"),
    workers: int = typer.Option(1, help="Ingest+retrieve this many units concurrently"),
    fail_fast: bool = typer.Option(False, "--fail-fast", help="Stop the cell at the first failed unit"),
    unit: list[str] = typer.Option([], "--unit", help="Run only matching unit(s) (repeatable)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Per-query progress output"),
    output_dir: Path = typer.Option(Path("results"), help="Where to write per-cell JSON"),
    judge: bool | None = typer.Option(None, "--judge/--no-judge", help="Judge the answers"),
    judge_samples: int = typer.Option(1, help="Majority-vote samples per judge grade"),
    judge_workers: int = typer.Option(DEFAULT_JUDGE_WORKERS, help="Concurrent judge calls"),
    no_judge_cache: bool = typer.Option(False, "--no-judge-cache", help="Disable the judge response cache"),
    allow_empty_judge_coverage: bool = typer.Option(False, "--allow-empty-judge-coverage", help="Permit a judged run that scores 0 queries"),
    on: str = typer.Option("none", "--on", hidden=True,
                           help="internal: none (engines beside the task) or local (the local "
                                "background child provisions them); cloud is refused"),
    org: str | None = typer.Option(None, "--org", hidden=True,
                                   help="internal: the org whose credentials the parent loaded"),
    run_id: list[str] = typer.Option(None, "--run-id", help="Pre-created run id(s), one per target in order"),
    target_digest: str | None = typer.Option(None, "--target-digest", hidden=True,
                                             help="internal: source target definition guard"),
) -> None:
    """Internal: execute pre-created run(s) in this process. Not a user command.

    The command a cloud task runs (rendered by the API from a browser's structured config) and
    the one the local background child runs. It never places a run anywhere: it only executes
    run ids its caller already minted, which is why ``--run-id`` is required and ``--on cloud``
    is refused.
    """
    if not run_id:
        raise typer.BadParameter("task-exec executes pre-created runs; pass --run-id per target. "
                                 "To evaluate an agent, use `memrank run`.")
    if on not in _TASK_PLACEMENTS:
        raise typer.BadParameter(
            f"--on {on!r} is not a placement task-exec runs under; it accepts "
            f"{' or '.join(_TASK_PLACEMENTS)}")
    if org:
        _load_org_credentials(org)
    component_overrides, run_overrides = _split_overrides(overrides or [])
    benchmark, factories = _run_targets(
        target=target, benchmark_arg=benchmark_arg, overrides=component_overrides,
        # A live engine can only be asked what it is configured with where one already answers;
        # under `local` the placement starts it later and the check happens at construction.
        verify_engine=on == "none")
    on, source_targets = _validate_source_target(
        explicit_on=on, on=on, factories=factories, overrides=component_overrides)
    if source_targets:
        target_digest = _source_digest_guard(source_targets, target_digest)
    _validate_positive(k=k, repeats=repeats, token_budget=token_budget,
                       judge_samples=judge_samples, workers=workers)
    try:
        eval_name, eval_kwargs, eval_ref = parse_eval_ref(benchmark)
    except RefError as exc:
        raise typer.BadParameter(str(exc)) from exc
    bench_kwargs: dict[str, Any] = {**eval_kwargs, "k": k}
    tier, slice = eval_kwargs.get("tier"), eval_kwargs.get("slice")
    # The caller states --judge/--no-judge explicitly; deriving it here is only a backstop.
    judge = judge_required(eval_name) if judge is None else judge
    _validate_reader(run_overrides.get("reader"), judged=judge)
    judge_cfg = _judge_cfg_or_refuse(
        enabled=judge, samples=judge_samples, no_cache=no_judge_cache,
        allow_empty_coverage=allow_empty_judge_coverage,
        reader=run_overrides.get("reader"))
    gated_units, judge_cfg = question_gates(
        benchmark=eval_name, bench_kwargs=bench_kwargs, slice_=slice, judge_cfg=judge_cfg,
        unit=unit, targets=len(factories))
    experiments_by_ref = _cli_experiments(
        factories, eval_name, component_overrides,
        {**locals(), "reader": run_overrides.get("reader"), "slice_": slice})
    for checked in source_targets or [None]:
        _require_placement_ready(on, target=checked)
    _ensure_run_credentials(targets=[label for label, _ in factories], judged=judge,
                            provisioning=(on == "local"))
    output_dir.mkdir(parents=True, exist_ok=True)
    opts = _ExecOpts(benchmark=eval_name, eval_ref=eval_ref,
                     bench_kwargs=bench_kwargs, tier=tier, slice_=slice,
                     on=on, output_dir=output_dir, k=k, repeats=repeats, model=model,
                     token_budget=token_budget, seed=seed, workers=workers, verbose=verbose,
                     judged=judge, judge_workers=judge_workers, fail_fast=fail_fast,
                     judge_cfg=judge_cfg,
                     units=gated_units, experiments=experiments_by_ref)
    _run_local_sweep(factories, opts, run_ids=list(run_id))


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
