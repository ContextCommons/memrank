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
"""How a run's parameters become the ``memrank`` arguments a submitted task runs.

One renderer, two callers on opposite sides of the wire. The CLI renders from the parameters
Typer parsed and POSTs the result; the API renders from a structured config a browser sent and
never asks the caller for flags at all. Both feed
:func:`memrank.placement.cloud_submit.remote_command`, which is why this lives here rather than
beside either caller -- the flags exist to be interpolated into a container command, and a second
copy of the table in a second language is the drift this arrangement exists to prevent.

Deliberately pure and stdlib-only. The API imports it at module scope, and it must not drag in
Typer's execution graph to render a list of strings -- ``tests/placement/test_remote_argv.py``
asserts that it does not.
"""
from __future__ import annotations

from typing import Any

# How each `run` parameter crosses to a submitted task. Built as a table, and enumerated by
# tests/placement/test_remote_argv.py, because the failure mode is silent: a forwarded flag that
# gets forgotten does not error, it runs the remote evaluation with a DIFFERENT configuration than
# the one asked for -- a mislabelled row, which is the defect this whole design exists to prevent.
REMOTE_VALUE_FLAGS: tuple[tuple[str, str], ...] = (
    ("k", "--k"), ("repeats", "--repeats"),
    ("model", "--model"), ("token_budget", "--token-budget"), ("seed", "--seed"),
    ("workers", "--workers"), ("judge_workers", "--judge-workers"),
    ("judge_samples", "--judge-samples"))
REMOTE_SWITCHES: tuple[tuple[str, str], ...] = (
    ("verbose", "--verbose"), ("no_judge_cache", "--no-judge-cache"),
    ("allow_empty_judge_coverage", "--allow-empty-judge-coverage"),
    ("fail_fast", "--fail-fast"))
#: Parameters whose value must be STATED rather than implied by a flag's absence, as
#: ``(name, when_true, when_false)``. A switch says nothing when it is off, which is the right
#: shape for a knob whose default is fixed -- and the wrong one for judging, whose default is
#: DERIVED from the eval. The executing side runs a different build, so re-deriving there could
#: answer differently and the same command would mean two things. The submitter decides once and
#: the argv carries the decision either way.
REMOTE_TRISTATE: tuple[tuple[str, str, str], ...] = (("judge", "--judge", "--no-judge"),)
REMOTE_REPEATED: tuple[tuple[str, str], ...] = (("unit", "--unit"),)
# Parameters that describe THIS invocation, not the evaluation: they are consumed by the submitter
# and must not be forwarded. `on` above all -- a task that inherited it would submit another task.
LOCAL_ONLY_PARAMS: frozenset[str] = frozenset({
    "target", "benchmark_arg", "overrides",          # rebuilt positionally
    "adapter", "benchmark", "all_adapters",          # legacy form, refused for cloud
    # Retired with no replacement to forward: the egress gate is derived from the benchmark's
    # own `is_synthetic` and the judge's ceiling is the eval's question count, so the task
    # reconstructs both from the eval ref it is already given. Declared solely to be refused.
    "ack_egress", "max_judge_calls",
    # tier/slice travel INSIDE the eval ref (`benchmark_arg` -- `beam:100k-smoke`), never as
    # flags: `--tier`/`--slice` are retired, so rendering them makes the executing process --
    # the background child, or a cloud task after its image was pulled and paid for -- refuse
    # its own command. A run died exactly this way on 2026-08-13 (stale `queued`, the child's
    # death note in run.log). The values still ride in `params` for the submitter's own
    # bookkeeping (run ids, row fields); they are simply never argv.
    "tier", "slice",
    "output_dir", "on", "run_id",
    "source", "target_digest",
    # --org is resolved by the SUBMITTER: the CLI exchanges its session token for the org's
    # credentials and injects them into its own process. Forwarding the flag would make the task
    # try to log in, which it cannot -- it has no browser and no stored token. When cloud runs
    # learn to carry an org, they will do it by SSM reference in the task definition, not by argv.
    "org"})


def remote_argv(params: dict[str, Any], *, ref: str | None = None) -> list[str]:
    """The ``memrank`` arguments the task should run, rebuilt from the parsed parameters.

    Rebuilt rather than filtered out of ``sys.argv``: argv is not reliably this command's own (a
    ``CliRunner`` invocation proved it -- the remote command came out as
    ``memrank tests/... -q``), and scraping it also silently drops any flag whose spelling the
    filter does not anticipate.
    """
    # `ref` overrides the parameter because a sweep is fanned out here, one task per target. Left
    # as the raw comma string, the task would sweep INTERNALLY -- N cells under one run id, silently
    # disagreeing with the local layout where each target gets its own.
    argv: list[str] = ["submit", ref or params["target"], params["benchmark_arg"],
                       *(params.get("overrides") or [])]
    for name, flag in REMOTE_VALUE_FLAGS:
        value = params.get(name)
        if value is not None:
            argv += [flag, str(value)]
    for name, flag in REMOTE_SWITCHES:
        if params.get(name):
            argv.append(flag)
    for name, when_true, when_false in REMOTE_TRISTATE:
        argv.append(when_true if params.get(name) else when_false)
    for name, flag in REMOTE_REPEATED:
        for value in params.get(name) or []:
            argv += [flag, str(value)]
    return argv


def submit_command(params: dict[str, Any], *, refs: list[str], on: str,
                   org: str | None = None) -> list[str]:
    """The ``memrank submit`` command line that reproduces this submission, for printing back.

    Derived from the same table the submission itself renders from, and that is the whole point. The
    one caller used to rebuild this string from the three fields someone remembered to pass it --
    benchmark, slice, tier -- so a sweep submitted with ``--judge --ack-egress`` was offered back
    without them, and pasting the suggestion silently produced UNJUDGED runs. Nothing failed; the
    numbers just answered a different question. Every flag added since inherited that silence.

    Rendering from :func:`remote_argv` inverts it: a flag added to ``REMOTE_VALUE_FLAGS`` or
    ``REMOTE_SWITCHES`` appears here the same day, because there is nothing to remember to update.

    Every value is stated, defaults included, which makes for a long line. That is the deliberate
    half of the trade. ``params`` cannot say whether a value was typed or defaulted -- Typer has
    already filled the defaults in -- so trimming would mean holding a second table of what the
    defaults ARE, which is the duplication this module exists to avoid, and the suggestion would
    then silently change meaning the day a default does.

    Args:
        params: The submission's parameters, as the CLI parsed them.
        refs: The targets this command should submit -- the refused ones, not the original sweep.
        on: The placement to name explicitly; it is a LOCAL_ONLY param and never in the argv.
        org: The org to name, when there is one. Also local-only, for the same reason.

    Returns:
        The command as tokens, ready to join with spaces.
    """
    return ["memrank", *remote_argv(params, ref=",".join(refs)), "--on", on,
            *(["--org", org] if org else [])]
