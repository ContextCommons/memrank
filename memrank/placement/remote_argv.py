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

#: The internal command a task runs (ATO-2366). Not `submit`: nothing a task runs may place
#: another run, so the verb that executes pre-minted run ids is the only one rendered.
TASK_VERB = "task-exec"

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
# Parameters of `task-exec` that describe THIS invocation, not the evaluation: they are set by
# whoever renders the full command and are never forwarded from the parameters. `on` above all
# -- the cloud renderer appends `--on none` itself, so a task cannot inherit a placement.
#
# tier/slice are not here and never argv: they travel INSIDE the eval ref (`benchmark_arg` --
# `beam:100k-smoke`). Rendering them as flags made the executing process refuse its own command
# (2026-08-13: a stale `queued` run, the child's death note in run.log). The values still ride in
# `params` for the submitter's bookkeeping (run ids, row fields).
LOCAL_ONLY_PARAMS: frozenset[str] = frozenset({
    "target", "benchmark_arg", "overrides",          # rebuilt positionally
    "output_dir", "on", "run_id", "target_digest",
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
    argv: list[str] = [TASK_VERB, ref or params["target"], params["benchmark_arg"],
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
