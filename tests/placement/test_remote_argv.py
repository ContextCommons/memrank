"""Every `run` parameter must be deliberately forwarded to a cloud task, or deliberately not.

This is the enumeration gate `CLAUDE.md` calls for. The failure it guards is silent: a new flag that
nobody remembers to forward does not raise -- the remote evaluation just runs with a different
configuration than the one asked for, and the row is labelled as though it did not. That is exactly
the mislabel the whole placement design exists to prevent, one level down.

Adding a parameter to `memrank run` fails this test until it is classified.
"""
from __future__ import annotations

import inspect
import subprocess
import sys

import pytest

from memrank import runner
from memrank.placement.remote_argv import (
    LOCAL_ONLY_PARAMS,
    REMOTE_REPEATED,
    REMOTE_SWITCHES,
    REMOTE_TRISTATE,
    REMOTE_VALUE_FLAGS,
    remote_argv,
)

_CLASSIFIED = (
    {name for name, _ in REMOTE_VALUE_FLAGS}
    | {name for name, _ in REMOTE_SWITCHES}
    | {name for name, _, _ in REMOTE_TRISTATE}
    | {name for name, _ in REMOTE_REPEATED}
    | set(LOCAL_ONLY_PARAMS)
)

BASE = {"target": "word-overlap", "benchmark_arg": "demo", "overrides": []}


def test_every_run_parameter_is_classified():
    """Fails when a parameter is added to `run` without deciding whether a cloud task needs it."""
    params = set(inspect.signature(runner.submit).parameters)
    unclassified = params - _CLASSIFIED
    assert not unclassified, (
        f"{sorted(unclassified)} not classified for --on cloud. Add each to REMOTE_VALUE_FLAGS / "
        f"REMOTE_SWITCHES / REMOTE_REPEATED (the task needs it) or LOCAL_ONLY_PARAMS (it "
        f"describes the submitting process). Forwarding nothing silently changes what runs.")


def test_no_parameter_is_classified_twice():
    forwarded = [name for name, _ in (*REMOTE_VALUE_FLAGS, *REMOTE_SWITCHES, *REMOTE_REPEATED)]
    forwarded += [name for name, _, _ in REMOTE_TRISTATE]
    assert len(forwarded) == len(set(forwarded))
    assert not set(forwarded) & set(LOCAL_ONLY_PARAMS)


def test_no_classified_name_is_stale():
    """A renamed or removed parameter must not linger in the table pretending to be forwarded."""
    params = set(inspect.signature(runner.submit).parameters)
    assert _CLASSIFIED <= params, f"{sorted(_CLASSIFIED - params)} no longer exist on `run`"


def test_the_measurement_knobs_are_forwarded():
    """A dropped --token-budget or --k would change the score without changing the label."""
    argv = remote_argv({**BASE, "k": 20, "token_budget": 9000, "seed": 7, "repeats": 5,
                         "model": "claude-sonnet-4-5", "workers": 4})
    assert "--token-budget 9000" in " ".join(argv)
    assert "--k 20" in " ".join(argv)
    assert "--seed 7" in " ".join(argv)


def test_no_renderer_emits_a_retired_flag():
    """Render-side pinned to parse-side: a flag the CLI retires must leave the tables too.

    The 2026-08-13 failure: `--tier`/`--slice` were retired at the parser but stayed in
    REMOTE_VALUE_FLAGS, so the background child (and any cloud task) was handed a command its
    own build refuses -- a run dead at parse time, after submission said "queued"."""
    from memrank.cli.retired import RETIRED_FLAGS

    rendered = {flag for _, flag in (*REMOTE_VALUE_FLAGS, *REMOTE_SWITCHES, *REMOTE_REPEATED)}
    rendered |= {flag for _, *flags in REMOTE_TRISTATE for flag in flags}
    overlap = rendered & set(RETIRED_FLAGS)
    assert not overlap, f"{sorted(overlap)} are retired at the parser but still rendered"


def test_tier_and_slice_ride_the_ref_not_flags():
    """After the submit path derives tier/slice from the ref, they must NOT come back as argv."""
    argv = remote_argv({**BASE, "benchmark_arg": "beam:100k-smoke",
                        "tier": "100k", "slice": "smoke"})
    assert argv[:3] == ["submit", "word-overlap", "beam:100k-smoke"]
    assert "--tier" not in argv and "--slice" not in argv
    assert "100k" not in argv[3:] and "smoke" not in argv[3:]


def test_switches_are_forwarded_only_when_set():
    assert "--verbose" in remote_argv({**BASE, "verbose": True})
    assert "--verbose" not in remote_argv({**BASE, "verbose": False})


def test_the_judge_decision_is_stated_either_way():
    """A switch says nothing when it is off, and silence is the one thing this may not mean.

    Judging is derived from the eval when the flag is omitted, and the task runs a DIFFERENT
    build -- so an absent `--judge` that the task re-derived could come back judged or not
    depending on which side answered. The submitter decides once and the argv carries it.
    """
    assert "--judge" in remote_argv({**BASE, "judge": True})
    assert "--no-judge" not in remote_argv({**BASE, "judge": True})
    assert "--no-judge" in remote_argv({**BASE, "judge": False})
    assert "--judge" not in remote_argv({**BASE, "judge": False})


def test_repeated_options_keep_every_value():
    argv = remote_argv({**BASE, "unit": ["u1", "u2"]})
    assert argv.count("--unit") == 2
    assert "u1" in argv and "u2" in argv


def test_component_overrides_survive():
    argv = remote_argv({**BASE, "overrides": ["embedder=voyage/voyage-4-large"]})
    assert "embedder=voyage/voyage-4-large" in argv


@pytest.mark.parametrize("local", ["on", "output_dir", "run_id"])
def test_local_only_parameters_never_appear(local):
    """`on` above all: a task that inherited it would submit another task."""
    argv = remote_argv({**BASE, local: "cloud"})
    assert "cloud" not in argv[3:]
    assert f"--{local.replace('_', '-')}" not in argv


def test_rendering_argv_does_not_load_the_cli():
    """The API renders argv on a request path; it must not import Typer's execution graph to do it.

    This is the whole reason the table lives in `placement` rather than beside `memrank run`. In a
    subprocess against a real import, because the question is what the module graph actually pulls
    in -- inside this suite `memrank.runner` is already imported and the check would pass vacuously.
    """
    script = ("import sys, memrank.placement.remote_argv\n"
              "print(' '.join(sorted(sys.modules)))\n")
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    loaded = set(done.stdout.split())
    # `click` is deliberately NOT in this set, for the reason tests/repo/test_import_weight.py
    # states about its own `TERMINAL_ONLY`: click reaches the run path from `httpx.__init__`,
    # which imports its own `_main` CLI module, and httpx is the HTTP library every adapter is
    # required to use. Naming it here asserts something about somebody else's package, and
    # since `memrank.systems` put the shipped adapter classes on the front page that assertion
    # is permanently red. What this test guards -- Typer's execution graph -- is `typer` and
    # `memrank.runner`, and those are what it names.
    forbidden = sorted({"typer", "memrank.runner"} & loaded)
    assert forbidden == [], (
        f"importing the argv renderer pulled in {forbidden}. It is a pure table plus a loop; "
        f"the API imports it at module scope and must not pay for the CLI to do so.")
