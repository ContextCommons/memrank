"""``_submit_local`` builds a real ``_ExecOpts``, and nothing else in the suite proves it.

Every other submission test monkeypatches ``_submit_local`` itself (see
``tests/application/test_submission.py``), so the ``_ExecOpts(...)`` construction inside it never
executes under test. A field added to that bundle without a matching keyword here would ship
green, and an API- or browser-submitted ``--on local`` run would fail at submission with a
``TypeError`` -- after the caller believed it had been accepted.

These tests stub the runner internals *around* the construction, never the construction itself,
so the bundle is the code under test.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from memrank import runner  # noqa: F401 - kept while some stubs remain here
from memrank.application.planning import plan_sweep
from memrank.application.submission import _submit_local
from memrank.application.types import SweepRequest
from memrank.orchestration import placement_gate, resolve, sweep


def _fake_run() -> SimpleNamespace:
    """A minted run that accepts annotations instead of touching the run directory."""
    return SimpleNamespace(status=SimpleNamespace(annotate=lambda **fields: None),
                           run_dir=SimpleNamespace(name="run-1"))


@pytest.fixture
def captured_opts(monkeypatch) -> list:
    """Stub everything ``_submit_local`` reaches except the ``_ExecOpts`` construction.

    ``_sweep_gates`` is the bundle's first consumer, so capturing its argument proves the real
    construction ran: a missing required field raises before this is ever reached.
    """
    seen: list = []
    monkeypatch.setattr(resolve, "_run_targets", lambda **kwargs: ("beam", []))
    monkeypatch.setattr(placement_gate, "_require_placement_ready", lambda on: None)
    monkeypatch.setattr(resolve, "_ensure_run_credentials", lambda **kwargs: None)
    monkeypatch.setattr(sweep, "question_gates",
                        lambda **kwargs: (["unit"], kwargs["judge_cfg"]))
    monkeypatch.setattr(sweep, "_sweep_gates", lambda factories, opts: seen.append(opts))
    monkeypatch.setattr(sweep, "_mint_local_runs", lambda *args, **kwargs: [_fake_run()])
    monkeypatch.setattr(sweep, "_spawn_child", lambda params, runs: None)
    return seen


def _beam_experiment(monkeypatch):
    """A planned single-cell beam experiment, whose eval ref differs from its registry name."""
    monkeypatch.setattr("memrank.application.planning.config.secret", lambda name: "set")
    plan = plan_sweep(SweepRequest(targets=["word-overlap"], evals=["beam:100k-smoke"]))
    return plan, plan.experiments[0]


def test_submit_local_builds_a_real_exec_opts(monkeypatch, captured_opts):
    plan, experiment = _beam_experiment(monkeypatch)

    _submit_local(plan, experiment, [])

    assert len(captured_opts) == 1


def test_exec_opts_carries_the_canonical_eval_ref_not_the_registry_name(monkeypatch,
                                                                       captured_opts):
    """``benchmark`` keys the registry; ``eval_ref`` names the evaluation and files the artifact.

    Passing the registry name as both would let two beam variants overwrite one another's
    results, which is why the bundle carries them separately.
    """
    plan, experiment = _beam_experiment(monkeypatch)

    _submit_local(plan, experiment, [])

    opts = captured_opts[0]
    assert opts.benchmark == "beam"
    assert opts.eval_ref == experiment.eval_ref == "beam:100k-smoke"
