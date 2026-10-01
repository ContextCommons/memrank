"""One-cell submission validates plan membership and acknowledgements first."""

from __future__ import annotations

from memrank.application.planning import plan_sweep
from memrank.application.submission import submit_experiment
from memrank.application.types import JudgeSettings, SweepAxes, SweepRequest


def test_unknown_experiment_is_refused_without_dispatch(monkeypatch):
    monkeypatch.setattr("memrank.application.planning.config.secret", lambda name: "set")
    plan = plan_sweep(SweepRequest(targets=["word-overlap"], evals=["demo"]))

    outcome = submit_experiment(plan, "not-in-plan")

    assert outcome.outcome == "refused"
    assert outcome.refusals[0].code == "unknown_experiment"


def test_judging_a_real_corpus_needs_no_acknowledgement(monkeypatch):
    """`data_egress` was the only warning any planner raised, and it is gone: `--judge` names the
    provider it sends to, and this door never asked -- it supplied the acknowledgement itself.

    Plans it, rather than submitting it, because submission spawns a background child. What is
    under test is that nothing is REFUSED, and a plan with no warnings is what makes that true.
    """
    monkeypatch.setattr("memrank.application.planning.config.secret", lambda name: "set")
    request = SweepRequest(targets=["word-overlap"], evals=["locomo"],
                           axes=SweepAxes(judges=[JudgeSettings(enabled=True)]))

    plan = plan_sweep(request)

    assert plan.warnings == []
    assert plan.refusals == []


def test_valid_plan_dispatches_only_selected_cell(monkeypatch):
    monkeypatch.setattr("memrank.application.planning.config.secret", lambda name: "set")
    plan = plan_sweep(SweepRequest(targets=["word-overlap"], evals=["demo"],
                                   axes=SweepAxes(seeds=[1, 2])))
    selected = plan.experiments[1]
    seen = []
    monkeypatch.setattr("memrank.application.submission._submit_local",
                        lambda sent_plan, cell, acknowledgements: seen.append(cell) or {
                            "run_id": "run-1", "experiment_id": cell.experiment_id,
                            "state": "queued", "placement": "none"})

    outcome = submit_experiment(plan, selected.experiment_id)

    assert outcome.outcome == "ok"
    assert seen == [selected]
