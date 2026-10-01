"""Contract tests for deterministic, non-executing sweep planning."""

from __future__ import annotations

from memrank.application.planning import plan_sweep
from memrank.application.types import ModelRef, Routing, SweepAxes, SweepRequest
from tests import withheld


def test_model_axis_produces_distinct_stable_experiments(monkeypatch):
    withheld.require("mem0")
    monkeypatch.setattr("memrank.application.planning.config.secret", lambda name: "set")
    request = SweepRequest(
        targets=["mem0"], evals=["demo"],
        axes=SweepAxes(engine_llms=[
            ModelRef(provider="anthropic", model="claude-a"),
            ModelRef(provider="openai", model="gpt-a"),
        ]),
    )

    first = plan_sweep(request)
    second = plan_sweep(request)

    assert len(first.experiments) == 2
    assert [cell.experiment_id for cell in first.experiments] == [
        cell.experiment_id for cell in second.experiments]
    assert first.plan_hash == second.plan_hash


def test_routing_changes_plan_but_not_experiment_identity(monkeypatch):
    monkeypatch.setattr("memrank.application.planning.config.secret", lambda name: "set")
    local = SweepRequest(targets=["word-overlap"], evals=["demo"])
    cloud = local.model_copy(update={"routing": Routing(placement="cloud", org="acme")})

    local_plan = plan_sweep(local)
    cloud_plan = plan_sweep(cloud)

    assert local_plan.experiments[0].experiment_id == cloud_plan.experiments[0].experiment_id
    assert local_plan.plan_hash != cloud_plan.plan_hash


def test_cloud_credentials_are_org_scoped_and_revalidated_at_submission(monkeypatch):
    withheld.require("mem0")
    monkeypatch.setattr("memrank.application.planning.config.secret", lambda name: None)
    request = SweepRequest(targets=["mem0"], evals=["demo"],
                           routing=Routing(placement="cloud", org="acme"))

    plan = plan_sweep(request)

    assert plan.requirements
    assert all(item.scope == "org" and item.satisfied is None for item in plan.requirements)


def test_matrix_over_limit_is_refused_without_experiments():
    request = SweepRequest(targets=["word-overlap"], evals=["demo"],
                           axes=SweepAxes(seeds=list(range(101))))

    plan = plan_sweep(request)

    assert plan.experiments == []
    assert plan.refusals[0].code == "sweep_too_large"


def test_duplicate_cells_are_reported(monkeypatch):
    monkeypatch.setattr("memrank.application.planning.config.secret", lambda name: None)
    request = SweepRequest(targets=["word-overlap", "word-overlap"], evals=["demo"])

    plan = plan_sweep(request)

    assert len(plan.experiments) == 1
    assert plan.deduplicated == 1
