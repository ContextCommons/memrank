"""Registering, starting and watching an ECS task.

Exercised through ``botocore.stub.Stubber``, which validates every request against the real ECS
service model -- so a misspelled parameter or a wrong shape fails here rather than on a paid launch.
It cannot tell us that ECS *accepts* a generated task definition semantically; only a real run does
that, and until one has happened this path is unproven (recorded in tech-debt.md).
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from botocore.stub import ANY, Stubber

from memrank.placement import cloud_submit
from memrank.placement.cloud_submit import CloudLaunchError, LaunchTarget, TaskOutcome
from tests import withheld

LAUNCH = LaunchTarget(region="us-east-1", cluster="memrank-bench-dev",
                      subnet="subnet-aaa", security_group="sg-bbb")
TASK_ARN = "arn:aws:ecs:us-east-1:1:task/memrank-bench-dev/abc123"
TASKDEF_ARN = "arn:aws:ecs:us-east-1:1:task-definition/memrank-bench:7"
_MINIMAL_TASKDEF = {"family": "memrank-bench",
                    "containerDefinitions": [{"name": "memrank", "image": "r.example/b:v1"}]}


@pytest.fixture
def ecs(monkeypatch):
    """A stubbed ECS client, substituted for the lazily-created real one."""
    import boto3

    client = boto3.client("ecs", region_name="us-east-1",
                          aws_access_key_id="t", aws_secret_access_key="t")
    stub = Stubber(client)
    stub.activate()
    monkeypatch.setattr(cloud_submit, "_client", lambda launch: client)
    yield stub
    stub.deactivate()


def _task(status="RUNNING", exit_code=None, reason="", stopped_at=None):
    container = {"name": "memrank"}
    if exit_code is not None:
        container["exitCode"] = exit_code
    task = {"taskArn": TASK_ARN, "lastStatus": status,
            "stoppedReason": reason, "containers": [container]}
    if stopped_at is not None:
        task["stoppedAt"] = stopped_at
    return {"tasks": [task]}


def test_register_returns_the_definition_arn(ecs):
    ecs.add_response("register_task_definition",
                     {"taskDefinition": {"taskDefinitionArn": TASKDEF_ARN}},
                     {"family": ANY, "containerDefinitions": ANY})
    assert cloud_submit.register(_MINIMAL_TASKDEF, LAUNCH) == TASKDEF_ARN


def test_register_rejects_a_malformed_definition_before_launching(ecs):
    """Parameter validation is the point of doing this in Python: a definition missing a required
    field fails here rather than after the AWS call is billed.

    The response is queued but never consumed -- botocore validates the request before the call is
    made, which is precisely the behaviour being asserted.
    """
    ecs.add_response("register_task_definition",
                     {"taskDefinition": {"taskDefinitionArn": TASKDEF_ARN}},
                     {"family": ANY, "containerDefinitions": ANY})
    with pytest.raises(CloudLaunchError, match="containerDefinitions"):
        cloud_submit.register({"family": "memrank-bench"}, LAUNCH)


def test_a_real_rendered_taskdef_satisfies_the_ecs_service_model(ecs):
    """The M6 renderer's output, validated against botocore's ECS model.

    Not proof that ECS accepts it -- only a real register-task-definition call proves that -- but it
    catches an unknown field or a wrong shape, which is what the old templates' bare json.loads
    could never do.
    """
    from memrank.placement.cloud import AwsContext, render_taskdef
    from memrank.targets import resolve_target

    withheld.require("mem0")
    taskdef = render_taskdef(
        resolve_target("mem0"),
        aws=AwsContext(family="memrank-bench-mem0", region="us-east-1", log_group="/ecs/x",
                       execution_role_arn="arn:aws:iam::1:role/e",
                       task_role_arn="arn:aws:iam::1:role/t",
                       memrank_image="r.example/bench:v1", command="memrank submit ...",
                       secret_arns={"ANTHROPIC_API_KEY": "arn:a", "OPENAI_API_KEY": "arn:o"}))
    ecs.add_response("register_task_definition",
                     {"taskDefinition": {"taskDefinitionArn": TASKDEF_ARN}},
                     dict.fromkeys(taskdef, ANY))
    assert cloud_submit.register(taskdef, LAUNCH) == TASKDEF_ARN


def test_submit_uses_fargate_never_spot(ecs):
    """A Spot reclamation mid-run discards the whole evaluation, and runs are long."""
    ecs.add_response("run_task", {"tasks": [{"taskArn": TASK_ARN}]},
                     {"cluster": LAUNCH.cluster, "taskDefinition": TASKDEF_ARN,
                      "launchType": "FARGATE", "networkConfiguration": ANY})
    assert cloud_submit.submit(TASKDEF_ARN, LAUNCH) == TASK_ARN


def test_submit_assigns_a_public_ip(ecs):
    """This is what lets the task reach ECR without a NAT gateway; the SG has no ingress."""
    ecs.add_response("run_task", {"tasks": [{"taskArn": TASK_ARN}]},
                     {"cluster": ANY, "taskDefinition": ANY, "launchType": ANY,
                      "networkConfiguration": {"awsvpcConfiguration": {
                          "subnets": ["subnet-aaa"], "securityGroups": ["sg-bbb"],
                          "assignPublicIp": "ENABLED"}}})
    cloud_submit.submit(TASKDEF_ARN, LAUNCH)


def test_a_failure_response_is_not_reported_as_a_launch(ecs):
    """run-task can return 200 with an empty task list and a failures array."""
    ecs.add_response("run_task",
                     {"tasks": [], "failures": [{"reason": "RESOURCE:MEMORY"}]},
                     {"cluster": ANY, "taskDefinition": ANY, "launchType": ANY,
                      "networkConfiguration": ANY})
    with pytest.raises(CloudLaunchError, match="RESOURCE:MEMORY"):
        cloud_submit.submit(TASKDEF_ARN, LAUNCH)


def test_a_full_availability_zone_is_named_as_transient_not_as_a_breakage(ecs):
    """The 2026-08-03 locomo sweep died on this. "Capacity is unavailable" means ask again;
    every other launch failure means fix something. Conflating them costs a fleet."""
    ecs.add_response("run_task",
                     {"tasks": [], "failures": [{"reason": "Capacity is unavailable at this "
                                                 "time. Please try again later or in a "
                                                 "different availability zone"}]},
                     {"cluster": ANY, "taskDefinition": ANY, "launchType": ANY,
                      "networkConfiguration": ANY})
    with pytest.raises(cloud_submit.CloudCapacityError) as exc:
        cloud_submit.submit(TASKDEF_ARN, LAUNCH)

    assert "subnet-aaa" in str(exc.value), "name the zone that is full"
    assert "transient" in str(exc.value)


def test_a_capacity_error_is_still_a_launch_error(ecs):
    """Callers that only care that the launch failed must not have to learn a new type."""
    ecs.add_response("run_task",
                     {"tasks": [], "failures": [{"reason": "Capacity is unavailable"}]},
                     {"cluster": ANY, "taskDefinition": ANY, "launchType": ANY,
                      "networkConfiguration": ANY})
    with pytest.raises(CloudLaunchError):
        cloud_submit.submit(TASKDEF_ARN, LAUNCH)


def test_describe_many_reads_the_harness_container_exit_code(ecs):
    """Sidecars are non-essential and exit non-zero routinely; only memrank's code is the verdict."""
    ecs.add_response("describe_tasks", _task("STOPPED", exit_code=0),
                     {"cluster": ANY, "tasks": [TASK_ARN]})
    got = cloud_submit.describe_many([TASK_ARN], LAUNCH)[TASK_ARN]
    assert got.succeeded and got.stopped


def test_a_nonzero_exit_is_a_failure_not_a_success(ecs):
    ecs.add_response("describe_tasks", _task("STOPPED", exit_code=1, reason="Essential exited"),
                     {"cluster": ANY, "tasks": [TASK_ARN]})
    got = cloud_submit.describe_many([TASK_ARN], LAUNCH)[TASK_ARN]
    assert got.stopped and not got.succeeded
    assert got.stopped_reason == "Essential exited"


def test_a_task_ecs_no_longer_knows_is_absent_not_invented(ecs):
    """ECS forgets stopped tasks after ~1h. Absence is the caller's signal (the accounts API
    reports it as `unknown`); inventing an outcome here would launder a guess into a record."""
    ecs.add_response("describe_tasks", {"tasks": []}, {"cluster": ANY, "tasks": [TASK_ARN]})
    assert cloud_submit.describe_many([TASK_ARN], LAUNCH) == {}


def test_outcome_still_running_is_neither_success_nor_stopped():
    running = TaskOutcome(last_status="RUNNING")
    assert not running.stopped and not running.succeeded


def test_a_stopped_task_carries_the_time_ecs_says_it_stopped(ecs):
    """ECS reports `stoppedAt` and we were discarding it, so every downstream record dated the
    stop from whenever someone first looked. A run that ended at 14:00 and was first read at
    22:00 was stored as stopping at 22:00 -- plausible, wrong, and silent."""
    ended = datetime(2026, 8, 13, 14, 0, tzinfo=timezone.utc)
    ecs.add_response("describe_tasks", _task("STOPPED", exit_code=0, stopped_at=ended),
                     {"cluster": ANY, "tasks": [TASK_ARN]})

    assert cloud_submit.describe_many([TASK_ARN], LAUNCH)[TASK_ARN].stopped_at == ended


def test_a_running_task_has_no_stop_time(ecs):
    """`stoppedAt` is absent until it stops, and inventing one would be the same defect in the
    opposite direction."""
    ecs.add_response("describe_tasks", _task("RUNNING"), {"cluster": ANY, "tasks": [TASK_ARN]})

    assert cloud_submit.describe_many([TASK_ARN], LAUNCH)[TASK_ARN].stopped_at is None
