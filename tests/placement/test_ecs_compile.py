"""Compiling a Compose graph into ECS containers, and refusing what it cannot compile.

Every rule here maps something with a published meaning on both sides, so these tests are readable
as "does the translation say what Compose says". The refusals matter more than the translations:
each one exists because ECS cannot express a Compose feature faithfully, and a compiler that
quietly drops what it does not understand produces a task definition that launches and measures
something other than what was declared.

None of these paths had ever executed. They were added with the compiler and left untested, and
writing these tests found three defects -- a valid `depends_on` form that crashed, a missing image
that raised `KeyError`, and a host rewrite that replaced service names inside unrelated values.
"""
from __future__ import annotations

import pytest

from memrank.placement.base import CloudRenderError
from memrank.placement.cloud import AwsContext, render_taskdef
from memrank.placement.ecs_compile import (
    HEALTH_CHECK_LIMITS,
    LOCAL_ONLY,
    UNSUPPORTED,
    compile_service,
    duration_seconds,
    localhost_environment,
    runtime_platform,
)

LOG = {"logDriver": "awslogs", "options": {"awslogs-stream-prefix": "engine"}}
SERVICES = ["engine", "datastore", "embedder"]
AWS = AwsContext(
    family="memrank-bench", region="us-east-1", log_group="/ecs/x",
    execution_role_arn="arn:e", task_role_arn="arn:t", memrank_image="i", command="c",
    secret_arns={"ANTHROPIC_API_KEY": "arn:a", "OPENAI_API_KEY": "arn:o",
                 "VOYAGE_API_KEY": "arn:v"})


def _compiled(declared, name="engine"):
    return compile_service(name, declared, services=SERVICES, log_configuration=LOG)


# --- refusals: what ECS cannot express, named rather than dropped ------------------------------ #

@pytest.mark.parametrize("key,value", [("volumes", ["./data:/data"]), ("build", {"context": "."}),
                                       ("network_mode", "host"), ("extends", {"service": "base"})])
def test_a_feature_with_no_ecs_equivalent_is_refused_by_name(key, value):
    """Enumerated from UNSUPPORTED, so a key added there without a test fails here."""
    with pytest.raises(CloudRenderError) as exc:
        _compiled({"image": "i", key: value})
    assert key in str(exc.value) and "engine" in str(exc.value)


def test_every_unsupported_key_is_covered_by_the_test_above():
    """The parametrize list is a literal; this is what keeps it honest as UNSUPPORTED grows."""
    covered = {"volumes", "build", "network_mode", "extends"}
    assert set(UNSUPPORTED) == covered


def test_a_local_only_key_is_left_out_of_the_container_rather_than_refused():
    """`shm_size` has no Fargate counterpart; refusing it would stop hindsight, atomicmemory and
    mem0 in the cloud, so the compiled container simply carries nothing for it."""
    assert LOCAL_ONLY == ("shm_size",)
    assert _compiled({"image": "i", "shm_size": "2g"}) == _compiled({"image": "i"})


def test_a_service_naming_no_image_is_refused_rather_than_raising_keyerror():
    """`build:` is refused above; a service with neither is the remaining case.

    It used to reach `declared["image"]` and raise a bare KeyError from inside the compiler, which
    tells an author nothing about which service is wrong or why.
    """
    with pytest.raises(CloudRenderError) as exc:
        _compiled({"environment": {"A": "b"}}, name="datastore")
    assert "datastore" in str(exc.value) and "image" in str(exc.value)


def test_an_untranslatable_wait_condition_is_refused_naming_what_is_known():
    """`service_completed_successfully` has an ECS counterpart we have not proven, so it is
    refused rather than translated untested -- the honest position for a condition no graph uses."""
    with pytest.raises(CloudRenderError) as exc:
        _compiled({"image": "i",
                   "depends_on": {"datastore": {"condition": "service_completed_successfully"}}})
    message = str(exc.value)
    assert "service_completed_successfully" in message
    assert "service_healthy" in message and "service_started" in message


# --- translations ------------------------------------------------------------------------------ #

def test_both_depends_on_forms_are_translated():
    """The list form is valid Compose and means `service_started`.

    It used to crash with `AttributeError: 'list' object has no attribute 'items'` -- neither a
    translation nor a refusal, and from inside the compiler rather than at the graph.
    """
    mapping = _compiled({"image": "i", "depends_on": {"datastore": {"condition": "service_healthy"}}})
    assert mapping["dependsOn"] == [{"containerName": "datastore", "condition": "HEALTHY"}]
    short = _compiled({"image": "i", "depends_on": ["datastore", "embedder"]})
    assert short["dependsOn"] == [{"containerName": "datastore", "condition": "START"},
                                  {"containerName": "embedder", "condition": "START"}]


def test_a_health_check_is_translated_with_its_durations_converted():
    check = _compiled({"image": "i", "healthcheck": {
        "test": ["CMD-SHELL", "true"], "interval": "5s", "timeout": "10s",
        "retries": 9, "start_period": "1m"}})["healthCheck"]
    assert check == {"command": ["CMD-SHELL", "true"], "interval": 5, "timeout": 10,
                     "retries": 9, "startPeriod": 60}


def test_a_service_declaring_no_health_check_gets_none():
    """ECS treats a container with no healthCheck as always ready; inventing one would change
    ordering that the graph deliberately left unstated."""
    assert "healthCheck" not in _compiled({"image": "i"})


def test_only_the_harness_may_end_the_task():
    """Every compiled container is non-essential: a sidecar exiting must not kill the task before
    results upload. The harness container is added by the renderer, not compiled from the graph."""
    assert _compiled({"image": "i"})["essential"] is False


@pytest.mark.parametrize("value,seconds", [("5s", 5), ("30s", 30), ("1m", 60), ("2h", 7200),
                                           (45, 45), ("45", 45)])
def test_durations_convert_to_the_whole_seconds_ecs_wants(value, seconds):
    assert duration_seconds(value, service="engine", key="interval") == seconds


@pytest.mark.parametrize("value", ["500ms", "1h30m", "forever", ""])
def test_a_duration_this_compiler_cannot_convert_is_refused_naming_the_field(value):
    """Sub-second rounds to zero, which ECS rejects with a message naming neither file nor field.

    Compound forms like `1h30m` are legal Compose that this compiler does not parse; refusing is
    honest, silently reading `1h` and dropping `30m` would not be.
    """
    with pytest.raises(CloudRenderError) as exc:
        duration_seconds(value, service="datastore", key="start_period")
    assert "datastore" in str(exc.value) and "start_period" in str(exc.value)


# --- ECS's health-check limits, which Compose does not have ------------------------------------ #

@pytest.mark.parametrize("field,compose_key,over", [
    ("retries", "retries", 11), ("interval", "interval", "301s"),
    ("timeout", "timeout", "61s"), ("startPeriod", "start_period", "301s")])
def test_a_health_check_outside_ecs_limits_is_refused_naming_the_limit(field, compose_key, over):
    """`retries: 20` on a datastore reached AWS and came back as a 502 naming a container the author
    never wrote. Refused here, the message names the service, the field and what ECS accepts."""
    declared = {"test": ["CMD-SHELL", "true"], compose_key: over}
    with pytest.raises(CloudRenderError) as exc:
        _compiled({"image": "i", "healthcheck": declared}, name="datastore")
    low, high = HEALTH_CHECK_LIMITS[field]
    assert "datastore" in str(exc.value) and f"{low}-{high}" in str(exc.value)


@pytest.mark.parametrize("field", sorted(HEALTH_CHECK_LIMITS))
def test_a_value_at_the_limit_is_accepted(field):
    """A bound, not an off-by-one. `interval: 300s` is legal and must render."""
    low, high = HEALTH_CHECK_LIMITS[field]
    compose_key = {"startPeriod": "start_period"}.get(field, field)
    value = high if field == "retries" else f"{high}s"
    check = _compiled({"image": "i", "healthcheck": {
        "test": ["CMD-SHELL", "true"], compose_key: value}})["healthCheck"]
    assert check[field] == high
    assert low <= check[field] <= high


def test_every_shipped_graph_renders_within_ecs_limits():
    """The test that would have caught the 502, enumerated so a new graph cannot reintroduce it.

    Compose bounds none of these fields, so a health check that works perfectly under `docker
    compose up` can still be rejected at RegisterTaskDefinition -- and botocore's own model carries
    no ranges, so validating the rendered document against it does not see them either.
    """
    from memrank.placement.cloud import render_taskdef
    from memrank.targets import list_targets, resolve_target

    for ref in [r for r in list_targets() if resolve_target(r).kind == "stack"]:
        for container in render_taskdef(resolve_target(ref), aws=AWS)["containerDefinitions"]:
            for field, value in (container.get("healthCheck") or {}).items():
                if field in HEALTH_CHECK_LIMITS:
                    low, high = HEALTH_CHECK_LIMITS[field]
                    assert low <= value <= high, f"{ref}/{container['name']}: {field}={value}"


# --- the localhost rewrite: a host, and only a host -------------------------------------------- #

def test_a_neighbour_is_rewritten_where_it_is_actually_a_host():
    """Under awsvpc a task is one network namespace, so a service name addresses localhost.

    Both real forms: a bare hostname, and the authority of a URL.
    """
    rewritten = localhost_environment(
        {"POSTGRES_HOST": "datastore",
         "MEM0_EMBEDDER_ENDPOINT": "http://embedder/v1",
         "PG_DSN": "postgresql://user@datastore:5432/db",
         "ADDR": "datastore:5432"}, SERVICES)
    assert rewritten == {"POSTGRES_HOST": "localhost",
                         "MEM0_EMBEDDER_ENDPOINT": "http://localhost/v1",
                         "PG_DSN": "postgresql://user@localhost:5432/db",
                         "ADDR": "localhost:5432"}


def test_a_service_name_that_is_not_a_host_is_left_alone():
    """The rewrite used to replace a service name ANYWHERE in ANY value, so `MODE: engine` became
    `MODE: localhost`. An engine whose configuration happens to contain a word matching a service
    name would have been silently misconfigured, with no error and nothing to fail.

    A service naming ITSELF is the case that can be told apart: it is far more likely saying what
    it is than where to reach itself.
    """
    values = {"MODE": "engine", "NOTE": "the datastore is fine", "PREFIX": "embedder-v2"}
    assert localhost_environment(values, SERVICES, "engine") == values


def test_a_self_reference_written_as_a_url_is_still_rewritten():
    """That one is unambiguous -- and under awsvpc the service name resolves to nothing, so leaving
    it would trade a silent misconfiguration for a DNS failure at runtime."""
    assert localhost_environment({"SELF": "http://engine:8000/x"}, SERVICES, "engine") == \
        {"SELF": "http://localhost:8000/x"}


def test_a_neighbours_name_as_a_whole_value_is_a_host_whatever_the_variable_means():
    """The residual ambiguity, pinned so it is a decision rather than a surprise.

    A compiler with no schema cannot distinguish `POSTGRES_HOST: datastore` from a variable that
    merely happens to hold a neighbour's name. Requiring a URL would be unambiguous and would fail
    to rewrite the commonest form there is.
    """
    assert localhost_environment({"ANYTHING": "datastore"}, SERVICES, "engine") == \
        {"ANYTHING": "localhost"}


def test_a_longer_service_name_is_not_half_rewritten_by_a_shorter_one():
    """A graph with `db` and `db-replica` must not have the second rewritten into `localhost-replica`."""
    assert localhost_environment({"H": "db-replica"}, ["db", "db-replica"]) == {"H": "localhost"}


# --- runtimePlatform: one per task, taken from the graph --------------------------------------- #

def _graph(*platforms):
    return {"services": {f"s{i}": {"image": "i", **({"platform": p} if p else {})}
                         for i, p in enumerate(platforms)}}


def test_agreeing_services_collapse_to_one_platform():
    assert runtime_platform(_graph("linux/amd64", "linux/amd64", None)) == \
        {"cpuArchitecture": "X86_64", "operatingSystemFamily": "LINUX"}


def test_a_graph_that_states_no_platform_defaults_to_x86():
    """What Fargate assumes, made explicit rather than left to the API's default."""
    assert runtime_platform(_graph(None, None))["cpuArchitecture"] == "X86_64"


def test_an_arm_graph_is_compiled_as_arm_rather_than_silently_run_as_x86():
    assert runtime_platform(_graph("linux/arm64"))["cpuArchitecture"] == "ARM64"


def test_services_that_disagree_about_platform_are_refused_naming_both():
    """An ECS task has one runtimePlatform and Fargate does not emulate, so this cannot run as one
    task -- and must say so here, where the symptom is a message, rather than at launch, where the
    symptom is a container that never starts."""
    with pytest.raises(CloudRenderError) as exc:
        runtime_platform(_graph("linux/amd64", "linux/arm64"))
    message = str(exc.value)
    assert "s0=linux/amd64" in message and "s1=linux/arm64" in message


def test_a_platform_with_no_fargate_equivalent_is_refused_listing_what_is_known():
    with pytest.raises(CloudRenderError) as exc:
        runtime_platform(_graph("linux/riscv64"))
    assert "linux/riscv64" in str(exc.value) and "linux/amd64" in str(exc.value)


# --- the same refusals through the command an operator actually runs --------------------------- #
#
# The acceptance criterion is that `targets render --for cloud` refuses, naming the service. Asserted
# through the real path, because a compiler that refuses correctly is worth nothing if the renderer
# above it turns the refusal into a traceback.

def test_a_graph_whose_services_disagree_about_platform_is_refused_end_to_end(declare_target):
    """An ECS task has one runtimePlatform and Fargate does not emulate."""
    target = declare_target("mixed-arch", {
        "engine": {"image": "example/engine:1", "platform": "linux/amd64"},
        "sidecar": {"image": "example/sidecar:1", "platform": "linux/arm64"}})
    with pytest.raises(CloudRenderError) as exc:
        render_taskdef(target, aws=AWS)
    assert "engine=linux/amd64" in str(exc.value) and "sidecar=linux/arm64" in str(exc.value)


def test_a_graph_declaring_a_volume_is_refused_end_to_end(declare_target):
    """The first thing reached for when adapting a dev compose file, and the first thing an ECS
    task cannot honour -- a bind mount has no meaning in a filesystem pulled from a registry."""
    target = declare_target("mounted", {
        "engine": {"image": "example/engine:1", "volumes": ["./data:/data"]}})
    with pytest.raises(CloudRenderError) as exc:
        render_taskdef(target, aws=AWS)
    assert "engine" in str(exc.value) and "volumes" in str(exc.value)


#: The `.aws-context.json` shape `--aws` reads -- the raw terraform outputs, not `AwsContext`'s
#: fields. Supplied explicitly so the render reaches the refusal under test: without it the
#: command stops earlier, at `config.aws_context()`, refusing for want of a context file.
AWS_CONTEXT_FILE = {
    "region": "us-east-1", "cluster": "c", "subnet": "subnet-1", "security_group": "sg-1",
    "log_group": "/ecs/x", "execution_role_arn": "arn:e", "task_role_arn": "arn:t",
    "artifact_bucket": "b", "runner_repository": "r", "engines_repository": "e",
    "secret_arns": {"ANTHROPIC_API_KEY": "arn:a", "OPENAI_API_KEY": "arn:o",
                    "VOYAGE_API_KEY": "arn:v"}}


def test_the_cli_prints_a_refusal_as_an_error_not_a_traceback(declare_target, tmp_path):
    """`targets render` catches CloudRenderError so an author sees the reason, not a stack."""
    import json

    from typer.testing import CliRunner

    from memrank.runner import app

    context = tmp_path / "aws-context.json"
    context.write_text(json.dumps(AWS_CONTEXT_FILE), encoding="utf-8")
    declare_target("mounted-cli", {"engine": {"image": "e:1", "volumes": ["./d:/d"]}})
    result = CliRunner().invoke(app, ["targets", "render", "--for", "cloud",
                                      "--aws", str(context), "mounted-cli"])
    assert result.exit_code == 1
    assert "volumes" in result.output and "Traceback" not in result.output
