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
"""cloud_launch.launch -- the one render->register->run implementation both submitters use.

These assertions used to live against the CLI's `--on cloud`; they moved here when the CLI
became an API client, because the behaviors they guard (sizing, sidecars, the command's
shape, the launch target) now execute inside the accounts API -- through exactly this
function. AWS is stubbed at the cloud_submit seam; everything above it is real.
"""
from __future__ import annotations

import pytest

from memrank.placement import cloud_launch, cloud_submit, registry
from tests import withheld

CONTEXT = {
    "region": "us-east-1", "cluster": "memrank-bench-staging", "subnet": "subnet-aaa",
    "security_group": "sg-bbb", "log_group": "/ecs/memrank-bench-staging",
    "execution_role_arn": "arn:aws:iam::1:role/exec", "task_role_arn": "arn:aws:iam::1:role/task",
    "artifact_bucket": "memrank-artifacts", "runner_repository": "r.example/bench",
    "engines_repository": "r.example/engines",
    "secret_arns": {"ANTHROPIC_API_KEY": "arn:a", "OPENAI_API_KEY": "arn:o",
                    "VOYAGE_API_KEY": "arn:v"},
}


@pytest.fixture
def aws(monkeypatch):
    """cloud_submit with no AWS behind it; records what would have been launched."""
    monkeypatch.setenv("MEMRANK_ENGINES_TAG", "poc-1")
    captured: dict = {}

    def _register(taskdef, launch):
        captured["taskdef"] = taskdef
        captured["launch"] = launch
        return "arn:aws:ecs:us-east-1:1:task-definition/memrank-bench:1"

    def _resolve(image, *, platform, client=None):
        repository, tag = registry.split_reference(image)
        return registry.Resolution(repository=repository, tag=tag,
                                   platform_digest="sha256:engine", index_digest="sha256:index",
                                   platform=platform)

    monkeypatch.setattr(cloud_submit, "register", _register)
    monkeypatch.setattr(cloud_submit, "submit", lambda arn, launch: "arn:task/deadbeef")
    monkeypatch.setattr(cloud_submit, "verify_image_exists", lambda **kw: "sha256:engine")
    monkeypatch.setattr(registry, "resolve", _resolve)
    return captured


def _launch(aws, ref, argv=None):
    return cloud_launch.launch(CONTEXT, ref, argv or ["submit", ref, "demo"],
                               run_id="20260801-120000__demo__abc123", image_tag="abc1234")


def test_an_in_process_target_gets_a_single_container_task(aws):
    result = _launch(aws, "word-overlap")
    taskdef = aws["taskdef"]
    assert [c["name"] for c in taskdef["containerDefinitions"]] == ["memrank"]
    assert (taskdef["cpu"], taskdef["memory"]) == ("1024", "2048")
    assert result.task_arn == "arn:task/deadbeef"
    assert result.artifact_prefix == "cloud-runs/20260801-120000__demo__abc123"


def test_a_stack_target_gets_its_sidecars_and_bigger_sizing(aws):
    withheld.require("mem0")
    _launch(aws, "mem0")
    taskdef = aws["taskdef"]
    assert {c["name"] for c in taskdef["containerDefinitions"]} == {"memrank", "engine",
                                                                    "datastore"}
    assert (taskdef["cpu"], taskdef["memory"]) == ("4096", "8192")


def test_the_manifest_decides_the_engine_environment(aws):
    """M6's defect, still closed: the embedder comes from the target, never a hardcode."""
    withheld.require("mem0")
    _launch(aws, "mem0")
    env = {p["name"]: p["value"]
           for c in aws["taskdef"]["containerDefinitions"] if c["name"] == "engine"
           for p in c["environment"]}
    assert env["MEM0_EMBEDDER_MODEL"] == "text-embedding-3-small"


def test_the_command_evaluates_then_uploads_and_never_resubmits(aws):
    """If `--on cloud` survived into the task's command, each task would launch another."""
    _launch(aws, "word-overlap", ["submit", "word-overlap", "demo"])
    command = aws["taskdef"]["containerDefinitions"][0]["command"][0]
    # The environment prefix carries where to publish live progress. Deliberately not a
    # REMOTE_CLI_CONTRACT bump -- an unknown variable cannot make an older image fail to parse.
    assert command.startswith("MEMRANK_PROGRESS_BUCKET=memrank-artifacts memrank submit "
                              "word-overlap demo")
    assert "--on none" in command      # pinned by the renderer, once
    assert "cloud" not in command.split("&&")[0].split()
    assert "--output-dir /work/results" in command
    assert "cloud-runs/20260801-120000__demo__abc123" in command


def test_the_launch_target_comes_from_the_context(aws):
    _launch(aws, "word-overlap")
    launch = aws["launch"]
    assert (launch.cluster, launch.subnet, launch.security_group) == \
        ("memrank-bench-staging", "subnet-aaa", "sg-bbb")


def test_a_digest_that_cannot_be_resolved_refuses_the_launch(aws, monkeypatch):
    """Deliberately the OPPOSITE of what this asserted before, when a failed lookup degraded to an
    empty digest so that "provenance is not worth a failed launch".

    That reasoning holds only while the digest is decoration. It is now what the containers RUN,
    and a receipt that names an artifact nobody verified is worse than a run that did not happen --
    especially since the refusal costs nothing: it lands before any task is registered.
    """
    withheld.require("mem0")

    def _boom(image, *, platform, client=None):
        raise registry.RegistryError(f"could not resolve {image!r}: ghcr flaked")
    monkeypatch.setattr(registry, "resolve", _boom)

    with pytest.raises(registry.RegistryError, match="ghcr flaked"):
        _launch(aws, "mem0")
    assert "taskdef" not in aws, "nothing may be registered for a run that cannot name its images"


def test_what_the_task_runs_is_a_digest_not_a_tag(aws):
    """The cloud half of resolve-once: ECS is handed identities, so it cannot resolve the tag
    itself and land on a different build than the local run of the same target ref."""
    _launch(aws, "hindsight")
    images = [c["image"] for c in aws["taskdef"]["containerDefinitions"] if c["name"] == "engine"]

    assert images == ["ghcr.io/vectorize-io/hindsight@sha256:engine"]


def test_both_digests_reach_the_harness(aws):
    """What makes a cloud row comparable with a local one: same fields, same function."""
    _launch(aws, "hindsight")
    env = {p["name"]: p["value"]
           for c in aws["taskdef"]["containerDefinitions"] if c["name"] == "memrank"
           for p in c["environment"]}

    assert env["HINDSIGHT_ENGINE_IMAGE_DIGEST"] == "sha256:engine"
    assert env["HINDSIGHT_ENGINE_IMAGE_INDEX_DIGEST"] == "sha256:index"
    assert env["HINDSIGHT_ENGINE_IMAGE_PLATFORM"] == "linux/amd64"
    assert env["HINDSIGHT_ENGINE_VERSION"] == "0.6.2"


# -- the progress channel, and where its credential is put -------------------------------

PROGRESS_ENV = {"MEMRANK_PROGRESS_URL": "https://api.example.com/runs/r-1/progress",
                "MEMRANK_RUN_TOKEN": "mrr_secret-value.deadbeef"}


def _harness_env(aws) -> dict[str, str]:
    return {p["name"]: p["value"]
            for c in aws["taskdef"]["containerDefinitions"] if c["name"] == "memrank"
            for p in c["environment"]}


def test_the_progress_channel_reaches_the_harness_container(aws):
    cloud_launch.launch(CONTEXT, "baseline", ["submit", "baseline", "demo"],
                        run_id="r-1", image_tag="abc1234", harness_env_extra=PROGRESS_ENV)

    assert _harness_env(aws) | PROGRESS_ENV == _harness_env(aws)


def test_the_run_token_is_not_in_the_rendered_command(aws):
    """The command string is readable through DescribeTaskDefinition and, inside the task, through
    /proc/<pid>/cmdline. A container's environment is neither, which is why the token goes there --
    the same exposure Buildkite re-execs over a pipe to avoid."""
    cloud_launch.launch(CONTEXT, "baseline", ["submit", "baseline", "demo"],
                        run_id="r-1", image_tag="abc1234", harness_env_extra=PROGRESS_ENV)

    assert "mrr_secret-value" not in aws["taskdef"]["containerDefinitions"][0]["command"][0]


def test_no_sidecar_is_given_the_run_token(aws):
    """Containers in one task share an ENI, never an environment. The engine images are vendor
    code; the credential that reaches the API must not be in reach of them."""
    cloud_launch.launch(CONTEXT, "mem0", ["submit", "mem0", "demo"],
                        run_id="r-1", image_tag="abc1234", harness_env_extra=PROGRESS_ENV)

    for container in aws["taskdef"]["containerDefinitions"]:
        if container["name"] == "memrank":
            continue
        assert "MEMRANK_RUN_TOKEN" not in {p["name"] for p in container["environment"]}


def test_a_stack_targets_topology_survives_the_progress_channel(aws):
    """The extras are merged, not substituted: an engine URL the harness needs must still be
    there once the launch path adds its own variables."""
    cloud_launch.launch(CONTEXT, "hindsight", ["submit", "hindsight", "demo"],
                        run_id="r-1", image_tag="abc1234", harness_env_extra=PROGRESS_ENV)
    env = _harness_env(aws)

    assert env["HINDSIGHT_API_URL"].startswith("http://localhost:")
    assert env["MEMRANK_RUN_TOKEN"] == PROGRESS_ENV["MEMRANK_RUN_TOKEN"]


def test_a_launch_with_no_progress_channel_renders_as_before(aws):
    """The operator path mints nothing. It must not gain an empty variable that reads as
    configuration, and must not fail for want of one."""
    _launch(aws, "baseline")

    assert "MEMRANK_RUN_TOKEN" not in _harness_env(aws)
