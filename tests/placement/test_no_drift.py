"""The gate: local and cloud may not describe different systems under one target name.

This is the spec's headline defect, expressed as a test. `deploy/ecs/taskdef.mem0.json.tpl` welded
`BAAI/bge-small-en-v1.5` / `384` into the file, so asking for `mem0:voyage` in the cloud ran
bge-small while the leaderboard row still said voyage. Both renderers now derive components from
`memrank.targets.engine_env`, and these tests prove it for EVERY target rather than for the one
someone remembered to check.

Enumerated from the catalog, per the `CLAUDE.md` chokepoint rule: a new target is covered the moment
it is added, and fails here until it renders on both sides. A literal list would silently pass.
"""
from __future__ import annotations

import pytest

from memrank.placement.cloud import AwsContext, render_taskdef
from memrank.placement.graph import engine_image, reference_parts
from memrank.placement.local import render_compose
from memrank.targets import list_targets, resolve_target
from memrank.targets.engine_env import (
    ENGINE_COMMAND,
    ENGINE_ENV,
    component_env,
    engine_command,
    harness_env,
)
from tests import withheld

AWS = AwsContext(
    family="memrank-bench", region="us-east-1", log_group="/ecs/memrank-bench",
    execution_role_arn="arn:aws:iam::1:role/exec", task_role_arn="arn:aws:iam::1:role/task",
    memrank_image="registry.example/bench:v1", command="memrank submit ...",
    secret_arns={"ANTHROPIC_API_KEY": "arn:aws:ssm:::parameter/anthropic",
                 "OPENAI_API_KEY": "arn:aws:ssm:::parameter/openai",
                 "VOYAGE_API_KEY": "arn:aws:ssm:::parameter/voyage"})

STACK_TARGETS = [r for r in list_targets() if resolve_target(r).kind == "stack"]


def _local_env(target):
    services = render_compose(target, project="p")["services"]
    return services[target.service]["environment"]


def _cloud_containers(target):
    """Every compiled container keyed by name -- the same keys the graph uses."""
    return {c["name"]: c for c in render_taskdef(target, aws=AWS)["containerDefinitions"]}


def _cloud_env(target):
    engine = _cloud_containers(target)[target.service]
    return {pair["name"]: pair["value"] for pair in engine["environment"]}


@pytest.mark.parametrize("ref", STACK_TARGETS)
def test_both_placements_declare_identical_components(ref):
    """The gate. Component values must be byte-identical wherever the target runs."""
    target = resolve_target(ref)
    names = set(ENGINE_ENV[target.adapter].values())
    local = {k: v for k, v in _local_env(target).items() if k in names}
    cloud = {k: v for k, v in _cloud_env(target).items() if k in names}
    assert local == cloud == component_env(target)


@pytest.mark.parametrize("ref", STACK_TARGETS)
def test_every_stack_target_renders_on_both_placements(ref):
    """Coverage itself is the assertion: an unrenderable target fails rather than being skipped."""
    target = resolve_target(ref)
    assert render_compose(target, project="p")["services"]
    assert render_taskdef(target, aws=AWS)["containerDefinitions"]


@pytest.mark.parametrize("ref", STACK_TARGETS)
def test_the_same_image_build_runs_on_both_placements(ref):
    """EVERY container, not just the engine -- which is exactly how the last divergence survived.

    While this compared the engine alone, `mem0:bge-tei` ran the vendor's published embedder
    locally and our mirrored copy in the cloud, at whatever tag the environment happened to
    supply. One target name, two systems measured, and nothing to notice it. Both placements now
    compile the same declared graph, so the comparison is over its whole service list.
    """
    target = resolve_target(ref)
    local = render_compose(target, project="p")["services"]
    cloud = _cloud_containers(target)
    assert {name: service["image"] for name, service in local.items()} == \
        {name: cloud[name]["image"] for name in local}


@pytest.mark.parametrize("ref", STACK_TARGETS)
def test_the_same_start_command_runs_on_both_placements(ref):
    """The same image started two different ways is the same defect as two different images.

    It happened: the compose renderer applied mem0's `alembic upgrade head` to every engine while
    cloud guarded it with `if adapter == "mem0"`, so `--on local` told a Node image to run a Python
    migration tool and three of four engines could not start.

    What this locks in is that both renderers still READ ``ENGINE_COMMAND`` -- it fails the moment
    either one hardcodes a command again. It cannot catch a wrong value *in* the table, since both
    sides would then be wrong together; that is the same limit test_template_parity.py has, and
    only a live run closes it.
    """
    target = resolve_target(ref)
    local = render_compose(target, project="p")["services"][target.service].get("command")
    cloud = _cloud_containers(target)[target.service].get("command")
    assert local == cloud == engine_command(target)


@pytest.mark.parametrize("ref", STACK_TARGETS)
def test_every_stack_adapter_states_a_start_command_decision(ref):
    """Coverage of the table itself: a new engine must decide, not inherit whatever was first.

    ``None`` is a valid decision -- "the image's CMD is correct" -- but it has to be written down.
    """
    assert resolve_target(ref).adapter in ENGINE_COMMAND


@pytest.mark.parametrize("ref", STACK_TARGETS)
def test_the_harness_is_told_the_same_things_on_both_placements(ref):
    """The harness half of the gate, which nothing compared until it had already diverged.

    Local returned only the engine's base URL, so a local run silently lost the 900s adapter
    timeout, the paired intra-task token, and every provenance pin -- including the image digest
    that :mod:`memrank.provenance.engine` needs to build a component purl. A local row and a cloud
    row were therefore NOT comparable, which is the property this whole design exists to guarantee.

    This is the CLOUD call site; the local one is asserted in test_local_placement.py, because it
    needs a provisioned endpoint rather than a rendered document. Both must equal ``harness_env``.
    Compared by name only -- the base URL and image digest are runtime facts that legitimately
    differ, and where an engine listens is topology.
    """
    target = resolve_target(ref)
    rendered = {p["name"] for p in _cloud_containers(target)["memrank"]["environment"]}
    repository, tag = reference_parts(engine_image(target))
    expected = set(harness_env(target, url="http://engine:1234", engines_repo=repository,
                               tag=tag, digest="sha256:" + "0" * 64))
    assert rendered == expected
    assert f"{target.adapter.upper()}_TIMEOUT_S" in rendered
    assert f"{target.adapter.upper()}_ENGINE_IMAGE_DIGEST" in rendered


def test_the_variant_the_old_template_could_not_express():
    """The retired .tpl said bge-small/384 for every mem0 ref, so no variant could differ from it.

    The two that proved this -- `mem0:voyage` and `mem0:bge-tei` -- moved to the research lane on
    2026-08-19, so the variant is written here. What the gate protects is the renderer reading the
    MANIFEST, and that is now the only place it can be checked in this repo.
    """
    withheld.require("mem0")
    cloud = _cloud_env(resolve_target("mem0", ["embedder=voyage/voyage-4-large",
                                               "embedder.dims=1024"]))
    assert cloud["MEM0_EMBEDDER_MODEL"] == "voyage-4-large"
    assert cloud["MEM0_EMBEDDING_DIMS"] == "1024"


def test_variants_of_one_engine_render_differently():
    """Two refs sharing an adapter must not collapse to one configuration."""
    withheld.require("mem0")
    assert _cloud_env(resolve_target("mem0", ["embedder=voyage/voyage-4-large",
                                              "embedder.dims=1024"])) != \
        _cloud_env(resolve_target("mem0"))


@pytest.mark.parametrize("ref", STACK_TARGETS)
def test_topology_is_allowed_to_differ_and_does(ref):
    """The gate must compare components ONLY -- forcing hostnames equal would break both placements.

    Compose gives each service its own DNS name; under awsvpc every container in a task shares one
    namespace. The compiler rewrites neighbour names from the graph's own service list, so this is
    a translation of what was declared rather than a table someone must remember to extend.
    """
    target = resolve_target(ref)
    services = render_compose(target, project="p")["services"]
    if "datastore" not in services:
        pytest.skip(f"{ref} runs no datastore sidecar")
    assert _local_env(target)["POSTGRES_HOST"] == "datastore"     # compose service name
    assert _cloud_env(target)["POSTGRES_HOST"] == "localhost"     # awsvpc shared namespace


def test_a_neighbour_named_inside_a_url_is_rewritten_too(sidecar_stack):
    """`http://embedder/v1` is a host reference like any other, and was missed by the old table.

    Uses the fixture stack rather than a shipped ref: no built-in target runs a self-hosted
    embedder since the mem0 matched pair moved to the research lane, and the rewrite is the
    compiler's behaviour rather than one target's property."""
    assert _local_env(sidecar_stack)["MEM0_EMBEDDER_ENDPOINT"] == "http://embedder/v1"
    assert _cloud_env(sidecar_stack)["MEM0_EMBEDDER_ENDPOINT"] == "http://localhost/v1"
