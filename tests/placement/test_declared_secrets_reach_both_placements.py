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
"""A credential a target declares reaches the engine on EVERY placement.

The defect these pin: the cloud renderer built its credential list from the provider table alone
while the local renderer also honoured a target's declared `secrets:`. A target declaring its own
credentials therefore launched correctly under `--on local` and reached Fargate with them missing --
and nothing failed loudly, because the wallet preflight passes on names the taskdef then omits.

`defaults.on` is `cloud`, so the broken path was the default one.

Enumerated rather than asserted once per renderer: per-surface defence is leaky by construction,
and a third placement added later must fail here rather than ship the same hole (CLAUDE.md).
"""

from __future__ import annotations

import pytest

from memrank.placement.cloud import AwsContext, _engine_secrets
from memrank.placement.local import engine_secrets
from memrank.targets.manifest import from_dict

#: supermemory declares no launch providers, so nothing is DERIVED and these exercise only the
#: declared path. An engine with a derived key would trip the preflight on that key first.
DECLARED = {"AM_SLM_TOKEN": ["LLM_API_KEY", "EMBEDDING_API_KEY"]}
TARGET = from_dict({"name": "t", "kind": "stack", "adapter": "supermemory", "secrets": DECLARED})

AWS = AwsContext(
    family="memrank-bench", region="us-east-1", log_group="/ecs/memrank-bench",
    execution_role_arn="arn:aws:iam::1:role/exec", task_role_arn="arn:aws:iam::1:role/task",
    memrank_image="registry.example/bench:v1", command="memrank submit ...",
    secret_arns={"AM_SLM_TOKEN": "arn:aws:ssm:::parameter/am-slm"})


def _local(monkeypatch) -> set[str]:
    monkeypatch.setenv("AM_SLM_TOKEN", "edge-token")
    return set(engine_secrets(TARGET))


def _cloud(monkeypatch) -> set[str]:
    return {entry["name"] for entry in _engine_secrets(TARGET, AWS)}


#: Every renderer that puts credentials in front of an engine. Adding a placement means adding it
#: here; leaving it out is what this file exists to catch.
RENDERERS = {"local": _local, "cloud": _cloud}


@pytest.mark.parametrize("renderer", sorted(RENDERERS))
def test_every_placement_injects_a_declared_credential(renderer, monkeypatch):
    assert RENDERERS[renderer](monkeypatch) >= {"LLM_API_KEY", "EMBEDDING_API_KEY"}


def test_a_declared_credential_without_an_arn_refuses_to_render():
    """Rendering without one would launch a task that dies on a missing credential after the pull."""
    from memrank.placement.base import CloudRenderError

    bare = AwsContext(
        family=AWS.family, region=AWS.region, log_group=AWS.log_group,
        execution_role_arn=AWS.execution_role_arn, task_role_arn=AWS.task_role_arn,
        memrank_image=AWS.memrank_image, command=AWS.command)

    with pytest.raises(CloudRenderError, match="AM_SLM_TOKEN"):
        _engine_secrets(TARGET, bare)


def test_both_placements_carry_the_same_credential_under_the_same_names(monkeypatch):
    """One table names them, so the two answers cannot drift apart again."""
    assert _local(monkeypatch) == _cloud(monkeypatch)
