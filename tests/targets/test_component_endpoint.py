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
"""A component states WHERE its provider is reached, and the receipt records it.

Every provider but one carries its address implicitly: a vendor's API is where the vendor says it
is, `transformers` runs in-process, `regex` is the engine's own code. `openai-compatible` names a
protocol rather than a place, so its URL used to travel through the ambient environment -- and two
runs against different deployments produced byte-identical component records.
"""

from __future__ import annotations

import pytest

from memrank.targets.engine_env import EngineEnvError, component_env
from memrank.targets.manifest import ManifestError, definition_digest, from_dict

#: A stand-in for a self-hosted OpenAI-compatible deployment. Any absolute URL will do -- the
#: assertions below check that the address travels intact and that stating one changes the
#: digest, never which host it is. RFC 2606's reserved domain keeps a real deployment's
#: address out of the test corpus.
SLM = "https://slm.example.com/v1"


def atomicmemory(**component_overrides):
    components = {
        "llm": {"provider": "openai-compatible", "model": "am-slm-core"},
        "embedder": {"provider": "openai-compatible", "model": "nomic-embed-text", "dims": 768},
    }
    for role, extra in component_overrides.items():
        components[role] = {**components[role], **extra}
    return from_dict({"name": "t", "kind": "stack", "adapter": "atomicmemory",
                      "components": components})


def test_an_endpoint_reaches_the_variable_the_engine_reads():
    """core reads LLM_API_URL / EMBEDDING_API_URL (packages/core/src/config.ts:1342)."""
    env = component_env(atomicmemory(llm={"endpoint": SLM}, embedder={"endpoint": SLM}))

    assert env["LLM_API_URL"] == SLM
    assert env["EMBEDDING_API_URL"] == SLM


def test_an_unstated_endpoint_is_omitted_rather_than_blanked():
    """An empty string is a value the engine would act on; absence lets its default stand."""
    env = component_env(atomicmemory())

    assert "LLM_API_URL" not in env
    assert "EMBEDDING_API_URL" not in env


def test_an_endpoint_an_adapter_cannot_name_is_refused():
    """Silent drift: the manifest would promise an address the engine never received."""
    target = from_dict({"name": "t", "kind": "stack", "adapter": "supermemory",
                        "components": {"llm": {"provider": "anthropic", "model": "x",
                                               "endpoint": SLM}}})

    with pytest.raises(EngineEnvError, match="no env var for it"):
        component_env(target)


@pytest.mark.parametrize("bad", ["", "   ", 7])
def test_a_malformed_endpoint_is_rejected(bad):
    with pytest.raises(ManifestError, match="must be a non-empty URL"):
        atomicmemory(llm={"endpoint": bad})


def test_adding_the_field_did_not_change_the_digest_of_targets_without_one():
    """A digest identifies the definition a run used; a schema addition is not a definition change.

    The expected value was read off the tree at 4a462eb, before `endpoint` existed. It is pinned
    rather than recomputed: a test that derives both sides would pass however the serializer drifts.
    """
    stated = definition_digest(atomicmemory(llm={"endpoint": SLM}))

    assert definition_digest(atomicmemory()) == (
        "b6ddd86948120857f3a492d4cefca183e8c9264da0bed0f3349c0647339c2d7a")
    assert stated != definition_digest(atomicmemory())


def _hindsight(**llm):
    return from_dict({"name": "t", "kind": "stack", "adapter": "hindsight",
                      "components": {"llm": {"provider": "openai", "model": "m", **llm}},
                      "secrets": {"ROUTER_TOKEN": ["HINDSIGHT_API_LLM_API_KEY"]}})


def test_a_component_with_its_own_endpoint_spends_no_provider_key():
    """Its provider's key would otherwise be handed to whatever the endpoint is."""
    from memrank.targets.engine_env import engine_secret_vars

    assert engine_secret_vars(_hindsight(endpoint=SLM)) == {
        "ROUTER_TOKEN": ["HINDSIGHT_API_LLM_API_KEY"]}


def test_without_an_endpoint_the_provider_key_is_still_required():
    from memrank.targets.catalog import required_secrets_for

    assert required_secrets_for(_hindsight()) == ["OPENAI_API_KEY", "ROUTER_TOKEN"]


def test_hindsight_and_mem0_endpoints_reach_the_variables_their_servers_read():
    assert component_env(_hindsight(endpoint=SLM))["HINDSIGHT_API_LLM_BASE_URL"] == SLM
    mem0 = from_dict({"name": "t", "kind": "stack", "adapter": "mem0", "components": {
        "llm": {"provider": "openai", "model": "m", "endpoint": SLM},
        "embedder": {"provider": "openai", "model": "e", "dims": 1536, "endpoint": SLM}}})
    env = component_env(mem0)
    assert (env["MEM0_LLM_ENDPOINT"], env["MEM0_EMBEDDER_ENDPOINT"]) == (SLM, SLM)
