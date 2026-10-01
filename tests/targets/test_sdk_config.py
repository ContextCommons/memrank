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
"""`sdk_config:` states what memrank has no vocabulary for, and is fenced so it cannot lie.

`components:` is memrank's cross-engine vocabulary, rendered onto the environment variables a
SERVER reads (engine_env.py). An in-process SDK has no environment to render onto -- it takes a
config object whose shape belongs to the library -- and some settings have no memrank vocabulary at
all: which vector store mem0 builds, for instance, which the server image cannot even do
(main.py:238 hardcodes pgvector with no variable to change it).

An opaque passthrough beside a checked declaration is how a receipt starts describing a system
nobody ran, so these pin the two rules that stop it.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from memrank.targets import resolve_target
from memrank.targets.factory import build_adapter
from memrank.targets.manifest import ManifestError, from_dict, to_dict
from tests import withheld

CHROMA = {"vector_store": {"provider": "chroma", "config": {"collection_name": "t"}}}


def _mem0():
    """The shipped `mem0` manifest, which is the worked example throughout this file.

    Every rule here is about `sdk_config` rather than about mem0, but mem0 is the engine whose SDK
    and server genuinely diverge, so it is what the rules are stated against. A tree without the
    manifest skips (`tests/withheld.py`) rather than asserting on a target it does not have.
    """
    withheld.require("mem0")
    return resolve_target("mem0")


def _sdk_target():
    """mem0 re-pointed at the SDK, with components dropped for sdk_config to replace."""
    return replace(_mem0(), transport="sdk", components={},
                   sdk_config=CHROMA)


def test_an_sdk_config_reaches_the_adapter_verbatim():
    adapter = build_adapter(_sdk_target(), verify_engine=False)
    assert adapter.config == CHROMA
    assert adapter.mode == "sdk"


def test_sdk_config_is_refused_over_http():
    """Over HTTP the engine reads the environment, so a config object would be recorded in the
    receipt and obeyed by nothing."""
    target = replace(_mem0(), sdk_config=CHROMA)
    with pytest.raises(ManifestError) as excinfo:
        build_adapter(target, verify_engine=False)
    assert "transport" in str(excinfo.value)


def test_components_and_sdk_config_cannot_both_be_declared():
    """Two vocabularies for the same settings, of which only one reaches an in-process SDK."""
    target = replace(_mem0(), transport="sdk", sdk_config=CHROMA)
    with pytest.raises(ManifestError) as excinfo:
        build_adapter(target, verify_engine=False)
    assert "components" in str(excinfo.value)


def test_a_non_mapping_sdk_config_is_rejected_at_parse_time():
    with pytest.raises(ManifestError):
        from_dict({"name": "t", "kind": "stack", "adapter": "mem0", "transport": "sdk",
                   "sdk_config": ["not", "a", "mapping"]})


def test_an_absent_sdk_config_does_not_change_a_targets_serialisation():
    """Adding a field to the schema is not a change to targets that predate it -- the rule
    tests/targets/test_component_endpoint.py pins for `endpoint`."""
    assert "sdk_config" not in to_dict(_mem0())


def test_a_declared_sdk_config_is_recorded():
    assert to_dict(_sdk_target())["sdk_config"] == CHROMA


# --------------------------------------------------------------------------------------------- #
# Credentials: resolved through memrank's chokepoint, never written in a manifest
# --------------------------------------------------------------------------------------------- #

_WITH_PROVIDERS = {
    "vector_store": {"provider": "chroma", "config": {"collection_name": "t"}},
    "llm": {"provider": "anthropic", "config": {"model": "claude-sonnet-4-6"}},
    "embedder": {"provider": "openai", "config": {"model": "text-embedding-3-large"}},
}


def _adapter_with(monkeypatch, resolved: dict[str, str]):
    """A configured sdk adapter whose credentials resolve ONLY through config.secret."""
    from memrank import config as memrank_config
    monkeypatch.setattr(memrank_config, "secret", lambda name: resolved.get(name))
    target = replace(_mem0(), transport="sdk", components={},
                     sdk_config=_WITH_PROVIDERS)
    return build_adapter(target, verify_engine=False)


def test_a_wallet_only_key_reaches_the_sdk_config(monkeypatch):
    """The case a library reading os.environ would fail: nothing is exported, the key is in the
    wallet, and config.secret is the only way to it."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    adapter = _adapter_with(monkeypatch, {"ANTHROPIC_API_KEY": "wallet-anthropic",
                                          "OPENAI_API_KEY": "wallet-openai"})

    filled = adapter._with_credentials(adapter.config)
    assert filled["llm"]["config"]["api_key"] == "wallet-anthropic"
    assert filled["embedder"]["config"]["api_key"] == "wallet-openai"


def test_filling_credentials_does_not_mutate_the_declared_config(monkeypatch):
    """`self.config` is the object a receipt serialises; a live key must never land in it."""
    adapter = _adapter_with(monkeypatch, {"ANTHROPIC_API_KEY": "a", "OPENAI_API_KEY": "b"})
    adapter._with_credentials(adapter.config)

    assert "api_key" not in adapter.config["llm"]["config"]
    assert "api_key" not in adapter.config["embedder"]["config"]


def test_a_missing_credential_names_the_command_that_fixes_it(monkeypatch):
    from memrank.config import ConfigError

    adapter = _adapter_with(monkeypatch, {"OPENAI_API_KEY": "b"})   # anthropic absent
    with pytest.raises(ConfigError) as excinfo:
        adapter._with_credentials(adapter.config)
    assert "memrank secrets set ANTHROPIC_API_KEY" in str(excinfo.value)


def test_a_keyless_provider_needs_nothing(monkeypatch):
    """chroma is a store, not a vendor; transformers and ollama are local. None has a key."""
    adapter = _adapter_with(monkeypatch, {"ANTHROPIC_API_KEY": "a", "OPENAI_API_KEY": "b"})
    filled = adapter._with_credentials(adapter.config)

    assert "api_key" not in filled["vector_store"].get("config", {})


def test_a_credential_written_into_a_manifest_is_refused():
    """Manifests are committed, so a value here is already leaked."""
    with pytest.raises(ManifestError) as excinfo:
        from_dict({"name": "t", "kind": "stack", "adapter": "mem0", "transport": "sdk",
                   "sdk_config": {"llm": {"config": {"api_key": "sk-leaked"}}}})
    assert "sdk_config.llm.config.api_key" in str(excinfo.value)


@pytest.mark.parametrize("spelling", ["apiKey", "x-api-key", "API_KEY", "authToken", "password"])
def test_credential_spellings_are_all_caught(spelling):
    """The detector canonicalises, so camelCase and header spellings cannot slip past."""
    with pytest.raises(ManifestError):
        from_dict({"name": "t", "kind": "stack", "adapter": "mem0", "transport": "sdk",
                   "sdk_config": {"llm": {spelling: "sk-leaked"}}})


def test_an_innocent_field_that_merely_contains_key_is_allowed():
    """`collection_name`, `top_k`, `token_budget` must not trip the detector."""
    target = from_dict({"name": "t", "kind": "stack", "adapter": "mem0", "transport": "sdk",
                        "sdk_config": {"vector_store": {"config": {"collection_name": "c"}},
                                       "retrieval": {"top_k": 5, "token_budget": 5000}}})
    assert target.sdk_config["retrieval"]["top_k"] == 5


# --------------------------------------------------------------------------------------------- #
# In-process lifecycle: isolation and teardown that containers used to provide for free
# --------------------------------------------------------------------------------------------- #

class _FakeMemory:
    """Records what mem0 would have been asked to build and delete."""

    built_with: dict = {}

    def __init__(self, cfg):
        type(self).built_with = cfg
        self.deleted: list[str] = []

    @classmethod
    def from_config(cls, cfg):
        return cls(cfg)

    def delete_all(self, user_id=None):
        self.deleted.append(user_id)


def _sdk_adapter(monkeypatch, sdk_config):
    from memrank import config as memrank_config
    from memrank.adapters import mem0 as mem0_mod

    monkeypatch.setattr(memrank_config, "secret", lambda name: f"resolved-{name}")
    monkeypatch.setattr(mem0_mod, "_try_import_mem0", lambda: _FakeMemory)
    monkeypatch.setattr(mem0_mod, "_mem0_module", lambda: type("m", (), {"__version__": "0.1.114"}))
    target = replace(_mem0(), transport="sdk", components={},
                     sdk_config=sdk_config)
    return build_adapter(target, verify_engine=False)


def test_the_collection_name_is_per_run(monkeypatch):
    """Chroma outlives the process, so without this run 2 retrieves run 1's memories -- and the
    contamination reads as an accuracy change rather than a bug."""
    adapter = _sdk_adapter(monkeypatch, {
        "vector_store": {"provider": "chroma", "config": {"collection_name": "arena_{run_id}"}}})
    adapter.prepare("run-abc")

    assert _FakeMemory.built_with["vector_store"]["config"]["collection_name"] == "arena_run-abc"


def test_the_declared_config_keeps_its_placeholder(monkeypatch):
    """Substitution is per-run, so the manifest's own value must survive for the next unit."""
    adapter = _sdk_adapter(monkeypatch, {
        "vector_store": {"provider": "chroma", "config": {"collection_name": "arena_{run_id}"}}})
    adapter.prepare("run-abc")

    assert adapter.config["vector_store"]["config"]["collection_name"] == "arena_{run_id}"


def test_sdk_mode_declares_and_performs_destructive_cleanup(monkeypatch):
    """`baseline` sets the flag and drops its store together; SDK mode must do the same, because
    no placement teardown exists to do it (placement/inprocess.py returns None)."""
    adapter = _sdk_adapter(monkeypatch, {"vector_store": {"provider": "chroma"}})
    assert adapter.cleanup_is_destructive is True

    adapter.prepare("run-abc")
    memory = adapter._memory
    adapter.cleanup()

    assert memory.deleted == ["run-abc"]


def test_http_mode_leaves_cleanup_to_the_placement():
    """Over HTTP `compose down -v` destroys the store, so the adapter must not claim to."""
    adapter = build_adapter(_mem0(), verify_engine=False)
    assert adapter.cleanup_is_destructive is False


def test_sdk_mode_reports_the_real_engine_version(monkeypatch):
    """The arm exists to compare mem0 generations, so a receipt saying "unknown" defeats it."""
    adapter = _sdk_adapter(monkeypatch, {"vector_store": {"provider": "chroma"}})
    adapter.prepare("run-abc")

    assert adapter.engine_version == "0.1.114"


# --------------------------------------------------------------------------------------------- #
# Two mem0 generations, two spellings of the same search
# --------------------------------------------------------------------------------------------- #

class _SearchV1:
    """mem0ai 0.1.114: depth is `limit`, scoping is a direct `user_id` keyword."""

    def search(self, query, *, user_id=None, agent_id=None, run_id=None,
               limit=100, filters=None, threshold=None):
        return {"results": []}


class _SearchV2:
    """mem0 2.0.1: depth is `top_k`, scoping goes through `filters`; no user_id parameter."""

    def search(self, query, *, top_k=20, filters=None, threshold=0.1, rerank=False, **kwargs):
        return {"results": []}


def _kwargs_against(memory):
    from memrank.adapters.mem0 import Mem0

    adapter = Mem0(mode="sdk")
    adapter._memory = memory
    return adapter._search_kwargs(5, "run-abc")


def test_the_older_sdk_is_called_with_limit_and_user_id():
    """Calling 0.1.114 with `top_k` raises TypeError -- at the first retrieve, after a unit's
    documents have already been extracted and paid for."""
    assert _kwargs_against(_SearchV1()) == {"limit": 5, "user_id": "run-abc", "threshold": 0.0}


def test_the_newer_sdk_is_called_with_top_k_and_filters():
    """2.x has no user_id parameter, so scoping must go through filters."""
    assert _kwargs_against(_SearchV2()) == {"top_k": 5, "filters": {"user_id": "run-abc"},
                                            "threshold": 0.0}


def test_the_threshold_is_permissive_in_both():
    """2.x defaults to 0.1, which filters real matches out; 0.1.114 defaults to None."""
    for memory in (_SearchV1(), _SearchV2()):
        assert _kwargs_against(memory)["threshold"] == 0.0


def test_the_engine_version_is_known_before_prepare(monkeypatch):
    """The receipt is built in run_cell BEFORE the first unit is prepared, so a version resolved
    when the SDK handle is created is always too late -- the run records "unknown" and the arm
    cannot say which mem0 produced its number. Asserting without prepare() is the point."""
    from memrank.adapters import mem0 as mem0_mod

    monkeypatch.setattr(mem0_mod, "_mem0_module",
                        lambda: type("m", (), {"__version__": "0.1.114"}))
    adapter = mem0_mod.Mem0(mode="sdk")

    assert adapter.engine_version == "0.1.114"


def test_http_mode_still_reports_the_container_tag(monkeypatch):
    """Over HTTP the engine is a container whose tag arrives through MEM0_ENGINE_VERSION; a locally
    installed mem0 says nothing about what that container runs."""
    from memrank.adapters import mem0 as mem0_mod

    monkeypatch.setattr(mem0_mod, "_mem0_module",
                        lambda: type("m", (), {"__version__": "0.1.114"}))
    adapter = mem0_mod.Mem0(mode="http")

    assert adapter.engine_version != "0.1.114"
