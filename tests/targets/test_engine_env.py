"""The one table naming the env vars that configure an engine's components.

Three surfaces have to agree on these names: the engine process, the local compose renderer, and the
cloud taskdef renderer. Until now that agreement was a coincidence -- it held for mem0 because mem0's
server happens to use the names the adapter convention assumes, and it already failed for
atomicmemory and hindsight. These tests pin the names to the templates they were read from and make
an unnameable component a loud error rather than a silently dropped value.
"""
from __future__ import annotations

import pytest

from memrank.targets import list_targets, resolve_target
from memrank.targets.engine_env import ENGINE_ENV, EngineEnvError, component_env
from memrank.targets.manifest import Component, Manifest
from tests import withheld


def test_mem0_names_match_what_the_server_reads():
    """The env NAMES mem0's server reads, transcribed from the cloud taskdef template (retired in
    M6; see git history). Asserted against the shipped `mem0` -- the matched variants that used to
    stand in here moved to the research lane, and the names are the engine's, not a variant's."""
    withheld.require("mem0")
    assert component_env(resolve_target("mem0")) == {
        "MEM0_LLM_PROVIDER": "openai",
        "MEM0_LLM_MODEL": "gpt-4o-mini",
        "MEM0_EMBEDDER_PROVIDER": "openai",
        "MEM0_EMBEDDER_MODEL": "text-embedding-3-small",
        "MEM0_EMBEDDING_DIMS": "1536",
    }


def test_hindsight_uses_its_own_api_prefixed_names():
    """HINDSIGHT_API_LLM_*, not HINDSIGHT_LLM_* -- taskdef.hindsight.json.tpl:44-45."""
    assert component_env(resolve_target("hindsight")) == {
        "HINDSIGHT_API_LLM_PROVIDER": "anthropic",
        "HINDSIGHT_API_LLM_MODEL": "claude-sonnet-4-5-20250929",
    }


def test_atomicmemory_uses_unprefixed_names():
    """Our own engine reads EMBEDDING_MODEL/EMBEDDING_DIMENSIONS -- taskdef.atomicmemory.json.tpl:49-53.

    The adapter reads ATOMICMEMORY_EMBEDDER_MODEL instead, which is why the receipt recorded None
    for our own engine before this table existed.
    """
    got = component_env(resolve_target("atomicmemory"))
    assert got["EMBEDDING_MODEL"] == "Xenova/all-MiniLM-L6-v2"
    assert got["EMBEDDING_DIMENSIONS"] == "384"
    assert got["LLM_MODEL"] == "claude-sonnet-4-5-20250929"


def test_supermemory_exposes_no_knobs():
    """An empty component env is a real answer, not a missing one."""
    withheld.require("supermemory")
    assert component_env(resolve_target("supermemory")) == {}


def test_a_component_with_no_env_name_is_loud():
    """The load-bearing rule: a declared value that cannot reach the engine must not be silent."""
    unnameable = Manifest(name="x", kind="stack", adapter="supermemory",
                          components={"llm": Component(provider="anthropic")})
    with pytest.raises(EngineEnvError, match=r"llm\.provider.*'supermemory' has no env var"):
        component_env(unnameable)


def test_an_unknown_adapter_is_loud():
    with pytest.raises(EngineEnvError, match="no engine-env table"):
        component_env(Manifest(name="x", kind="stack", adapter="nonesuch",
                               components={"llm": Component(provider="anthropic")}))


@pytest.mark.parametrize("ref", [r for r in list_targets()
                                 if resolve_target(r).kind == "stack"])
def test_every_stack_target_is_nameable(ref):
    """Enumerated from the catalog: adding a target with an untabled component fails here."""
    component_env(resolve_target(ref))


def test_unset_fields_are_omitted_not_blanked():
    """A component that states only a provider must not emit an empty model.

    Both roles are declared because mem0's adapter exposes both, and an unstated ROLE is refused --
    a blank knob is not neutral, the engine fills it. Unstated FIELDS within a declared role are
    the thing under test here, and they are legitimately omitted.
    """
    got = component_env(Manifest(name="x", kind="stack", adapter="mem0",
                                 components={"embedder": Component(provider="voyage"),
                                             "llm": Component(provider="anthropic")}))
    assert got == {"MEM0_EMBEDDER_PROVIDER": "voyage", "MEM0_LLM_PROVIDER": "anthropic"}


def test_table_covers_every_adapter_that_has_a_stack_target():
    """No stack target may reference an adapter the table has never heard of."""
    adapters = {resolve_target(r).adapter for r in list_targets()
                if resolve_target(r).kind == "stack"}
    assert adapters <= set(ENGINE_ENV)


# --- readiness is per engine ------------------------------------------------------------------ #

@pytest.mark.parametrize("ref", [r for r in list_targets()
                                 if resolve_target(r).kind == "stack"])
def test_every_stack_target_has_a_readiness_probe(ref):
    """Enumerated from the catalog. A missing probe does not fail loudly -- the harness waits on a
    health check that can never pass, which is how a hindsight task sat in PENDING until killed."""
    from memrank.targets.engine_env import readiness_path, readiness_probe

    command, start_period = readiness_probe(resolve_target(ref))
    assert command and start_period > 0
    assert readiness_path(resolve_target(ref)).startswith("/")


def test_an_adapter_without_a_probe_is_loud():
    from memrank.targets.engine_env import readiness_probe

    with pytest.raises(EngineEnvError, match="no readiness probe"):
        readiness_probe(Manifest(name="x", kind="stack", adapter="nonesuch"))


def test_the_probe_targets_the_manifest_port():
    """A probe against the wrong port hangs exactly like a probe against the wrong path."""
    from memrank.targets.engine_env import readiness_probe

    withheld.require("supermemory")
    command, _ = readiness_probe(resolve_target("supermemory"))
    assert ":6767/" in command


def test_a_concrete_target_leaving_an_exposed_knob_blank_is_refused():
    """The defect this exists for: `memrank submit mem0` started a Fargate task that died with
    `openai.OpenAIError: Missing credentials`, because mem0 defaults its unset embedder to
    text-embedding-3-small. With a key present it would have SUCCEEDED, measuring an embedder no
    manifest declared and no receipt recorded."""
    import pytest

    from memrank.targets.engine_env import EngineEnvError

    with pytest.raises(EngineEnvError, match="embedder"):
        component_env(Manifest(name="incomplete", kind="stack", adapter="mem0",
                               components={"llm": Component(provider="anthropic")}))


def test_a_base_may_leave_a_knob_blank_because_that_is_what_it_is_for():
    """Its variants fill it. Refusing here would make inheritance impossible."""
    got = component_env(Manifest(name="base", kind="stack", adapter="mem0", abstract=True,
                                 components={"llm": Component(provider="anthropic")}))

    assert got == {"MEM0_LLM_PROVIDER": "anthropic"}


def test_an_adapter_that_exposes_nothing_is_never_incomplete():
    """supermemory's table is empty: nothing is settable, so nothing can be left unstated."""
    from memrank.targets.engine_env import unstated_components

    assert unstated_components(
        Manifest(name="supermemory", kind="stack", adapter="supermemory")) == []
