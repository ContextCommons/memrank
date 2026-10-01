"""Building an adapter from a resolved manifest, and the manifest/env cross-check."""
from __future__ import annotations

import pytest

from memrank.targets import factory, resolve_target
from tests import withheld

VOYAGE_ENV = {"MEM0_EMBEDDER_PROVIDER": "voyage", "MEM0_EMBEDDER_MODEL": "voyage-4-large",
              "MEM0_EMBEDDING_DIMS": "1024", "MEM0_LLM_PROVIDER": "anthropic",
              "MEM0_LLM_MODEL": "claude-sonnet-4-5-20250929"}


def _voyage_target():
    """A mem0 stack configured the way our matched variant is: Anthropic extraction, Voyage 1024d.

    Reached through overrides rather than by ref. `mem0:voyage` was that ref until 2026-08-19, when
    it moved to the research lane -- but the components are what this module needs, since the
    manifest/env cross-check compares component values against MEM0_* variables, and the shipped
    `mem0` names OpenAI for both roles.
    """
    withheld.require("mem0")
    return resolve_target("mem0", ["llm=anthropic/claude-sonnet-4-5-20250929",
                                   "embedder=voyage/voyage-4-large", "embedder.dims=1024"])


def _clear_mem0_env(monkeypatch):
    for var in ("MEM0_EMBEDDER_PROVIDER", "MEM0_EMBEDDER_MODEL", "MEM0_EMBEDDING_DIMS",
                "MEM0_LLM_PROVIDER", "MEM0_LLM_MODEL"):
        monkeypatch.delenv(var, raising=False)


def test_builds_the_adapter_named_by_the_manifest(monkeypatch):
    _clear_mem0_env(monkeypatch)
    adapter = factory.build_adapter(resolve_target("word-overlap"))
    assert adapter.name == "word-overlap"


def test_declares_manifest_components_on_the_instance(monkeypatch):
    _clear_mem0_env(monkeypatch)
    # verify_engine=False: this asserts what the manifest DECLARES. Engine verification is
    # tested in test_engine_verification.py against a stub, so no test depends on a live server.
    cfg = factory.build_adapter(_voyage_target(), verify_engine=False).effective_config()
    assert cfg["embedder"] == {"provider": "voyage", "model": "voyage-4-large", "dims": 1024}
    assert cfg["engine"]["transport"] == "http"


def test_agreeing_environment_is_accepted(monkeypatch):
    """local-eval.sh sets these from the same profile the container used."""
    for key, value in VOYAGE_ENV.items():
        monkeypatch.setenv(key, value)
    cfg = factory.build_adapter(_voyage_target(),
                                verify_engine=False).effective_config()
    assert cfg["embedder"]["model"] == "voyage-4-large"


def test_conflicting_environment_is_a_hard_error(monkeypatch):
    """The whole point: a server started with bge must not be reported as voyage."""
    _clear_mem0_env(monkeypatch)
    monkeypatch.setenv("MEM0_EMBEDDER_MODEL", "BAAI/bge-small-en-v1.5")
    monkeypatch.setenv("MEM0_EMBEDDING_DIMS", "384")
    with pytest.raises(factory.DeclarationConflict) as exc:
        factory.build_adapter(_voyage_target())
    assert "MEM0_EMBEDDER_MODEL" in str(exc.value)
    assert "voyage-4-large" in str(exc.value)
    assert "BAAI/bge-small-en-v1.5" in str(exc.value)


def test_conflict_reports_every_disagreement_at_once(monkeypatch):
    _clear_mem0_env(monkeypatch)
    monkeypatch.setenv("MEM0_EMBEDDER_MODEL", "BAAI/bge-small-en-v1.5")
    monkeypatch.setenv("MEM0_EMBEDDING_DIMS", "384")
    monkeypatch.setenv("MEM0_LLM_MODEL", "gpt-4o")
    message = str(pytest.raises(factory.DeclarationConflict,
                                factory.build_adapter, _voyage_target()).value)
    assert "MEM0_EMBEDDER_MODEL" in message and "MEM0_LLM_MODEL" in message


def test_env_var_the_manifest_does_not_declare_is_ignored(monkeypatch):
    """hindsight declares no embedder, so an embedder env var cannot conflict with it."""
    monkeypatch.setenv("HINDSIGHT_EMBEDDER_MODEL", "whatever")
    monkeypatch.setenv("HINDSIGHT_LLM_MODEL", "claude-sonnet-4-5-20250929")
    monkeypatch.setenv("HINDSIGHT_LLM_PROVIDER", "anthropic")
    assert factory.build_adapter(resolve_target("hindsight")).name == "hindsight"


def test_factory_is_repeatable(monkeypatch):
    """workers>1 builds one instance per unit; the receipt uses a different instance entirely."""
    _clear_mem0_env(monkeypatch)
    target = _voyage_target()
    first = factory.build_adapter(target, verify_engine=False).effective_config()
    second = factory.build_adapter(target, verify_engine=False).effective_config()
    assert first == second
