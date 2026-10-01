"""Credentials wallet round-trip + resolver/preflight tests."""
from __future__ import annotations

import stat

import pytest

from memrank import config
from memrank.config import ConfigError


def test_wallet_roundtrip_and_perms(wallet):
    wallet.put("OPENAI_API_KEY", "sk-abc")
    assert wallet.get("OPENAI_API_KEY") == "sk-abc"
    assert wallet.names() == ["OPENAI_API_KEY"]
    assert wallet.delete("OPENAI_API_KEY") is True
    assert wallet.get("OPENAI_API_KEY") is None
    wallet.put("ANTHROPIC_API_KEY", "sk-x")
    assert stat.S_IMODE(wallet.store_path().stat().st_mode) == 0o600


def test_secret_env_wins_over_pool(wallet, monkeypatch):
    wallet.put("OPENAI_API_KEY", "from-pool")
    monkeypatch.setenv("OPENAI_API_KEY", "from-env")
    assert config.secret("OPENAI_API_KEY") == "from-env"          # env wins


def test_secret_from_pool_and_ref(wallet, monkeypatch):
    wallet.put("VOYAGE_API_KEY", "pa-literal")
    wallet.put("ANTHROPIC_API_KEY", "${MY_REF_KEY}")
    monkeypatch.setenv("MY_REF_KEY", "resolved-ref")
    assert config.secret("VOYAGE_API_KEY") == "pa-literal"
    assert config.secret("ANTHROPIC_API_KEY") == "resolved-ref"   # ${VAR} resolved
    assert config.secret("OPENAI_API_KEY") is None                # neither env nor pool


def test_preflight_lists_all_missing(wallet):
    # The component overrides are how a target's manifest reaches this call (mem0:bge-tei's pair:
    # keyless TEI embedder + the held-constant Anthropic LLM). Bare "mem0" is the vendor's own
    # OpenAI pair, so passing the overrides is what keeps this a test of TWO different keys.
    mem0_bge = {"embedder": "huggingface", "llm": "anthropic"}
    assert config.missing_requirements("mem0", **mem0_bge) == ["ANTHROPIC_API_KEY"]
    with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY"):
        config.preflight("mem0", **mem0_bge)
    wallet.put("ANTHROPIC_API_KEY", "sk-x")
    config.preflight("mem0", **mem0_bge)                           # now satisfied -> no raise


def test_resolved_secret_env_injects_even_when_the_name_is_in_our_own_environment(
        wallet, monkeypatch):
    """A container inherits nothing, so "already set here" is not a reason to omit it.

    This asserted the opposite until it was caught by a live run: because `.env` carries the same
    credentials, anything that loaded it first (`engines_registry()`, which `placement_for()` calls
    before provisioning) made every required secret look present, and `--on local` launched engines
    with no credentials. The engine died at startup and the placement reported only "connection
    refused" after a full readiness budget.
    """
    mem0_bge = {"embedder": "huggingface", "llm": "anthropic"}   # mem0:bge-tei's resolved pair
    wallet.put("ANTHROPIC_API_KEY", "sk-pool")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-env")
    assert config.resolved_secret_env("mem0", **mem0_bge) == {"ANTHROPIC_API_KEY": "sk-env"}
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    assert config.resolved_secret_env("mem0", **mem0_bge) == {"ANTHROPIC_API_KEY": "sk-pool"}
