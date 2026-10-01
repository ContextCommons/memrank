"""Manifest-declared components take precedence in the run record."""
from __future__ import annotations

from memrank.adapters import get_adapter
from memrank.adapters.word_overlap import WordOverlap


def test_undeclared_adapter_falls_back_to_env(monkeypatch):
    """The pre-M2 behaviour is unchanged when nothing declares components."""
    monkeypatch.setenv("MEM0_LLM_PROVIDER", "openai")
    monkeypatch.setenv("MEM0_LLM_MODEL", "gpt-4o-mini")
    cfg = get_adapter("mem0").effective_config()
    assert cfg["llm"] == {"provider": "openai", "model": "gpt-4o-mini"}


def test_declared_components_win_over_env(monkeypatch):
    monkeypatch.setenv("MEM0_LLM_PROVIDER", "openai")
    monkeypatch.setenv("MEM0_LLM_MODEL", "gpt-4o-mini")
    adapter = get_adapter("mem0")
    adapter.declare_components(
        llm={"provider": "anthropic", "model": "claude-sonnet-4-5-20250929"},
        embedder={"provider": "voyage", "model": "voyage-4-large", "dims": 1024})
    cfg = adapter.effective_config()
    assert cfg["llm"] == {"provider": "anthropic", "model": "claude-sonnet-4-5-20250929"}
    assert cfg["embedder"] == {"provider": "voyage", "model": "voyage-4-large", "dims": 1024}


def test_declared_transport_reaches_the_engine_block():
    adapter = get_adapter("hindsight")          # declares no transport class attribute
    assert adapter.effective_config()["engine"]["transport"] == "unknown"
    adapter.declare_components(transport="http")
    assert adapter.effective_config()["engine"]["transport"] == "http"


def test_declaring_nothing_leaves_the_record_untouched():
    adapter = WordOverlap()
    before = adapter.effective_config()
    adapter.declare_components()
    assert adapter.effective_config() == before
