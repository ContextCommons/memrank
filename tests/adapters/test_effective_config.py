"""Tests for MemoryAdapter.effective_config() -- the per-run component capture."""
from __future__ import annotations

from memrank.adapters import get_adapter
from memrank.adapters.word_overlap import WordOverlap


def test_baseline_effective_config_shape():
    cfg = WordOverlap().effective_config()
    assert cfg["engine"]["name"] == "word-overlap"
    assert cfg["engine"]["transport"] == "in-process"
    # In-process baseline has no external LLM/embedder.
    assert cfg["llm"] == {"provider": None, "model": None}
    assert cfg["embedder"] == {"model": None, "dims": None}


def test_http_adapter_reads_operator_declared_llm(monkeypatch):
    monkeypatch.setenv("MEM0_LLM_PROVIDER", "openai")
    monkeypatch.setenv("MEM0_LLM_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("MEM0_EMBEDDER_PROVIDER", "openai")
    monkeypatch.setenv("MEM0_EMBEDDER_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("MEM0_EMBEDDING_DIMS", "1536")
    cfg = get_adapter("mem0").effective_config()  # construction does not connect
    assert cfg["engine"]["name"] == "mem0"
    assert cfg["llm"] == {"provider": "openai", "model": "gpt-4o-mini"}
    assert cfg["embedder"] == {"provider": "openai", "model": "text-embedding-3-small",
                               "dims": 1536}


def test_http_adapter_defaults_to_none_when_undeclared(monkeypatch):
    for var in ("ATOMICMEMORY_LLM_PROVIDER", "ATOMICMEMORY_LLM_MODEL",
                "ATOMICMEMORY_EMBEDDER_PROVIDER", "ATOMICMEMORY_EMBEDDER_MODEL",
                "ATOMICMEMORY_EMBEDDING_DIMS"):
        monkeypatch.delenv(var, raising=False)
    cfg = get_adapter("atomicmemory").effective_config()
    assert cfg["llm"] == {"provider": None, "model": None}
    assert cfg["embedder"] == {"provider": None, "model": None, "dims": None}
