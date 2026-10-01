"""Per-engine launch-requirement derivation tests."""
from __future__ import annotations

import pytest

from memrank.errors import MemrankError
from memrank.secrets import requirements


def test_canonical_requirements():
    # mem0: the vendor's own OpenAI pair, matching the `mem0` target. Every shipped mem0
    # manifest declares both roles and shadows this row; it is the component-less fallback.
    assert requirements.required_secrets("mem0") == ["OPENAI_API_KEY"]
    assert requirements.required_secrets("atomicmemory") == ["ANTHROPIC_API_KEY"]
    assert requirements.required_secrets("hindsight") == ["ANTHROPIC_API_KEY"]
    assert requirements.required_secrets("supermemory") == []   # keyless: see the target's graph
    assert requirements.required_secrets("word-overlap") == []                      # in-process


def test_provider_overrides_change_the_key():
    # mem0:voyage, as its manifest resolves: hosted embedder + the held-constant Anthropic LLM.
    assert requirements.required_secrets("mem0", embedder="voyage",
                                         llm="anthropic") == ["ANTHROPIC_API_KEY",
                                                              "VOYAGE_API_KEY"]
    assert requirements.required_secrets("hindsight", llm="openai") == ["OPENAI_API_KEY"]


def test_keyless_providers_contribute_nothing():
    assert requirements.required_secrets("mem0", embedder="huggingface", llm="ollama") == []


def test_an_unregistered_engine_is_not_reported_as_a_memrank_bug():
    """The first wall a target naming an unknown adapter hits.

    A bare KeyError escaped to the CLI's last branch and printed "internal error ... this is a bug in
    memrank", which is wrong and sends the reader to the wrong place.

    Still a KeyError, so lookup call sites read naturally; also a MemrankError, so it prints as one
    actionable line.
    """
    with pytest.raises(requirements.UnknownEngine) as caught:
        requirements.required_secrets("does-not-exist")

    assert isinstance(caught.value, MemrankError), "must route through the error boundary"
    assert isinstance(caught.value, KeyError), "call sites still catch KeyError"
    assert "known:" in str(caught.value), "must name what is known"
