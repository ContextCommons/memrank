"""The engine confirms (or contradicts) what the manifest claims about it."""
from __future__ import annotations

import pytest

from memrank.targets.factory import DeclarationConflict, check_engine
from memrank.targets.manifest import from_dict

# What the real mem0 server reports for the hosted-embedder stack.
VOYAGE_REPORT = {
    "llm": {"provider": "anthropic", "model": "claude-sonnet-4-5-20250929"},
    "embedder": {"provider": "voyage", "model": "voyage-4-large", "dims": 1024},
}
# What it reports for the TEI stack -- the shape captured from a live GET /configure.
BGE_REPORT = {
    "llm": {"provider": "anthropic", "model": "claude-sonnet-4-5-20250929"},
    "embedder": {"provider": "huggingface", "model": "BAAI/bge-small-en-v1.5", "dims": 384},
}


def _target(name, embedder):
    """The manifest under test, built rather than resolved from the catalog.

    These two configurations were `mem0:voyage` and `mem0:bge-tei` until 2026-08-19, when both
    moved to the research lane. What this module checks is `check_engine`'s comparison -- a shipped
    ref was only ever a convenient carrier for two component sets that differ in every field, and
    building them here keeps the mislabel scenario exact without depending on another repo.
    """
    return from_dict({"name": name, "kind": "stack", "adapter": "mem0", "transport": "http",
                      "components": {"llm": {"provider": "anthropic",
                                             "model": "claude-sonnet-4-5-20250929"},
                                     "embedder": embedder}})


VOYAGE_TARGET = _target("mem0:hosted", VOYAGE_REPORT["embedder"])
BGE_TARGET = _target("mem0:self-hosted", BGE_REPORT["embedder"])


class _Engine:
    """A stand-in adapter that reports whatever the test tells it to."""

    def __init__(self, report):
        self._report = report

    def describe_engine(self):
        return self._report


def test_engine_that_agrees_is_reported_as_verified():
    assert check_engine(_Engine(VOYAGE_REPORT), VOYAGE_TARGET) == "engine"


def test_engine_that_cannot_be_asked_is_reported_as_declared():
    assert check_engine(_Engine(None), VOYAGE_TARGET) == "declared"


def test_the_exact_mislabel_this_milestone_exists_to_catch():
    """A bge-small server evaluated under a target claiming voyage -- a real run did this."""
    with pytest.raises(DeclarationConflict) as exc:
        check_engine(_Engine(BGE_REPORT), VOYAGE_TARGET)
    message = str(exc.value)
    assert "embedder.model" in message
    assert "voyage-4-large" in message and "BAAI/bge-small-en-v1.5" in message
    assert "embedder.dims" in message


def test_matching_target_against_the_same_engine_passes():
    assert check_engine(_Engine(BGE_REPORT), BGE_TARGET) == "engine"


def test_every_disagreement_is_reported_at_once():
    report = {"llm": {"provider": "openai", "model": "gpt-4o"},
              "embedder": {"provider": "voyage", "model": "voyage-3", "dims": 512}}
    message = str(pytest.raises(DeclarationConflict, check_engine,
                                _Engine(report), VOYAGE_TARGET).value)
    for expected in ("llm.provider", "llm.model", "embedder.model", "embedder.dims"):
        assert expected in message


def test_fields_the_engine_omits_are_not_compared():
    """A partial report must not be read as a contradiction."""
    assert check_engine(_Engine({"embedder": {"model": "voyage-4-large"}}),
                        VOYAGE_TARGET) == "engine"


def test_undeclared_component_cannot_conflict():
    """A knob no manifest states cannot contradict the engine -- there is nothing to compare.

    Built here rather than resolved from the catalog: `mem0` was the worked example until it
    declared mem0's own embedder (2026-08-18), and no builtin target leaves a knob blank now. The
    shape still reaches this code through operator manifests on `targets.path`."""
    llm_only = from_dict({"name": "llm-only", "kind": "stack", "adapter": "mem0",
                          "transport": "http", "abstract": True,
                          "components": {"llm": {"provider": "anthropic",
                                                 "model": "claude-sonnet-4-5-20250929"}}})
    assert check_engine(_Engine(BGE_REPORT), llm_only) == "engine"
