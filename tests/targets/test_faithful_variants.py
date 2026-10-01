"""Faithful variants declare a vendor's own depth, and that depth must reach the engine.

Matched mode holds every target to one budget; a faithful variant exists to answer a different
question -- "can we land near what the vendor published" -- and so must be able to state the
vendor's retrieval settings and run uncapped.

Two things were silently broken before these existed: a manifest could declare
``context_budget: uncapped`` and be capped anyway (the runner reads the attribute off the ADAPTER,
which only the in-process arms set), and there was no way at all to state an engine-specific
retrieval depth outside an adapter constant.

A faithful target used to be recognizable by its ``:suffix``. That inverted for mem0 on
2026-08-18 and for hindsight on 2026-08-19, and is now the general rule: a BARE ref names the
configuration its owner ships, so for a vendor's engine it is the vendor's and therefore faithful.
The mode lives in ``context_budget`` and nowhere else -- never in the ref.

Which makes the matched arms the fragile ones. ``hindsight:matched`` is comparable only because it
CLEARS the retrieval and ingest blocks it would otherwise inherit; several tests below exist to
fail if those lines are ever deleted as redundant.
"""

from __future__ import annotations

import pytest

from memrank.targets import resolve_target
from memrank.targets.factory import build_adapter
from memrank.targets.manifest import ManifestError, from_dict
from tests import withheld


def test_a_faithful_variant_declares_the_vendors_components():
    """mem0's own suite: gpt-4o-mini extraction, text-embedding-3-small at 1536d."""
    withheld.require("mem0")
    t = resolve_target("mem0")
    assert t.components["llm"].model == "gpt-4o-mini"
    assert t.components["embedder"].model == "text-embedding-3-small"
    assert t.components["embedder"].dims == 1536


def test_faithful_variants_are_uncapped():
    """The point of the mode: reproduce the vendor's depth, not memrank's fairness cap."""
    assert resolve_target("hindsight").context_budget == "uncapped"
    withheld.require("mem0")
    assert resolve_target("mem0").context_budget == "uncapped"


def test_a_matched_target_stays_matched():
    """A bare ref is the vendor's configuration, so the matched arm always carries a suffix.

    For hindsight that arm is `hindsight:matched`, shipped here. For mem0 there is no matched ref
    in the package at all: both moved to the research lane on 2026-08-19. So every BARE vendor ref
    memrank ships is uncapped, and that is the fact this asserts."""
    assert resolve_target("hindsight:matched").context_budget == "matched"
    assert resolve_target("hindsight").context_budget == "uncapped"
    withheld.require("mem0")
    assert resolve_target("mem0").context_budget == "uncapped"


def test_context_budget_reaches_the_adapter_instance():
    """Declared-but-ignored was the bug: the runner reads this off the adapter, and the factory
    never applied the manifest's value to a stack target."""
    adapter = build_adapter(resolve_target("hindsight"), verify_engine=False)
    assert adapter.context_budget == "uncapped"
    matched = build_adapter(resolve_target("hindsight:matched"), verify_engine=False)
    assert matched.context_budget == "matched"


def test_hindsight_carries_the_published_retrieval_depth():
    """Transcribed from the vendor's own harness (AMB): budget=high, 32768 facts, 16384 chunks."""
    t = resolve_target("hindsight")
    assert t.retrieval == {"budget": "high", "max_tokens": 32768, "chunk_max_tokens": 16384}


def test_the_retrieval_block_reaches_the_recall_payload():
    """A declared depth that never leaves the manifest would be a mislabelled row."""
    adapter = build_adapter(resolve_target("hindsight"), verify_engine=False)
    assert adapter.retrieval["budget"] == "high"
    assert adapter.retrieval["max_tokens"] == 32768


def test_a_matched_hindsight_sends_no_invented_budget_tier():
    """Unset means the engine's own default tier; sending one we chose would misreport the run.

    This is one of the two tests guarding `hindsight:matched`'s bare `retrieval:` key. Since the
    base became AMB's configuration, deleting that line would send budget=high/32768 on the
    leaderboard row -- the vendor's depth under a matched name."""
    adapter = build_adapter(resolve_target("hindsight:matched"), verify_engine=False)
    assert adapter.retrieval == {}


def test_retrieval_must_be_a_mapping():
    """A scalar here would reach an adapter as nothing, far from the manifest that caused it."""
    with pytest.raises(ManifestError, match="retrieval must be a mapping"):
        from_dict({"name": "x", "kind": "stack", "adapter": "hindsight", "retrieval": "high"})


def test_hindsight_disables_observations_the_way_amb_does():
    """AMB creates banks with observations off; hindsight ships them on (audit F6)."""
    assert resolve_target("hindsight").ingest == {"enable_observations": False}


def test_the_ingest_block_reaches_the_adapter():
    """A declared setting that never leaves the manifest is a target lying about its own run."""
    adapter = build_adapter(resolve_target("hindsight"), verify_engine=False)
    assert adapter.ingest_settings["enable_observations"] is False


def test_a_matched_hindsight_keeps_the_vendors_own_default():
    """Matched mode measures the product as shipped, which is observations ON.

    The second guard on `hindsight:matched`'s cleared blocks: without the bare `ingest:` key this
    would inherit AMB's observations-OFF, and the ranked row would measure a bank missing memory
    networks the shipped product writes."""
    assert resolve_target("hindsight:matched").ingest == {}
    assert build_adapter(resolve_target("hindsight:matched"),
                         verify_engine=False).ingest_settings == {}


def test_ingest_must_be_a_mapping():
    with pytest.raises(ManifestError, match="ingest must be a mapping"):
        from_dict({"name": "x", "kind": "stack", "adapter": "hindsight", "ingest": "off"})


def test_an_ingest_setting_is_recorded_in_what_the_receipt_hashes():
    """Two runs differing in which memory networks exist must not hash identically (F16)."""
    from memrank.targets.manifest import to_dict
    assert to_dict(resolve_target("hindsight"))["ingest"] == {"enable_observations": False}
    assert "ingest" not in to_dict(resolve_target("hindsight:matched"))
