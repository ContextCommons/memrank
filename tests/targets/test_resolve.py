"""Ref grammar, override parsing, and manifest merging."""
from __future__ import annotations

import pytest

from memrank.targets import resolve as r


def test_bare_name():
    ref = r.parse_ref("mem0")
    assert (ref.namespace, ref.name, ref.preset) == (None, "mem0", None)
    assert ref.canonical == "mem0"


def test_namespace_and_preset():
    ref = r.parse_ref("atomicstrata/mem0:voyage")
    assert (ref.namespace, ref.name, ref.preset) == ("atomicstrata", "mem0", "voyage")
    assert ref.canonical == "atomicstrata/mem0:voyage"


def test_preset_only():
    assert r.parse_ref("mem0:voyage").canonical == "mem0:voyage"


@pytest.mark.parametrize("bad", ["", "a/b/c", "mem0:", ":voyage", "/mem0", "mem0:a:b", "MEM0"])
def test_malformed_refs_are_rejected(bad):
    with pytest.raises(r.RefError):
        r.parse_ref(bad)


def test_comma_is_rejected_because_it_means_sweep():
    with pytest.raises(r.RefError, match="comma"):
        r.parse_ref("mem0,atomicmemory")


BASE = {"name": "mem0", "kind": "stack", "adapter": "mem0",
        "components": {"embedder": {"provider": "huggingface",
                                    "model": "BAAI/bge-small-en-v1.5", "dims": 384}}}


def test_parse_overrides_requires_key_equals_value():
    with pytest.raises(r.RefError, match="expected key=value"):
        r.parse_overrides(["beam"])


def test_component_shorthand_splits_on_first_slash():
    out = r.apply_overrides(BASE, {"embedder": "huggingface/BAAI/bge-small-en-v1.5",
                                   "embedder.dims": "384"})
    assert out["components"]["embedder"]["model"] == "BAAI/bge-small-en-v1.5"
    assert out["components"]["embedder"]["provider"] == "huggingface"


def test_overriding_embedder_clears_inherited_dims():
    out = r.apply_overrides(BASE, {"embedder": "voyage/voyage-4-large"})
    assert "dims" not in out["components"]["embedder"]


def test_dotted_key_applies_after_shorthand_regardless_of_order():
    overrides = {"embedder.dims": "1024", "embedder": "voyage/voyage-4-large"}
    out = r.apply_overrides(BASE, overrides)
    assert out["components"]["embedder"]["dims"] == 1024
    assert out["components"]["embedder"]["model"] == "voyage-4-large"


def test_shorthand_without_slash_is_rejected():
    with pytest.raises(r.RefError, match="provider/model"):
        r.apply_overrides(BASE, {"embedder": "voyage"})


def test_comma_in_value_is_rejected():
    with pytest.raises(r.RefError, match="comma"):
        r.apply_overrides(BASE, {"llm": "anthropic/a,b"})


def test_unknown_override_key_is_rejected():
    with pytest.raises(r.RefError, match="unknown override key 'reranker'"):
        r.apply_overrides(BASE, {"reranker": "cohere/rerank-3"})


def test_apply_overrides_does_not_mutate_input():
    r.apply_overrides(BASE, {"embedder": "voyage/voyage-4-large"})
    assert BASE["components"]["embedder"]["dims"] == 384


PARENT = {"name": "mem0", "kind": "stack", "adapter": "mem0", "depends": ["pgvector"],
          "components": {"embedder": {"provider": "huggingface", "model": "bge", "dims": 384},
                         "llm": {"provider": "anthropic", "model": "claude-sonnet-4-5"}}}


def test_merge_replaces_scalars_and_keeps_untouched_branches():
    got = r.merge(PARENT, {"from": "mem0", "name": "mem0:voyage",
                           "components": {"embedder": {"provider": "voyage",
                                                       "model": "voyage-4-large", "dims": 1024}}})
    assert got["name"] == "mem0:voyage"
    assert got["components"]["embedder"] == {"provider": "voyage",
                                             "model": "voyage-4-large", "dims": 1024}
    assert got["components"]["llm"]["provider"] == "anthropic"  # untouched branch survives
    assert got["adapter"] == "mem0"


def test_merge_drops_the_from_key():
    assert "from" not in r.merge(PARENT, {"from": "mem0", "name": "x"})


def test_merge_replaces_lists_wholesale():
    got = r.merge(PARENT, {"name": "x", "depends": ["tei"]})
    assert got["depends"] == ["tei"]


def test_merge_does_not_mutate_parent():
    r.merge(PARENT, {"name": "x", "components": {"embedder": {"dims": 99}}})
    assert PARENT["components"]["embedder"]["dims"] == 384


def test_binding_root_is_an_override_key_so_a_second_checkout_needs_no_link():
    """Comparing two commits should not require registering each as its own target first."""
    data = {"binding": {"kind": "source", "rootFrom": "link"}}
    got = r.apply_overrides(data, {"binding.root": "/checkouts/commit-b"})

    assert got["binding"]["root"] == "/checkouts/commit-b"
    # The link steps aside rather than tripping the both-declared refusal.
    assert "rootFrom" not in got["binding"]


def test_an_unknown_override_key_still_names_what_is_accepted():
    with pytest.raises(r.RefError, match="binding.root"):
        r.apply_overrides({}, {"binding.nonsense": "x"})
