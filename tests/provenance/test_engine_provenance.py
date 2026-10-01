"""Tests for the engine-provenance facet (memrank/provenance/engine.py).

Locks the standards-shaped representation: purl identity (digest = version),
CycloneDX pedigree for a fork, and origin *derived* from pedigree +
manufacturer/supplier rather than a bespoke enum.
"""
from __future__ import annotations

from memrank.provenance.engine import PROVENANCE, build_provenance, derive_origin

_MEM0_PINS = {
    "MEM0_ENGINE_IMAGE_DIGEST": "sha256:9f3c",
    "MEM0_ENGINE_IMAGE_REPO": "registry.example.test/bench_engines",
    "MEM0_ENGINE_IMAGE_TAG": "mem0-server-2414721",
    "MEM0_ENGINE_SOURCE_SHA": "2414721abc",
    "MEM0_ENGINE_SOURCE_DESCRIBE": "v2.0.1-10-g2414721",
    "MEM0_ENGINE_SOURCE_DIRTY": "false",
}


def test_mem0_is_a_fork_with_pedigree():
    prov = build_provenance("mem0", env=_MEM0_PINS)
    assert derive_origin(prov) == "fork"
    assert prov["pedigree"]["ancestors"] == ["pkg:github/mem0ai/mem0@v2.0.1"]
    assert prov["pedigree"]["patches"][0]["count"] == 10
    assert prov["manufacturer"] == prov["supplier"] == "atomicstrata"


def test_mem0_purl_is_oci_digest_pinned():
    prov = build_provenance("mem0", env=_MEM0_PINS)
    assert prov["purl"].startswith("pkg:oci/mem0-server@sha256:9f3c")
    assert "repository_url=registry.example.test/bench_engines" in prov["purl"]
    assert "tag=mem0-server-2414721" in prov["purl"]
    assert prov["generatedFrom"] == "pkg:generic/mem0@2414721abc"
    assert prov["properties"]["memrank:source_describe"] == "v2.0.1-10-g2414721"


def test_hindsight_is_a_mirror_no_pedigree():
    prov = build_provenance("hindsight", env={})
    assert derive_origin(prov) == "mirror"
    assert prov["pedigree"] is None
    assert prov["manufacturer"] == "vectorize-io" != prov["supplier"]


def test_atomicmemory_is_our_source():
    assert derive_origin(build_provenance("atomicmemory", env={})) == "our-source"


def test_supermemory_is_build_in_repo_with_binary_purl():
    prov = build_provenance("supermemory", env={"SUPERMEMORY_ENGINE_BINARY_VERSION": "0.0.3"})
    assert derive_origin(prov) == "build-in-repo"
    assert prov["properties"]["memrank:binary_purl"] == "pkg:generic/supermemory-server@0.0.3"


def test_baseline_is_in_process():
    assert derive_origin(build_provenance("word-overlap", env={})) == "in-process"


def test_local_run_falls_back_to_source_purl():
    # No image digest: identity is the source commit (still a real pin -> declared).
    prov = build_provenance("mem0", env={"MEM0_ENGINE_SOURCE_SHA": "deadbeef"})
    assert prov["purl"] == "pkg:generic/mem0@deadbeef"
    assert prov["declared"] is True


def test_declared_true_once_digest_injected():
    assert build_provenance("mem0", env=_MEM0_PINS)["declared"] is True


def test_static_only_run_is_not_declared():
    # No dynamic pins injected at all: manifest defaults only, no purl, declared False.
    prov = build_provenance("mem0", env={})
    assert prov["declared"] is False
    assert prov["purl"] is None


def test_unknown_adapter_returns_empty():
    assert build_provenance("nope", env={}) == {}


_NATIVE_SOURCE = {
    "NATIVE_ENGINE_SOURCE_SHA": "abc123def456",
    "NATIVE_ENGINE_SOURCE_DIRTY": "true",
    "NATIVE_ENGINE_SOURCE_DELTA_SHA256": "f29156",
    "NATIVE_ENGINE_EXECUTION_BINDING": "workspace",
}


def test_a_repo_less_engine_still_records_its_commit():
    """A translator wraps a third party's engine, so there is no source_repo to build a purl from
    -- and without this the receipt carried a dirty flag and a delta hash against a commit it never
    named, which identifies nothing: a delta is only meaningful relative to a stated base."""
    prov = build_provenance("native", env=_NATIVE_SOURCE)

    assert prov["purl"] is None            # memrank cannot name the artifact, and does not pretend
    assert prov["generatedFrom"] is None
    assert prov["properties"]["memrank:source_sha"] == "abc123def456"
    assert prov["properties"]["memrank:source_dirty"] == "true"


def test_an_engine_with_a_source_purl_does_not_duplicate_its_commit():
    """The SHA already rides in generatedFrom there; a second copy is one more thing to drift."""
    prov = build_provenance("atomicmemory", env={"ATOMICMEMORY_ENGINE_SOURCE_SHA": "deadbeef"})

    assert prov["generatedFrom"] == "pkg:generic/atomicmemory@deadbeef"
    assert "memrank:source_sha" not in prov["properties"]


def test_a_non_public_source_is_pinned_without_being_named():
    """A receipt is read by people who cannot clone our repositories.

    Naming one identifies nothing they can fetch and discloses where our code lives, so the
    commit is carried by a locationless `pkg:generic` purl and the absence is explained. What
    makes a patched engine reproducible is the pedigree, and that is public: it survives.
    """
    prov = build_provenance("mem0", env=_MEM0_PINS)

    assert prov["properties"]["memrank:source_visibility"] == "private"
    assert prov["generatedFrom"] == "pkg:generic/mem0@2414721abc"      # commit still pins the tree
    assert prov["pedigree"]["ancestors"] == ["pkg:github/mem0ai/mem0@v2.0.1"]
    assert prov["pedigree"]["patches"][0]["resolves"] == "voyage-embedder"
    assert derive_origin(prov) == "fork"                              # lineage still derivable


def test_no_manifest_entry_names_a_non_public_repository():
    """The guard the manifest is: a private name cannot reach a receipt if it is never in here."""
    for name, facts in PROVENANCE.items():
        if facts.get("source_visibility") == "private":
            assert facts["source_repo"] is None, f"{name} names a repository it declares private"
            assert facts["source_name"], f"{name} has no public name to pin its commit against"
        assert "-internal" not in repr(facts), f"{name} carries an internal repository reference"


def test_every_registered_engine_has_a_manifest():
    for name in ("mem0", "hindsight", "atomicmemory", "supermemory", "word-overlap"):
        assert name in PROVENANCE
