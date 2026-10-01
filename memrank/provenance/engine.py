# Copyright 2026 AtomicStrata
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.
"""Engine provenance facet -- which exact engine artifact ran and how it derives
from upstream.

Every receipt records *config* but, until now, only a weak engine identity
(``engine_version``, ``"unknown"`` locally, and no image digest). This module
builds a standards-shaped provenance block so a run pins the exact engine build
and its lineage. The representation is deliberately NOT bespoke -- it mirrors what
SBOM tools (Syft, Trivy) emit:

- **Identity/location = Package URL (purl, ECMA-427).** For a container the image
  *digest is the version* (``pkg:oci/<name>@sha256:<digest>?repository_url=...&tag=...``)
  -- the immutable, registry-agnostic pin. Source built from a public repository is its own
  ``pkg:github/...@<sha>`` purl; a closed binary is ``pkg:generic/...``.
- **A source repository that is not public is never named.** A receipt is read by people who
  cannot clone it, and a name they cannot resolve identifies nothing while disclosing where our
  code lives. Such a source becomes ``pkg:generic/<name>@<commit>`` with no location qualifier,
  plus ``memrank:source_visibility=private``: the commit still pins the exact tree, and the
  lineage that makes a patched engine reproducible -- the upstream ancestor purl and the patch
  set in ``pedigree`` -- is public and survives intact.
- **Derivation = CycloneDX ``pedigree`` + ``manufacturer``/``supplier``.** A fork
  carries ``pedigree.ancestors`` (the upstream purl) + ``patches``; "our fork vs a
  vendor mirror" is the manufacturer (who built the image) vs supplier (who
  distributes it) distinction. So fork/mirror/our-source is *derived*
  (:func:`derive_origin`), never a hand-rolled enum.
- **The one non-standard axis -- deployment model** (self-hosted OSS vs managed API
  vs embedded SDK) has no standard enum anywhere, so it lives in CycloneDX
  ``properties`` under a ``memrank:`` namespace, explicitly custom.

Following memrank's "operator-declared, not introspected" model, stable facts live
in the code-reviewed :data:`PROVENANCE` manifest and volatile pins (image digest,
source commit, dirty flag) are injected at run time via ``{prefix}ENGINE_*`` env
vars by the build/orchestration scripts, then merged by :func:`build_provenance`.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

# Static, code-reviewed provenance facts per adapter. `prefix` is the engine's
# env-var namespace; `None` marks an in-process adapter with no container. Dynamic
# pins are read from `{prefix}ENGINE_*` and merged in build_provenance().
PROVENANCE: dict[str, dict[str, Any]] = {
    "mem0": {
        "prefix": "MEM0_",
        "type": "container",
        "image_name": "mem0-server",
        "manufacturer": "atomicstrata",  # we build the forked image
        "supplier": "atomicstrata",
        # The fork is not public, so no repository is named: see `source_visibility`.
        "source_repo": None,
        "source_visibility": "private",
        "source_name": "mem0",
        # Fork lineage: our image = upstream base + a tracked patch set. "fork" is
        # CycloneDX's own example for pedigree.ancestors. The ancestor purl and the patch
        # set are what make the lineage checkable, and both are public facts about a public
        # upstream -- they survive unchanged. Only the `diff.url` is gone: it addressed a
        # file inside the non-public fork.
        "ancestors": ["pkg:github/mem0ai/mem0@v2.0.1"],
        "patches": [{
            "type": "unofficial",
            "resolves": "voyage-embedder",
            "count": 10,
        }],
        "properties": {
            "memrank:distribution": "self-hosted-oss",
            "memrank:mem0_config_version": "v1.1",  # MemoryConfig.version -- behavior selector, NOT a release
            "memrank:api_version": "none",
        },
    },
    "hindsight": {
        "prefix": "HINDSIGHT_",
        "type": "container",
        "image_name": "hindsight",
        "manufacturer": "vectorize-io",  # vendor builds the image; we only mirror it
        "supplier": "atomicstrata",
        "source_repo": None,  # unmodified mirror -- we do not build from source
        "ancestors": [],
        "patches": [],
        "properties": {
            "memrank:distribution": "self-hosted-oss",
            "memrank:api_version": "none",
            "memrank:upstream": "pkg:oci/hindsight?repository_url=ghcr.io/vectorize-io/hindsight",
        },
    },
    "atomicmemory": {
        "prefix": "ATOMICMEMORY_",
        "type": "container",
        "image_name": "atomicmemory-core",
        "manufacturer": "atomicstrata",  # our own source
        "supplier": "atomicstrata",
        "source_repo": None,  # not public: see `source_visibility`
        "source_visibility": "private",
        "source_name": "atomicmemory",
        "ancestors": [],
        "patches": [],
        "properties": {"memrank:distribution": "self-hosted-oss"},
    },
    "supermemory": {
        "prefix": "SUPERMEMORY_",
        "type": "container",
        "image_name": "supermemory",
        "manufacturer": "atomicstrata",  # we build the wrapper image in-repo
        "supplier": "atomicstrata",
        "source_repo": None,  # image built in-repo wrapping a closed vendor binary
        "ancestors": [],
        "patches": [],
        "properties": {
            "memrank:distribution": "n/a",  # closed binary; not a normal OSS/SDK/API line
            "memrank:api_version": "none",
            # Nested binary component (its own manufacturer is the vendor).
            "memrank:binary_name": "supermemory-server",
        },
    },
    # A translator, not an engine. memrank cannot name the system behind it: there is no image to
    # digest and no source repository it knows of, so `_component_purl` and `generatedFrom` both
    # resolve to None rather than to a purl asserting an artifact nobody verified. What the run
    # CAN identify -- the translator's own commit, working-tree delta and launch command -- the
    # workspace placement injects as properties. manufacturer/supplier are "unknown" because they
    # honestly are; claiming atomicstrata built a third party's engine would be false.
    "native": {
        "prefix": "NATIVE_",
        "type": "application",
        "image_name": None,
        "manufacturer": "unknown",
        "supplier": "unknown",
        "source_repo": None,
        "ancestors": [],
        "patches": [],
        "properties": {"memrank:distribution": "translator"},
    },
    "word-overlap": {
        "prefix": None,
        "type": "application",  # runs in-process; no container image
        "image_name": None,
        "manufacturer": "atomicstrata",
        "supplier": "atomicstrata",
        "source_repo": "ContextCommons/memrank",
        "ancestors": [],
        "patches": [],
        "properties": {"memrank:distribution": "in-process"},
    },
}


def _env(env: Mapping[str, str], prefix: str | None, suffix: str) -> str | None:
    """Read ``{prefix}{suffix}`` from ``env``; ``None`` for empty or prefix-less."""
    if not prefix:
        return None
    return env.get(f"{prefix}{suffix}") or None


def _oci_purl(image_name: str, digest: str, repo: str | None, tag: str | None,
              arch: str | None = None) -> str:
    """Assemble ``pkg:oci/<name>@<digest>`` with repository_url/tag/arch qualifiers.

    The digest is the purl version, and it is the PLATFORM digest -- the binary that executed, not
    the multi-arch index that contains it. ``arch`` is purl's own qualifier for exactly this, so
    two rows sharing a release and differing in architecture are distinguishable here rather than
    only in a property. The index digest itself lives in ``properties``: it identifies the release
    rather than the artifact, and a purl names one artifact.

    Colon is kept unencoded (the form Syft/Trivy emit in JSON), for human readability.
    """
    quals = [f"{key}={value}" for key, value in
             (("repository_url", repo), ("tag", tag), ("arch", arch)) if value]
    base = f"pkg:oci/{image_name}@{digest}"
    return f"{base}?{'&'.join(quals)}" if quals else base


def _source_purl(facts: dict[str, Any], sha: str | None) -> str | None:
    """The purl of the source we built from, or ``None`` when there is no source to name.

    A public repository gets ``pkg:github/<org>/<repo>@<commit>``: a reader can fetch exactly
    that tree. A repository that is not public gets ``pkg:generic/<name>@<commit>`` instead --
    purl's own type for an artifact with no public package repository -- carrying the commit,
    which is the part that pins the build, and no location qualifier, because there is no
    location an outsider could resolve. Naming an unreachable repository would assert a
    lineage the reader cannot check while telling them nothing they can act on; the commit
    plus :func:`_pedigree` is what stays checkable.
    """
    if not sha:
        return None
    source_repo = facts.get("source_repo")
    if source_repo:
        return f"pkg:github/{source_repo}@{sha}"
    if facts.get("source_visibility") == "private" and facts.get("source_name"):
        return f"pkg:generic/{facts['source_name']}@{sha}"
    return None


def _component_purl(facts: dict[str, Any], env: Mapping[str, str]) -> str | None:
    """The purl of what actually ran: the OCI image (digest = version) when a digest
    was injected, else the source-commit purl for a bare local run (or ``None``)."""
    prefix = facts["prefix"]
    digest = _env(env, prefix, "ENGINE_IMAGE_DIGEST")
    if digest and facts["image_name"]:
        repo = _env(env, prefix, "ENGINE_IMAGE_REPO")
        tag = _env(env, prefix, "ENGINE_IMAGE_TAG") or _env(env, prefix, "ENGINE_VERSION")
        platform = _env(env, prefix, "ENGINE_IMAGE_PLATFORM")
        return _oci_purl(facts["image_name"], digest, repo, tag,
                         arch=platform.rpartition("/")[2] if platform else None)
    return _source_purl(facts, _env(env, prefix, "ENGINE_SOURCE_SHA"))


def _pedigree(facts: dict[str, Any]) -> dict[str, Any] | None:
    """CycloneDX pedigree (ancestors + patches) for a fork, else ``None``."""
    if not facts["ancestors"] and not facts["patches"]:
        return None
    return {"ancestors": list(facts["ancestors"]),
            "patches": [dict(patch) for patch in facts["patches"]]}


def _properties(facts: dict[str, Any], env: Mapping[str, str]) -> dict[str, str]:
    """Static custom properties plus any injected source/binary pins."""
    props = dict(facts["properties"])
    prefix = facts["prefix"]
    # Why the source purl is a `pkg:generic` with no location, stated rather than left for the
    # reader to infer from an absence. Without it "no repository here" and "a repository we
    # forgot to record" look identical.
    visibility = facts.get("source_visibility")
    if visibility:
        props["memrank:source_visibility"] = visibility
    describe = _env(env, prefix, "ENGINE_SOURCE_DESCRIBE")
    if describe:
        props["memrank:source_describe"] = describe
    # The commit, when no standard field can carry it. A SHA normally rides in the purl or in
    # `generatedFrom`, but both are built by `_source_purl`, which needs a `source_repo` -- and a
    # translator wraps a third party's engine whose repository memrank does not know. Without this
    # the receipt recorded a dirty flag and a delta hash against a commit it never named, which
    # identifies nothing: the delta is only meaningful relative to a stated base.
    source_sha = _env(env, prefix, "ENGINE_SOURCE_SHA")
    if source_sha and not _source_purl(facts, source_sha):
        props["memrank:source_sha"] = source_sha
    dirty = _env(env, prefix, "ENGINE_SOURCE_DIRTY")
    if dirty is not None:
        props["memrank:source_dirty"] = dirty
    delta = _env(env, prefix, "ENGINE_SOURCE_DELTA_SHA256")
    if delta:
        props["memrank:source_delta_sha256"] = delta
    binding = _env(env, prefix, "ENGINE_EXECUTION_BINDING")
    if binding:
        props["memrank:execution_binding"] = binding
    # Stated only when true, and true only when there was no repository to read. Its absence beside
    # a source_sha is the normal case; its presence is why no SHA appears.
    unversioned = _env(env, prefix, "ENGINE_WORKSPACE_UNVERSIONED")
    if unversioned:
        props["memrank:workspace_unversioned"] = unversioned
    launcher = _env(env, prefix, "ENGINE_LAUNCHER_PROFILE_SHA256")
    if launcher:
        props["memrank:launcher_profile_sha256"] = launcher
    command = _env(env, prefix, "ENGINE_COMMAND_SHA256")
    if command:
        props["memrank:command_sha256"] = command
    binary_version = _env(env, prefix, "ENGINE_BINARY_VERSION")
    binary_name = props.get("memrank:binary_name")
    if binary_version and binary_name:
        props["memrank:binary_purl"] = f"pkg:generic/{binary_name}@{binary_version}"
    # The RELEASE, where the purl names the artifact. Two runs of one multi-arch tag on different
    # architectures share this and differ in their purl -- which is what lets a reader tell "same
    # release, different binary" from "different build", the distinction a single digest could not
    # express.
    index_digest = _env(env, prefix, "ENGINE_IMAGE_INDEX_DIGEST")
    if index_digest:
        props["memrank:index_digest"] = index_digest
    platform = _env(env, prefix, "ENGINE_IMAGE_PLATFORM")
    if platform:
        props["memrank:platform"] = platform
    return props


def _declared(facts: dict[str, Any], env: Mapping[str, str]) -> bool:
    """True when dynamic pins (image digest or source commit) were injected."""
    prefix = facts["prefix"]
    return bool(_env(env, prefix, "ENGINE_IMAGE_DIGEST") or _env(env, prefix, "ENGINE_SOURCE_SHA"))


def build_provenance(adapter_name: str, env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Build the engine-provenance facet for ``adapter_name``.

    Merges the static :data:`PROVENANCE` manifest with dynamic pins from ``env``
    (defaults to ``os.environ``). Returns ``{}`` for an unregistered adapter so the
    receipt never fails on an unknown engine.
    """
    env = os.environ if env is None else env
    facts = PROVENANCE.get(adapter_name)
    if facts is None:
        return {}
    prefix = facts["prefix"]
    properties = _properties(facts, env)
    component_type = ("application" if properties.get("memrank:execution_binding") == "workspace"
                      else facts["type"])
    return {
        "purl": _component_purl(facts, env),
        "type": component_type,
        "manufacturer": facts["manufacturer"],
        "supplier": facts["supplier"],
        "pedigree": _pedigree(facts),
        "generatedFrom": _source_purl(facts, _env(env, prefix, "ENGINE_SOURCE_SHA")),
        "properties": properties,
        "declared": _declared(facts, env),
    }


def derive_origin(prov: dict[str, Any] | None) -> str:
    """Classify an engine's origin from its provenance facet alone (for filtering).

    Derived from standard fields, never stored as a bespoke enum: ``fork`` (has
    pedigree), ``in-process`` (application type), ``build-in-repo`` (wraps a vendor
    binary), ``mirror`` (image manufacturer != our supplier), else ``our-source``.
    """
    if not prov:
        return "unknown"
    if (prov.get("properties") or {}).get("memrank:execution_binding") == "workspace":
        return "our-source"
    if prov.get("type") == "application":
        return "in-process"
    if prov.get("pedigree"):
        return "fork"
    if "memrank:binary_name" in (prov.get("properties") or {}):
        return "build-in-repo"
    if prov.get("manufacturer") != prov.get("supplier"):
        return "mirror"
    return "our-source"
