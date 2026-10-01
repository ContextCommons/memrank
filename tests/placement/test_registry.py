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
"""Turning a tag into the digests it points at.

Stubbed at the HTTP boundary, because the answers here are about parsing and selection, not about
whether ghcr is up. The manifest bodies are the real shapes: an OCI index for a multi-arch image,
including the `unknown/unknown` attestation entries buildx attaches, and a bare manifest for a
single-architecture one.
"""
from __future__ import annotations

import hashlib
import json

import httpx
import pytest

from memrank.placement import registry

INDEX = {
    "mediaType": "application/vnd.oci.image.index.v1+json",
    "manifests": [
        {"digest": "sha256:amd", "platform": {"os": "linux", "architecture": "amd64"}},
        {"digest": "sha256:arm", "platform": {"os": "linux", "architecture": "arm64"}},
        # buildx attestation: not a runnable image, and matching one would pin a run to a
        # provenance blob rather than to an engine.
        {"digest": "sha256:att", "platform": {"os": "unknown", "architecture": "unknown"}},
    ],
}
MANIFEST = {"mediaType": "application/vnd.oci.image.manifest.v1+json", "config": {}, "layers": []}


def _client(body, *, status=200, on_request=None):
    """An httpx client answering every manifest request with ``body``."""
    raw = json.dumps(body).encode()

    def handler(request: httpx.Request) -> httpx.Response:
        if on_request is not None:
            on_request(request)
        if request.url.path.endswith("/token"):
            return httpx.Response(200, json={"token": "anon"})
        return httpx.Response(status, content=raw)

    return httpx.Client(transport=httpx.MockTransport(handler)), raw


def test_the_index_digest_is_computed_from_the_body():
    """Not read from `Docker-Content-Digest`: the spec says a registry SHOULD send that header and
    ECR does not send it at all. A manifest's digest is defined as the hash of its bytes."""
    client, raw = _client(INDEX)
    resolution = registry.resolve("ghcr.io/o/r:latest", platform="linux/amd64", client=client)

    assert resolution.index_digest == "sha256:" + hashlib.sha256(raw).hexdigest()


def test_the_platform_digest_is_the_entry_for_the_requested_platform():
    client, _ = _client(INDEX)
    assert registry.resolve("ghcr.io/o/r:latest", platform="linux/arm64",
                            client=client).platform_digest == "sha256:arm"
    client, _ = _client(INDEX)
    assert registry.resolve("ghcr.io/o/r:latest", platform="linux/amd64",
                            client=client).platform_digest == "sha256:amd"


def test_an_attestation_entry_is_never_selected():
    """`unknown/unknown` manifests are buildx provenance, not images."""
    client, _ = _client(INDEX)
    resolution = registry.resolve("ghcr.io/o/r:latest", platform="linux/amd64", client=client)

    assert resolution.platform_digest != "sha256:att"


def test_a_platform_the_image_does_not_publish_is_refused_naming_what_it_has():
    """Fargate does not emulate, so a missing build is a fact about the run, not a warning."""
    client, _ = _client(INDEX)
    with pytest.raises(registry.RegistryError, match="linux/riscv64"):
        registry.resolve("ghcr.io/o/r:latest", platform="linux/riscv64", client=client)


def test_a_single_architecture_image_has_no_index_digest():
    """Absent and equal are different facts: "this release has one architecture" must not read the
    same as "we did not look for others"."""
    client, raw = _client(MANIFEST)
    resolution = registry.resolve("ghcr.io/o/r:latest", platform="linux/amd64", client=client)

    assert resolution.index_digest == ""
    assert resolution.platform_digest == "sha256:" + hashlib.sha256(raw).hexdigest()


def test_the_tag_is_carried_because_the_pinned_reference_loses_it():
    """`repo@sha256:...` contains no tag, and a receipt still wants to say which pointer it
    followed. Re-deriving it from the pinned form yields nothing."""
    client, _ = _client(INDEX)
    resolution = registry.resolve("ghcr.io/o/r:latest", platform="linux/amd64", client=client)

    assert resolution.tag == "latest"
    assert resolution.pinned == "ghcr.io/o/r@sha256:amd"


def test_an_already_pinned_reference_is_refused():
    """Resolving a digest would answer a question nobody asked, and quietly suggest that a pinned
    reference is a pointer."""
    with pytest.raises(registry.RegistryError, match="already pinned"):
        registry.resolve("ghcr.io/o/r@sha256:abc", platform="linux/amd64")


def test_a_reference_with_no_tag_is_refused():
    with pytest.raises(registry.RegistryError, match="names no tag"):
        registry.resolve("ghcr.io/o/r", platform="linux/amd64")


def test_an_unknown_registry_is_refused_by_name():
    """Rather than trying unauthenticated and reporting the 401 as a missing image, which sends
    whoever meets it looking for a deleted tag."""
    with pytest.raises(registry.RegistryError, match="quay.io"):
        registry.resolve("quay.io/o/r:latest", platform="linux/amd64")


def test_ghcr_is_asked_for_an_anonymous_pull_token():
    seen: list[str] = []
    client, _ = _client(INDEX, on_request=lambda request: seen.append(str(request.url)))
    registry.resolve("ghcr.io/o/r:latest", platform="linux/amd64", client=client)

    assert any("/token" in url and "repository%3Ao%2Fr%3Apull" in url or "repository:o/r:pull" in url
               for url in seen), seen
    assert any("/v2/o/r/manifests/latest" in url for url in seen), seen


def test_a_registry_error_names_the_reference():
    """The message a run fails with must say which image, or it names nothing actionable."""
    client, _ = _client(INDEX, status=404)
    with pytest.raises(registry.RegistryError, match="ghcr.io/o/r:latest"):
        registry.resolve("ghcr.io/o/r:latest", platform="linux/amd64", client=client)
