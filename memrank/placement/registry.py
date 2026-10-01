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
"""Turn a tag into the digests it points at, once, over the registry's own HTTP API.

A tag is a mutable pointer, so something has to ask "what does `:latest` mean right now". Until
this module that question was asked TWICE per target and by two different things -- `docker compose
--pull always` locally, and ECS at container start in the cloud. Two answers to one question is how
a sweep can straddle a tag that moved between its cells, with nothing in either receipt to say so.

Asked over HTTPS rather than through Docker for one reason: the API container renders every cloud
task definition and has no Docker daemon. A Docker-based resolver would work on a laptop and
nowhere else, which would put local and cloud back on separate mechanisms -- the exact split the
container-graph work spent two steps closing.

**Two digests, because a multi-arch image has two identities.** The INDEX digest names the release
-- one document listing a manifest per platform. The PLATFORM digest names the binary that actually
executes. `hindsight:latest` on an arm64 laptop and on Fargate share an index digest and differ in
platform digest: same release, different machine code. A record with one digest cannot tell that
apart from a genuinely different release, which is what makes a cross-machine comparison
unreadable.
"""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass
from typing import Any

import httpx

from memrank.errors import MemrankError

#: Media types a manifest request accepts. Both index forms first, because a multi-arch reference
#: must come back as an index -- asking only for the single-image types makes a registry pick a
#: platform for us, silently, and the index digest would then be unobtainable.
ACCEPT = ", ".join((
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.docker.distribution.manifest.v2+json",
))

#: buildx attaches attestation manifests to an index and marks them `unknown/unknown`. They are not
#: runnable images; matching one would pin a run to a provenance blob.
UNKNOWN = "unknown"

_ECR_HOST = ".dkr.ecr."
_GHCR_HOST = "ghcr.io"


class RegistryError(MemrankError):
    """A reference could not be resolved. Never swallowed -- see the module docstring."""


@dataclass(frozen=True)
class Resolution:
    """What a tag pointed at, at the moment a run started.

    ``tag`` is carried rather than re-derived: once a reference is pinned it reads
    ``repo@sha256:...`` and the tag is gone from it, but a receipt still wants to say which pointer
    was followed.
    """

    repository: str
    tag: str
    platform_digest: str
    platform: str
    #: Empty when the reference resolved straight to a manifest -- a single-architecture image has
    #: no index. Recorded as absent rather than as equal to the platform digest, because "this
    #: release has one architecture" and "we did not look" must not read the same.
    index_digest: str = ""

    @property
    def pinned(self) -> str:
        """The reference to actually run: identity, not a pointer."""
        return f"{self.repository}@{self.platform_digest}"


def split_reference(image: str) -> tuple[str, str]:
    """``(repository, tag)`` for a tagged reference.

    Deliberately narrow: a reference already pinned to a digest is not something to resolve, and
    saying so beats resolving the digest's own digest.

    Raises:
        RegistryError: If the reference names no tag.
    """
    if "@" in image:
        raise RegistryError(
            f"{image!r} is already pinned to a digest; resolution is for tags, and re-resolving a "
            f"digest would answer a question nobody asked")
    repository, separator, tail = image.rpartition(":")
    if not separator or "/" in tail:
        raise RegistryError(
            f"{image!r} names no tag. An image reference a run depends on must say which build it "
            f"means, even if that is ':latest'")
    return repository, tail


def _registry_and_name(repository: str) -> tuple[str, str]:
    """Split ``ghcr.io/vectorize-io/hindsight`` into its host and its repository name."""
    host, separator, name = repository.partition("/")
    if not separator or ("." not in host and ":" not in host):
        raise RegistryError(
            f"{repository!r} does not name a registry host. Every image a target runs is written "
            f"out in full, so a bare name means a reference lost its registry somewhere.")
    return host, name


def _authorization(host: str, name: str, client: httpx.Client) -> str:
    """The ``Authorization`` header value for ``host``.

    Only the two registries the shipped graphs use. Anything else raises by name rather than
    trying unauthenticated and reporting the resulting 401 as though the image were missing --
    which would send whoever meets it looking for a deleted tag.
    """
    if _ECR_HOST in host:
        import boto3

        token = boto3.client("ecr").get_authorization_token()
        return f"Basic {token['authorizationData'][0]['authorizationToken']}"
    if host == _GHCR_HOST:
        # Anonymous pull tokens: public images need no account, and memrank reads only public ones
        # here. A private ghcr image would need a credential this does not have, and would fail
        # with the registry's own 401 rather than silently.
        answer = client.get(f"https://{host}/token", params={"scope": f"repository:{name}:pull"})
        answer.raise_for_status()
        return f"Bearer {answer.json()['token']}"
    raise RegistryError(
        f"no way to authenticate to {host!r}; known registries are ghcr.io and ECR. Add one here "
        f"rather than letting an unauthenticated request fail as a missing image.")


def _select(manifests: list[dict[str, Any]], platform: str) -> str:
    """The digest of the entry matching ``platform``, e.g. ``linux/amd64``."""
    wanted_os, _, wanted_arch = platform.partition("/")
    for entry in manifests:
        declared = entry.get("platform") or {}
        if declared.get("os") == UNKNOWN or declared.get("architecture") == UNKNOWN:
            continue
        if declared.get("os") == wanted_os and declared.get("architecture") == wanted_arch:
            return str(entry["digest"])
    offered = sorted(
        f"{(entry.get('platform') or {}).get('os')}/{(entry.get('platform') or {}).get('architecture')}"
        for entry in manifests
        if (entry.get("platform") or {}).get("os") != UNKNOWN)
    raise RegistryError(
        f"this image publishes no {platform} build (it has: {', '.join(offered) or 'none'}). "
        f"Fargate does not emulate, so a run on this platform cannot be what the target declares.")


def resolve(image: str, *, platform: str, client: httpx.Client | None = None) -> Resolution:
    """Resolve ``image`` to the digests it points at right now.

    Args:
        image: A complete tagged reference, e.g. ``ghcr.io/vectorize-io/hindsight:latest``.
        platform: The platform this image will run on, ``os/arch``.
        client: An httpx client, for tests. One is created per call otherwise.

    Returns:
        The resolution, carrying both digests, the platform and the tag that was followed.

    Raises:
        RegistryError: If the reference cannot be resolved, for any reason. A run that cannot name
            what it will execute must not start: the receipt would claim a provenance nobody
            verified, which is worse than no run at all.
    """
    repository, tag = split_reference(image)
    host, name = _registry_and_name(repository)
    owned = client is None
    client = client or httpx.Client(timeout=30)
    try:
        headers = {"Accept": ACCEPT, "Authorization": _authorization(host, name, client)}
        answer = client.get(f"https://{host}/v2/{name}/manifests/{tag}", headers=headers)
        answer.raise_for_status()
        body = answer.content
        document = json.loads(body)
    except RegistryError:
        raise
    except Exception as exc:                    # noqa: BLE001 - re-raised with the reference named
        raise RegistryError(f"could not resolve {image!r}: {exc}") from exc
    finally:
        if owned:
            client.close()

    # Computed from the body rather than read from `Docker-Content-Digest`: that header is what the
    # spec says a registry SHOULD send and ECR does not send it at all. The digest of a manifest is
    # defined as the hash of its bytes, so computing it is both correct and universal.
    digest = "sha256:" + hashlib.sha256(body).hexdigest()
    manifests = document.get("manifests")
    if manifests is None:
        # A single-architecture image: the reference resolved straight to a manifest, so there is
        # one identity and no release-versus-binary distinction to record.
        return Resolution(repository=repository, tag=tag, platform_digest=digest, platform=platform)
    return Resolution(repository=repository, tag=tag, index_digest=digest,
                      platform_digest=_select(manifests, platform), platform=platform)


def decode_ecr_token(token: str) -> tuple[str, str]:
    """``(user, password)`` from an ECR authorization token. Exposed for tests and diagnostics."""
    user, _, password = base64.b64decode(token).decode().partition(":")
    return user, password
