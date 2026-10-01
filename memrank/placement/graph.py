"""A target's container graph, read from the Compose file its manifest names.

What a target runs is DECLARED, not assembled: the file lists every container, the images they run,
how they depend on one another and when each is healthy -- all of it in the Compose Specification's
own vocabulary, which both placements then read. Before this, the graph existed only as literals
inside the local renderer and, separately, inside an ECS task-definition template, so the two could
and did disagree about which model an engine was pointed at.

The one thing a graph must not state is what is being MEASURED. Component values -- the embedder,
the LLM -- come from the manifest and reach the containers by interpolation, so there is exactly one
place that says what a target is made of.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from memrank.targets.manifest import Manifest


def graph_path(target: Manifest) -> Path:
    """Where ``target``'s Compose file lives -- beside the manifest that names it."""
    from memrank.targets.catalog import graph_file

    if not target.compose:
        raise ValueError(f"{target.name!r} declares no compose file")
    return graph_file(target.compose)


def graph_variables(target: Manifest) -> dict[str, str]:
    """The values a graph interpolates for ``target``.

    Exactly one kind of value: what is being MEASURED. Component values are supplied here rather
    than written into the file, because the manifest's ``components:`` block is the one place that
    says what a target is made of -- the receipt and the leaderboard row read the same source. A
    model name typed into the graph as well would be a second place to change, and the ECS template
    that typed it twice is exactly how an engine came to be pointed at a model its sidecar was not
    serving.

    Image addresses are NOT here, and deliberately. Three graphs used to interpolate
    ``${MEMRANK_ENGINES_REGISTRY}``, completed at render time from an environment variable or a
    Terraform-derived file, so one target resolved differently depending on where it was rendered --
    the API, which has neither, could not render it at all. An address is part of what a target is;
    only the credentials to pull it belong to the deployment.

    Args:
        target: The resolved manifest.

    Returns:
        The substitutions to apply.
    """
    embedder = target.components.get("embedder")
    return {"MEMRANK_EMBEDDER_MODEL": embedder.model} if embedder and embedder.model else {}


def interpolate(document: str, variables: dict[str, str]) -> str:
    """Substitute ``${NAME}`` from ``variables``, leaving anything else untouched.

    Compose\'s own interpolation, applied here so ``targets render`` can resolve a graph without a
    Docker daemon. Unknown names are left as written rather than blanked: a silently empty image
    reference fails later and elsewhere, while an unresolved ``${...}`` names itself.
    """
    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        return variables.get(name, match.group(0))

    return re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", replace, document)


def service_name(target: Manifest) -> str:
    """The service the adapter drives, refused when the target names none.

    ``Manifest.service`` is optional because an in-process target has no containers at all. Every
    caller below is already past that branch, so a missing name there is an illegal state and not a
    mode to fall back through -- and stating the refusal once keeps it identical everywhere it is
    reached.

    Raises:
        ValueError: If the target declares no service.
    """
    if target.service is None:
        raise ValueError(
            f"{target.name!r} declares no compose service to drive; a target with containers "
            f"must name the one its adapter talks to")
    return target.service


def load_graph(target: Manifest) -> dict[str, Any]:
    """Read, interpolate and parse ``target``'s Compose file.

    Args:
        target: The resolved manifest.

    Returns:
        The Compose document, with every container the target needs.

    Raises:
        ValueError: If the target declares no graph, names a file that does not exist, or names a
            service that file does not define.
    """
    path = graph_path(target)
    if not path.is_file():
        raise ValueError(f"{target.name!r} names a compose file that does not exist: {path}")
    service = service_name(target)
    source = path.read_text(encoding="utf-8")
    document = yaml.safe_load(interpolate(source, graph_variables(target)))
    services = document.get("services") or {}
    if service not in services:
        raise ValueError(
            f"{target.name!r} names service {service!r}, which {path.name} does not "
            f"define (it has: {', '.join(sorted(services)) or 'nothing'})")
    return document


def engine_image(target: Manifest) -> str:
    """The image reference of the container the adapter drives, as the graph declares it."""
    return str(load_graph(target)["services"][service_name(target)]["image"])


def reference_parts(image: str) -> tuple[str, str]:
    """Split a declared image reference into ``(repository, tag)``.

    Provenance now comes from the reference the graph declares rather than from a registry and a
    tag supplied separately, so a receipt records where the image actually came from -- including
    for the vendor images memrank does not publish at all.

    Args:
        image: A complete OCI reference, e.g. ``ghcr.io/vectorize-io/hindsight:latest``.

    Returns:
        The repository and its tag; the tag is empty when the reference carries none or pins a
        digest instead, since a digest is not a tag and must not be recorded as one.
    """
    repository, _, digest = image.partition("@")
    if digest:
        return repository, ""
    # Only the last colon can introduce a tag, and only if no `/` follows it -- otherwise it is a
    # registry port, as in `localhost:5000/engine`.
    head, separator, tail = repository.rpartition(":")
    if separator and "/" not in tail:
        return head, tail
    return repository, ""


def pin(document: dict[str, Any], *, default_platform: str,
        resolver=None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve every image in ``document`` once, and return it pinned to digests.

    This is the moment a mutable pointer becomes a fact. The declaration keeps saying ``:latest``,
    because that is what a human means; what RUNS is ``repo@sha256:...``, identical in both
    placements because it was resolved once rather than asked of two registries independently.

    Args:
        document: A loaded Compose document.
        default_platform: The platform services run on when the graph declares none -- the ECS
            task's for cloud, the Docker daemon's locally.
        resolver: ``registry.resolve``, injectable for tests.

    Returns:
        ``(pinned document, {service: Resolution})``. The resolutions carry the tag that was
        followed, which the pinned reference no longer contains.

    Raises:
        RegistryError: If any image cannot be resolved. A run that cannot name what it will execute
            does not start, because its receipt would assert a provenance nobody verified.
    """
    from memrank.placement import registry

    resolve = resolver or registry.resolve
    pinned = {**document, "services": {}}
    resolutions: dict[str, Any] = {}
    for name, service in document["services"].items():
        resolution = resolve(service["image"],
                             platform=service.get("platform") or default_platform)
        resolutions[name] = resolution
        pinned["services"][name] = {**service, "image": resolution.pinned}
    return pinned, resolutions
