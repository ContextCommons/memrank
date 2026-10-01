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
"""Build a live adapter from a resolved target manifest.

memrank cannot introspect a running engine -- the LLM and embedder live server-side and the adapter
only ever *declares* them (see :mod:`memrank.adapters.effective`). Before manifests there was exactly
one thing keeping that declaration honest: ``scripts/local-eval.sh`` passes the same env profile to
both the engine container and to memrank, so the two cannot drift.

A manifest is a second, independent declaration. Rather than let it replace the environment -- which
would delete that guarantee -- this module makes the two **cross-check** each other: where both
declare a component and they disagree, the run fails before any work is done. Where the environment
is silent, the manifest stands alone.

Real verification arrives in M5, when memrank launches the engine *from* the manifest and the
declaration becomes true by construction.
"""

from __future__ import annotations

import inspect
import os
from typing import Any

from memrank.adapters import REGISTRY, get_adapter
from memrank.core import MemoryAdapter
from memrank.targets.manifest import Manifest, ManifestError

# Manifest component field -> the env var suffix the engine and adapter both use.
# Mirrors the names read by memrank.adapters.effective.
_ENV_SUFFIX: dict[tuple[str, str], str] = {
    ("llm", "provider"): "LLM_PROVIDER",
    ("llm", "model"): "LLM_MODEL",
    ("embedder", "provider"): "EMBEDDER_PROVIDER",
    ("embedder", "model"): "EMBEDDER_MODEL",
    ("embedder", "dims"): "EMBEDDING_DIMS",
}

# Adapters whose constructor takes the transport as a keyword argument.
_TRANSPORT_KWARG: dict[str, str] = {"mem0": "mode"}


class DeclarationConflict(ManifestError):
    """A manifest and the environment declare different components for the same target."""


def env_prefix(adapter_name: str) -> str:
    """The env-var prefix an adapter reads, e.g. ``mem0`` -> ``MEM0_``."""
    return f"{adapter_name.upper()}_"


def _accepts(adapter_name: str, kwarg: str) -> bool:
    """Whether an adapter's constructor takes ``kwarg`` (baseline takes none)."""
    return kwarg in inspect.signature(REGISTRY[adapter_name].__init__).parameters


def _conflicts(target: Manifest) -> list[str]:
    """Every field where the manifest and the environment disagree, as readable lines."""
    prefix = env_prefix(target.adapter)
    found: list[str] = []
    for (role, field_name), suffix in _ENV_SUFFIX.items():
        component = target.components.get(role)
        if component is None:
            continue
        declared = getattr(component, field_name)
        if declared is None:
            continue
        var = f"{prefix}{suffix}"
        actual = os.environ.get(var)
        if actual is not None and actual != str(declared):
            found.append(f"  {var}: manifest says {declared!r}, environment says {actual!r}")
    return found


def check_declaration(target: Manifest) -> None:
    """Raise if the environment contradicts the manifest.

    Args:
        target: The resolved manifest.

    Raises:
        DeclarationConflict: Listing EVERY disagreement at once, so one run surfaces them all.
    """
    found = _conflicts(target)
    if found:
        joined = "\n".join(found)
        raise DeclarationConflict(
            f"target {target.name!r} disagrees with the environment:\n{joined}\n"
            "The environment configures the running engine, so one of them is wrong. "
            "Unset the variable to trust the manifest, or pick the target that matches the engine.")


def _engine_conflicts(target: Manifest, reported: dict[str, Any]) -> list[str]:
    """Every field where the manifest and the engine's own report disagree."""
    found: list[str] = []
    for role in ("llm", "embedder"):
        component = target.components.get(role)
        actual_component = reported.get(role) or {}
        if component is None:
            continue
        for field_name in ("provider", "model", "dims"):
            declared = getattr(component, field_name, None)
            if declared is None or field_name not in actual_component:
                continue
            actual = actual_component[field_name]
            if actual is not None and str(actual) != str(declared):
                found.append(
                    f"  {role}.{field_name}: manifest says {declared!r}, engine says {actual!r}")
    return found


def check_engine(adapter: MemoryAdapter, target: Manifest) -> str:
    """Compare the manifest against the engine's own report of itself.

    Args:
        adapter: A constructed adapter for ``target``.
        target: The resolved manifest.

    Returns:
        ``"engine"`` when the engine confirmed the declaration, ``"declared"`` when the engine
        cannot report its configuration (most engines) and the record is operator-asserted.

    Raises:
        DeclarationConflict: Listing EVERY disagreement at once.
    """
    reported = adapter.describe_engine()
    if reported is None:
        return "declared"
    found = _engine_conflicts(target, reported)
    if found:
        joined = "\n".join(found)
        raise DeclarationConflict(
            f"target {target.name!r} does not describe the running engine:\n{joined}\n"
            "The engine reports its own configuration, so the manifest is wrong for this engine. "
            "Pick the target that matches it, or reconfigure the engine.")
    return "engine"


def _component_dict(target: Manifest, role: str) -> dict[str, Any] | None:
    """A manifest component as the plain dict the run record stores."""
    component = target.components.get(role)
    if component is None:
        return None
    fields: dict[str, Any] = {"provider": component.provider, "model": component.model}
    if role == "embedder":
        fields["dims"] = component.dims
    return fields


def build_adapter(target: Manifest, *, timeout_s: float | None = None,
                  verify_engine: bool = True) -> MemoryAdapter:
    """Construct the adapter a resolved manifest names, with its components declared.

    Args:
        target: The resolved manifest.
        timeout_s: A per-invocation request timeout. Deliberately NOT a manifest field: it is a
            runtime knob that does not change what is under test, so it must not affect the
            target's identity. Ignored by adapters whose constructor does not accept it.
        verify_engine: Ask the engine to confirm the declaration. Off only where no engine can
            exist yet -- e.g. resolving a target purely to inspect it.

    Returns:
        A fresh adapter instance. Safe to call repeatedly -- ``workers > 1`` builds one per unit.

    Raises:
        DeclarationConflict: If the environment contradicts the manifest.
    """
    check_declaration(target)
    kwargs: dict[str, Any] = {}
    transport_kwarg = _TRANSPORT_KWARG.get(target.adapter)
    if transport_kwarg and target.transport in ("http", "sdk"):
        kwargs[transport_kwarg] = target.transport
    if timeout_s is not None and _accepts(target.adapter, "timeout_s"):
        kwargs["timeout_s"] = timeout_s
    # Engine-specific retrieval depth, for faithful variants that must reproduce a vendor's own
    # setting. Only adapters that accept it are given it, so a block on a target whose adapter has
    # no such knob is inert rather than a TypeError at construction.
    if target.retrieval and _accepts(target.adapter, "retrieval"):
        kwargs["retrieval"] = dict(target.retrieval)
    # Engine-specific settings fixed when memory is written, for faithful variants whose vendor
    # benchmarked an ingest configuration different from the product they ship. Same inert-if-
    # unsupported rule as `retrieval` above, and deliberately NOT the refusal `partitioning` uses:
    # an ignored ingest setting leaves the engine on its own documented default, which the receipt
    # still describes correctly via `config.target`.
    if target.ingest and _accepts(target.adapter, "ingest"):
        kwargs["ingest"] = dict(target.ingest)
    # How the engine scopes memory, for faithful variants reproducing a vendor's own partitioning.
    #
    # REFUSED rather than ignored when the adapter cannot honour it, unlike `retrieval` above. An
    # unsupported retrieval block leaves the engine on its own defaults, which is merely a setting
    # that did not apply. An unsupported partitioning block would leave memory single-scoped while
    # `config.target.partitioning` in the receipt says it was split -- a run that reports a
    # configuration it did not have. That is the plausible-number failure this project exists to
    # catch, so it stops at construction instead.
    if target.partitioning:
        if not _accepts(target.adapter, "partitioning"):
            raise ManifestError(
                f"target {target.name!r} declares partitioning but adapter "
                f"{target.adapter!r} cannot scope memory that way. Remove the block or use an "
                f"adapter that supports it -- running single-scoped would record a partitioning "
                f"that never happened.")
        kwargs["partitioning"] = dict(target.partitioning)
    # The SDK's own configuration object, handed over untouched -- the only way to state something
    # memrank has no vocabulary for, such as which vector store an in-process engine builds.
    #
    # Fenced by two rules, because an opaque passthrough beside a checked declaration is how a
    # receipt starts describing a system nobody ran:
    #
    #   1. `transport: sdk` only. Over HTTP the engine is configured by the environment
    #      `engine_env.py` renders, and a config object would be read by nobody while appearing in
    #      the receipt as though it had been.
    #   2. Never alongside `components`. They would be two sources of truth for the same fields,
    #      and the one memrank checks is not the one the SDK would obey.
    if target.sdk_config:
        if target.transport != "sdk":
            raise ManifestError(
                f"target {target.name!r} declares sdk_config but its transport is "
                f"{target.transport!r}. An SDK config object is read only in-process; over HTTP "
                f"the engine takes its configuration from the environment, so this block would be "
                f"recorded in the receipt and obeyed by nothing.")
        if target.components:
            raise ManifestError(
                f"target {target.name!r} declares both components and sdk_config. Those are two "
                f"vocabularies for the same settings and only sdk_config reaches an in-process "
                f"SDK -- so the receipt would show components the engine never read. State the "
                f"configuration once, in sdk_config.")
        if not _accepts(target.adapter, "config"):
            raise ManifestError(
                f"target {target.name!r} declares sdk_config but adapter {target.adapter!r} takes "
                f"no config object.")
        kwargs["config"] = dict(target.sdk_config)
    adapter = get_adapter(target.adapter, **kwargs)
    # The manifest's context_budget, applied to the INSTANCE. Until now this field was declared,
    # validated and then ignored for every stack target: the runner reads
    # `getattr(adapter, "context_budget")` (runner.py:367), which only the in-process control arms
    # set as class attributes. So a manifest could say `uncapped` and be silently capped anyway --
    # the exact manifest-versus-reality drift the catalog exists to prevent, and a blocker for
    # faithful variants, which are uncapped by definition.
    adapter.context_budget = target.context_budget
    # The runner uses this declaration for receipt identity. It is metadata on the local adapter
    # instance, never part of the adapter protocol exposed to engine integrations.
    adapter._memrank_target = target  # type: ignore[attr-defined]
    adapter.declare_components(
        llm=_component_dict(target, "llm"),
        embedder=_component_dict(target, "embedder"),
        transport=target.transport,
        verified=check_engine(adapter, target) if verify_engine else None,
    )
    return adapter
