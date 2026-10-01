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
#
# Used by the current path (benchmarks/refs, through parse_ref): it sits in a deprecated package but is not
# deprecated itself, and it must not warn.
"""Pure resolution logic for target refs, overrides, and ``from:`` inheritance.

No file I/O lives here -- everything is string and mapping manipulation, so the grammar is testable
without a catalog on disk. The ref grammar follows the OCI/Ollama shape (``[namespace/]name[:preset]``)
with the deliberate restriction that the preset slot is a curated label, never a free-form variant
string (see docs/research/2026-07-30-artifact-reference-grammars.md).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from memrank.targets.manifest import COMPONENT_ROLES, ManifestError

# OCI path-component charset: lowercase alphanumerics with . _ - separators.
_PART = re.compile(r"^[a-z0-9]+([._-][a-z0-9]+)*$")


class RefError(ManifestError):
    """A target reference is malformed."""


@dataclass(frozen=True)
class TargetRef:
    """A parsed target reference."""

    name: str
    namespace: str | None = None
    preset: str | None = None

    @property
    def canonical(self) -> str:
        """The fully-spelled ref, echoed back to the user after resolution."""
        base = f"{self.namespace}/{self.name}" if self.namespace else self.name
        return f"{base}:{self.preset}" if self.preset else base


def _part(value: str, label: str, ref: str) -> str:
    """Validate one ref component against the path-component charset."""
    if not _PART.match(value):
        raise RefError(
            f"invalid {label} {value!r} in target ref {ref!r}; expected lowercase "
            "alphanumerics separated by . _ or -")
    return value


def _split_preset(text: str) -> tuple[str, str | None]:
    """Split ``body[:preset]``, rejecting a trailing colon with no preset."""
    if ":" not in text:
        return text, None
    body, _, preset = text.partition(":")
    if not preset:
        raise RefError(f"target ref {text!r} ends with ':' but names no preset")
    return body, _part(preset, "preset", text)


def _split_namespace(body: str, text: str) -> tuple[str | None, str]:
    """Split ``[namespace/]name``; an empty namespace (leading '/') is rejected."""
    if "/" not in body:
        return None, body
    namespace, _, name = body.partition("/")
    return _part(namespace, "namespace", text), name


def parse_ref(text: str) -> TargetRef:
    """Parse ``[namespace/]name[:preset]``.

    Args:
        text: The user-typed reference.

    Returns:
        The parsed TargetRef; ``.canonical`` is the form to echo back.

    Raises:
        RefError: On any malformed component.
    """
    if "," in text:
        raise RefError(
            f"target ref {text!r} contains a comma; a comma means 'sweep this axis' and is "
            "split by the caller, so it may not appear inside a single ref")
    if text.count("/") > 1:
        raise RefError(f"target ref {text!r} has more than one '/'; expected [namespace/]name")
    if text.count(":") > 1:
        raise RefError(f"target ref {text!r} has more than one ':'; expected name[:preset]")
    body, preset = _split_preset(text)
    namespace, name = _split_namespace(body, text)
    return TargetRef(name=_part(name, "name", text), namespace=namespace, preset=preset)


def parse_overrides(tokens: Sequence[str]) -> dict[str, str]:
    """Parse Hydra-style ``key=value`` argv tokens into a mapping.

    Args:
        tokens: Raw argv tokens trailing the target ref.

    Returns:
        The overrides as a mapping.

    Raises:
        RefError: On a token with no ``=`` (usually a misplaced benchmark name).
    """
    out: dict[str, str] = {}
    for token in tokens:
        key, sep, value = token.partition("=")
        if not sep or not key:
            raise RefError(f"override {token!r} is not valid; expected key=value")
        out[key] = value
    return out


def merge(parent: dict[str, Any], child: dict[str, Any]) -> dict[str, Any]:
    """Overlay ``child`` onto ``parent``: mappings merge recursively, everything else replaces.

    Lists replace wholesale rather than concatenating (Hydra's rule) -- a variant that narrows
    ``depends`` must be able to drop an inherited entry, which append semantics would make
    impossible. The ``from`` key is consumed here and does not appear in the result.

    Args:
        parent: The inherited mapping.
        child: The overlaying mapping.

    Returns:
        A new mapping; neither input is mutated.
    """
    out = deepcopy(parent)
    for key, value in child.items():
        if key == "from":
            continue
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def _set_component(data: dict[str, Any], role: str, value: str) -> None:
    """Apply a ``role=provider/model`` shorthand, replacing the component wholesale."""
    provider, sep, model = value.partition("/")
    if not sep or not model:
        raise RefError(
            f"override {role}={value!r} must be provider/model, e.g. voyage/voyage-4-large")
    # Replace wholesale: an inherited dims belongs to the OLD model and would silently be wrong.
    data.setdefault("components", {})[role] = {"provider": provider, "model": model}


def _set_dotted(data: dict[str, Any], key: str, value: str) -> None:
    """Apply a dotted ``role.field=value`` override, or ``binding.root=<path>``.

    ``binding.root`` is the one non-component override, and it earns the exception: pointing a
    source target at a different checkout for a single run is how two commits or two worktrees get
    compared without registering each as its own target first. It is the same escape hatch Nix
    spells ``--override-input`` and Bazel ``--override_repository``, and it beats whatever the
    machine has linked, because it was typed for this run.
    """
    role, _, field_name = key.partition(".")
    if key == "binding.root":
        binding = data.setdefault("binding", {})
        binding["root"] = value
        # A stated path and a link are mutually exclusive; the argv wins, so the link steps aside
        # rather than tripping that refusal.
        binding.pop("rootFrom", None)
        return
    if role not in COMPONENT_ROLES:
        raise RefError(f"unknown override key {role!r}; known: {', '.join(COMPONENT_ROLES)}, "
                       f"binding.root")
    component = data.setdefault("components", {}).setdefault(role, {})
    component[field_name] = int(value) if field_name == "dims" else value


def apply_overrides(data: dict[str, Any], overrides: dict[str, str]) -> dict[str, Any]:
    """Return a copy of ``data`` with ``overrides`` applied.

    Role shorthands (``embedder=``) are applied before dotted keys (``embedder.dims=``) so the
    result does not depend on argv order.

    Args:
        data: The flattened manifest mapping.
        overrides: Parsed ``key=value`` pairs.

    Returns:
        A new mapping; ``data`` is not mutated.

    Raises:
        RefError: On an unknown key, a malformed shorthand, or a comma in a value.
    """
    for key, value in overrides.items():
        if "," in value:
            raise RefError(
                f"override {key}={value!r} contains a comma; a comma means 'sweep this axis' "
                "and is split by the caller, so it may not appear inside a single value")
    out = deepcopy(data)
    for role in COMPONENT_ROLES:
        if role in overrides:
            _set_component(out, role, overrides[role])
    for key, value in overrides.items():
        if key in COMPONENT_ROLES:
            continue
        if "." not in key:
            raise RefError(f"unknown override key {key!r}; known: {', '.join(COMPONENT_ROLES)}")
        _set_dotted(out, key, value)
    return out
