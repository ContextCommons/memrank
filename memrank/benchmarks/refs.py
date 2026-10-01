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
"""Naming an evaluation, the way `hindsight:matched` names a target.

A tier and a slice are not options on a run the way `--workers` is. They select WHICH evaluation
is being run, and a number does not carry from one to another: BEAM at 100k/smoke is 20 questions
over one conversation, BEAM at 1m is 700 over thirty-five. The leaderboard already knew
(`leaderboard.board_key` segregates by tier and slice), but nothing gave the pair a NAME, so the
artifact layer wrote both to `results/{target}__beam.json` and the second silently replaced the
first.

The grammar is the target grammar, reused rather than reinvented: `memrank.targets.resolve` is
target-agnostic, `beam:100k-smoke` already parses under it, and its charset already rejects the
free-form shapes a combinatorial tag would invite.

Variants are declared on the benchmark class through the `tiers` and `slices` that `EvalInfo`
already carries, and enumerated up front. Enumerable rather than parsed-on-the-fly is the point:
`memrank evals ls` can then show every evaluation that exists, which is the question "what can I
run" actually asks.
"""

from __future__ import annotations

from typing import Any

from memrank.core import Benchmark
from memrank.targets.resolve import RefError, parse_ref


def _named_slices(cls: type[Benchmark]) -> tuple[str, ...]:
    """The slices that are actual values, not documentation.

    `relation_graph` declares `("smoke", "mini", "<group>", "<fixture-id>")` -- the last two are
    placeholders meaning "any group name" and "any fixture id", not slices you can name. They are
    excluded from the enumeration and handled by :func:`_takes_free_slice` instead.
    """
    return tuple(s for s in cls.info.slices if "<" not in s)


def _takes_free_slice(cls: type[Benchmark]) -> bool:
    """Whether this benchmark accepts a slice value it did not enumerate."""
    return any("<" in s for s in cls.info.slices)


def _variants(cls: type[Benchmark]) -> dict[str | None, dict[str, Any]]:
    """Every preset this benchmark declares, mapped to its constructor kwargs.

    The bare form (`None`) is always present and always canonicalises to something explicit: a
    tiered benchmark's default tier is its first, which `EvalInfo` already documents. Canonical,
    not refused -- `beam` remains runnable, but what it RAN is recorded as `beam:100k`, so nothing
    downstream has to reconstruct the default to know what it looked at.
    """
    tiers: tuple[str | None, ...] = cls.info.tiers or (None,)
    variants: dict[str | None, dict[str, Any]] = {}
    for tier in tiers:
        for slice_ in (None, *_named_slices(cls)):
            kwargs: dict[str, Any] = {}
            if tier:
                kwargs["tier"] = tier
            if slice_:
                kwargs["slice"] = slice_
            preset = "-".join(p for p in (tier, slice_) if p) or None
            variants[preset] = kwargs
    # The bare ref is the default variant: first tier, full slice.
    variants[None] = {"tier": tiers[0]} if tiers[0] else {}
    return variants


def presets_for(cls: type[Benchmark]) -> list[str]:
    """The named variants of one benchmark, excluding the bare default."""
    return sorted(p for p in _variants(cls) if p)


def parse_eval_ref(ref: str) -> tuple[str, dict[str, Any], str]:
    """``"beam:100k-smoke"`` -> ``("beam", {"tier": "100k", "slice": "smoke"}, "beam:100k-smoke")``.

    Returns the registry name, the constructor kwargs, and the CANONICAL ref -- the last of which
    is what artifacts and run rows are keyed on, so `beam` and `beam:100k` cannot produce two
    differently-named records of the same evaluation.
    """
    from memrank.benchmarks import REGISTRY

    parsed = parse_ref(ref)
    if parsed.namespace:
        raise RefError(f"eval refs take no namespace: {ref!r}")
    cls = REGISTRY.get(parsed.name)
    if cls is None:
        raise RefError(f"unknown eval {parsed.name!r}; known: {', '.join(sorted(REGISTRY))}")
    variants = _variants(cls)
    if parsed.preset in variants:
        kwargs = dict(variants[parsed.preset])
    elif parsed.preset and _takes_free_slice(cls):
        # relation_graph selects by group name or fixture id, which cannot be enumerated ahead of
        # time. The charset is still enforced by `parse_ref`, so this is a value, not a wildcard.
        kwargs = {"slice": parsed.preset}
    else:
        known = presets_for(cls)
        detail = f"; known: {', '.join(f'{parsed.name}:{p}' for p in known)}" if known else ""
        raise RefError(f"unknown variant {ref!r}{detail}")
    return parsed.name, kwargs, canonical_ref(parsed.name, kwargs)


def canonical_ref(name: str, kwargs: dict[str, Any]) -> str:
    """The one spelling of an evaluation, built from what it was actually constructed with."""
    preset = "-".join(str(p) for p in (kwargs.get("tier"), kwargs.get("slice")) if p)
    return f"{name}:{preset}" if preset else name


def compose_ref(ref: str, *, tier: str | None = None, slice: str | None = None) -> str:
    """One canonical ref from a ref PLUS the tier/slice a caller carried beside it.

    The application layer (browser configs, planned experiments) still models tier and slice as
    settings next to an eval ref. Once the argv renderers stopped forwarding them as flags, a bare
    `beam` beside `tier="100k", slice="smoke"` would silently canonicalise to the default variant --
    a different evaluation than the one planned. Overlaying here keeps both vocabularies honest,
    and a CONTRADICTION (`beam:1m` beside `tier="100k"`) is refused rather than arbitrated: two
    different evaluations were named, and picking one would mislabel a run.

    A bare ref canonicalises to its default variant, and a default is not a caller selection:
    `beam` beside `tier="500k"` means `beam:500k`, not a conflict with the default `100k`. An
    explicit preset is a selection -- a supplied value may REFINE it (`beam:1m` + slice="smoke"
    -> `beam:1m-smoke`) but not contradict it (`beam:1m` + tier="100k" names two evaluations,
    and arbitrating would mislabel a run).

    Raises:
        RefError: When the ref's explicit preset sets a key the settings contradict, or when
            the overlay names a variant that does not exist.
    """
    name, kwargs, canonical = parse_eval_ref(ref)
    supplied = {k: v for k, v in (("tier", tier), ("slice", slice)) if v}
    if not supplied:
        return canonical
    if parse_ref(ref).preset:
        for key, value in supplied.items():
            if kwargs.get(key) is not None and kwargs[key] != value:
                raise RefError(f"{ref!r} already selects {key}={kwargs[key]!r}; "
                               f"it cannot also be submitted with {key}={value!r}")
    # The caller's values fill what the preset (or the bare form's defaults) left open, and the
    # overlaid spelling is re-parsed so an invented combination is refused, not minted.
    return parse_eval_ref(canonical_ref(name, {**kwargs, **supplied}))[2]


def list_eval_refs() -> list[str]:
    """Every runnable evaluation, canonically spelled."""
    from memrank.benchmarks import REGISTRY

    refs: list[str] = []
    for name, cls in REGISTRY.items():
        # The bare default is an evaluation like any other and belongs in the list -- for a
        # tiered benchmark it canonicalises to an explicit tier, so it is not a duplicate.
        refs.append(canonical_ref(name, _variants(cls)[None]))
        refs.extend(f"{name}:{p}" for p in presets_for(cls))
    return sorted(set(refs))
