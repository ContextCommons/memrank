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
"""Per-engine launch requirements -- the credentials needed to *instantiate* a memory engine.

memrank benchmarks an engine over HTTP; whether that engine runs in a local docker container or in
the cloud, launching it needs credentials. Those requirements follow from the **provider** in use,
not the specific model: openai -> ``OPENAI_API_KEY``, anthropic -> ``ANTHROPIC_API_KEY``,
voyage -> ``VOYAGE_API_KEY``; local providers (TEI / in-process ONNX / ollama) need no key. This
module declares, per engine, the canonical launch providers + any fixed secrets, and derives the set
of required secret env-vars. It is deliberately launch-mode-agnostic -- it answers "what does this
experiment need to be launchable?", which :mod:`memrank.config` preflights before a run.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from memrank.docs import doc_url
from memrank.errors import MemrankError

# Provider -> the API-key env var it needs. Empty string == keyless (local/in-process) provider.
PROVIDER_KEY_ENV: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "voyage": "VOYAGE_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "groq": "GROQ_API_KEY",
    "huggingface": "",   # local TEI
    "transformers": "",  # local in-process ONNX (atomicmemory)
    "ollama": "",        # local server
}


@dataclass(frozen=True)
class EngineRequirements:
    """What one engine needs to be launched, expressed as providers (+ fixed secrets).

    Args:
        name: Engine/adapter name.
        providers: Canonical launch providers by component, e.g. ``{"embedder": "huggingface",
            "llm": "anthropic"}``. Each maps through :data:`PROVIDER_KEY_ENV` to a required key
            (keyless providers contribute nothing). Overridable per experiment.
        fixed_secrets: Env-var names always required regardless of provider (e.g. supermemory's
            ``OPENAI_API_KEY``).
    """

    name: str
    providers: dict[str, str] = field(default_factory=dict)
    fixed_secrets: tuple[str, ...] = ()


REQUIREMENTS: dict[str, EngineRequirements] = {
    # mem0's own pair, matching the `mem0` target. Every shipped mem0 manifest declares BOTH roles,
    # and catalog.required_secrets_for passes them as overrides that shadow this row entirely -- so
    # it is the fallback for a mem0 target that declares no components, not a description of what
    # any catalog target needs. It said huggingface/anthropic until 2026-08-18, which described a
    # matched variant rather than mem0 itself; those variants left the package on 2026-08-19.
    "mem0": EngineRequirements("mem0", providers={"embedder": "openai", "llm": "openai"}),
    "atomicmemory": EngineRequirements(
        "atomicmemory", providers={"embedder": "transformers", "llm": "anthropic"}),
    "hindsight": EngineRequirements("hindsight", providers={"llm": "anthropic"}),
    # Spends nothing: local embedder, and no LLM call on memrank's ingest path. The provider key
    # its binary demands at BOOT is a constant in supermemory.compose.yaml, not a credential --
    # asking an org for one made this the only target a sweep needed a second account to run.
    "supermemory": EngineRequirements("supermemory"),
    # Spends nothing THROUGH MEMRANK. A translator's engine may well call a paid provider, but the
    # translator's own launch environment supplies that key -- memrank cannot enumerate credentials
    # for an engine it has never seen, and inventing a requirement would refuse runs that are
    # perfectly launchable. The vendor states what their engine needs in their own README.
    "native": EngineRequirements("native"),
    "word-overlap": EngineRequirements("word-overlap"),  # in-process; no requirements
    # The two named lexical methods (ATO-2259). In-process, keyless, no dependency.
    "tfidf": EngineRequirements("tfidf"),
    "bm25": EngineRequirements("bm25"),
    # The mandatory baseline arms (PRD_Memrank_v2.md:86). In-process and keyless: they retrieve
    # nothing or everything, so they call no provider. A judged run still needs the JUDGE's key,
    # which is a run-level requirement gathered separately by runner.run_credentials().
    "no-context": EngineRequirements("no-context"),
    "fixed-context": EngineRequirements("fixed-context"),
    "full-context": EngineRequirements("full-context"),
}


class UnknownEngine(MemrankError, KeyError):
    """An adapter name nothing has registered.

    Keeps ``KeyError`` so lookup call sites read naturally, and gains ``MemrankError`` so it prints
    as one actionable line rather than as "internal error: KeyError ... this is a bug in memrank".

    This is the FIRST wall a target with an unregistered adapter hits -- required secrets are
    computed at planning time, before any placement -- so it names what is known.
    """


def get_requirements(engine: str) -> EngineRequirements:
    """Return the requirements for ``engine``, or raise :class:`UnknownEngine`."""
    if engine not in REQUIREMENTS:
        raise UnknownEngine(
            f"no engine registered as {engine!r}; known: {', '.join(sorted(REQUIREMENTS))}. "
            f"See {doc_url('systems.md')}.")
    return REQUIREMENTS[engine]


def _key_for(provider: str | None) -> str:
    """The API-key env var for ``provider`` ('' when keyless/unknown)."""
    return PROVIDER_KEY_ENV.get(provider or "", "")


def keyed_providers(components: Mapping[str, Any]) -> dict[str, str | None]:
    """The provider whose own API key each declared component spends, by role.

    A component that names its own ``endpoint`` is served by that address rather than by its
    provider's API, so it spends no provider key: ``""`` says so. Deriving one anyway would hand,
    say, an ``OPENAI_API_KEY`` to whatever the endpoint is. Its credential is the one the target
    declares under ``secrets:``.

    Args:
        components: A manifest's components, each with ``provider`` and ``endpoint``.

    Returns:
        ``{role: provider}`` to pass to :func:`required_secrets` as overrides.
    """
    return {role: "" if component.endpoint else component.provider
            for role, component in components.items()}


def required_secrets(engine: str, *, embedder: str | None = None, llm: str | None = None) -> list[str]:
    """Secret env-vars needed to launch ``engine``.

    Providers default to the engine's canonical set; ``embedder``/``llm`` override them so you can
    preflight a variant (e.g. a hosted-embedder mem0 with ``embedder="voyage"``). Keyless providers
    contribute nothing.

    Args:
        engine: Engine name.
        embedder: Override embedder provider (else the canonical one).
        llm: Override LLM provider (else the canonical one).

    Returns:
        Sorted, de-duplicated list of required secret env-var names.
    """
    req = get_requirements(engine)
    providers = dict(req.providers)
    if embedder is not None:
        providers["embedder"] = embedder
    if llm is not None:
        providers["llm"] = llm
    needed = set(req.fixed_secrets)
    for provider in providers.values():
        env = _key_for(provider)
        if env:
            needed.add(env)
    return sorted(needed)
