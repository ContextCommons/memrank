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
"""Judge specs: ``provider:model``, and what a provider name resolves to.

A spec names the provider first and passes everything after the FIRST colon to that provider's
adapter unchanged, the way LiteLLM's ``openai/gpt-4.1`` does. Router model ids carry ``/`` and
Bedrock ids a further ``:`` (``...-v1:0``), so splitting anywhere else would corrupt them. A spec
with no provider is refused: an implicit default provider is the ambiguity that makes a record
unable to say who judged.

The endpoint and the key come from configuration -- a built-in provider's fixed entry, or an
:class:`Endpoint` -- and never from the model string.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

# One source for what an endpoint may be called, shared with the organisation's registry: the
# built-ins, `custom` (a Level 2 completer's records) and `unknown` (records whose provider was
# never written down) are reserved.
from memrank.accounts.org_endpoints import NAME_PATTERN, RESERVED_NAMES

#: The adapters a named endpoint can speak through: the ones that complete text.
ADAPTER_NAMES = ("anthropic", "openai")

#: Built-ins that decide rather than complete: they answer typed questions with probabilities, so
#: they judge through their own grader (:mod:`memrank.judging.jev`) and never write an answer.
DECIDERS = ("typesafe",)

#: Model names that move when a release ships. A threshold is tied to the release it was set
#: against, so a decider is named by a versioned id and these are refused.
MOVING_ALIASES = ("-latest", "-preview")

#: The field each adapter sends its output budget under.
MAX_TOKENS = "max_tokens"
MAX_COMPLETION_TOKENS = "max_completion_tokens"


@dataclass(frozen=True)
class Builtin:
    """A provider memrank knows without configuration: its adapter, endpoint and key name."""

    adapter: str
    api_base: str
    key_name: str
    max_tokens_field: str


#: The vendors' own endpoints and key names, as LiteLLM and the vendors' SDKs document them.
#: OpenAI's own API documents ``max_completion_tokens``; compatible routers implement
#: ``max_tokens``, which is why the field is per entry and not per adapter.
BUILTINS: dict[str, Builtin] = {
    "anthropic": Builtin("anthropic", "https://api.anthropic.com", "ANTHROPIC_API_KEY",
                         MAX_TOKENS),
    "openai": Builtin("openai", "https://api.openai.com/v1", "OPENAI_API_KEY",
                      MAX_COMPLETION_TOKENS),
    # TypeSafe System One. It has no output budget, so it sends no max-tokens field at all.
    "typesafe": Builtin("typesafe", "https://api.typesafe.ai", "TYPESAFE_API_KEY", ""),
}


@dataclass(frozen=True)
class Endpoint:
    """A named endpoint: an adapter speaking to an ``api_base`` with the key named ``key_ref``.

    The counterpart of a LiteLLM proxy ``model_list`` entry, without a pinned model: an alias would
    hide which model judged. ``fingerprint`` is what records carry in place of the URL -- the
    organisation's ``<id>@<revision>``, or ``local:<name>`` for one built in code. ``api_base`` is
    kept out of the repr because a private URL does not belong in a log line or a traceback.
    """

    name: str
    adapter: str
    api_base: str = field(repr=False)
    key_ref: str
    fingerprint: str = ""

    def __post_init__(self) -> None:
        if not NAME_PATTERN.match(self.name):
            raise ValueError(f"an endpoint name is lowercase letters, digits and '-', starting "
                             f"with a letter, at most 32 characters; got {self.name!r}")
        if self.name in RESERVED_NAMES:
            raise ValueError(f"{self.name!r} is a reserved provider name; name the endpoint "
                             f"something else")
        if self.adapter not in ADAPTER_NAMES:
            raise ValueError(f"endpoint {self.name!r} has adapter {self.adapter!r}; choose one "
                             f"of {', '.join(ADAPTER_NAMES)}")
        if not self.api_base:
            raise ValueError(f"endpoint {self.name!r} has no api_base")
        if not self.key_ref:
            raise ValueError(f"endpoint {self.name!r} has no key_ref")
        if not self.fingerprint:
            object.__setattr__(self, "fingerprint", f"local:{self.name}")


@dataclass(frozen=True)
class ResolvedJudge:
    """What a spec resolved to: the adapter, the model it receives, the endpoint and key name.

    ``endpoint`` is ``None`` for a built-in provider.
    """

    adapter: str
    model: str
    endpoint: Endpoint | None
    key_name: str

    @property
    def provider(self) -> str:
        """The name the spec used: a built-in's, or the endpoint's."""
        return self.endpoint.name if self.endpoint else self.adapter

    @property
    def api_base(self) -> str:
        return self.endpoint.api_base if self.endpoint else BUILTINS[self.adapter].api_base

    @property
    def fingerprint(self) -> str | None:
        return self.endpoint.fingerprint if self.endpoint else None

    @property
    def decides(self) -> bool:
        """Whether this judge decides typed questions (:data:`DECIDERS`) rather than completing."""
        return self.endpoint is None and self.adapter in DECIDERS

    @property
    def max_tokens_field(self) -> str:
        # An endpoint is a compatible server, which implements the long-standing field.
        return MAX_TOKENS if self.endpoint else BUILTINS[self.adapter].max_tokens_field


#: How a message names a built-in provider; an endpoint is named by its own name.
_VENDORS = {"anthropic": "Anthropic", "openai": "OpenAI", "typesafe": "TypeSafe"}


def provider_label(provider: str) -> str:
    """``provider`` as a sentence names it: ``Anthropic``, ``OpenAI``, or ``the endpoint 'x'``.

    Never the URL: an endpoint's base URL may be an internal host.
    """
    return _VENDORS.get(provider, f"the endpoint {provider!r}")


def reach_label(target: ResolvedJudge) -> str:
    """Where ``target``'s calls go, as a "check that this machine can reach ..." step says it."""
    if target.endpoint is not None:
        return provider_label(target.provider)
    host = target.api_base.split("://", 1)[-1].split("/", 1)[0]
    return f"{provider_label(target.provider)} ({host})"


def parse_spec(spec: str) -> tuple[str, str]:
    """``(provider, model)`` from ``provider:model``, split on the first colon only."""
    provider, colon, model = spec.partition(":")
    if not colon:
        raise ValueError(f"judge {spec!r} names no provider; write it as provider:model, "
                         f"e.g. anthropic:{spec}" if spec else "the judge spec is empty")
    if not provider or not model:
        raise ValueError(f"judge {spec!r} must name both a provider and a model, "
                         f"as provider:model")
    return provider, model


def resolve(spec: str, endpoints: Sequence[Endpoint] = ()) -> ResolvedJudge:
    """Resolve ``spec`` against the built-ins and ``endpoints``; refuse anything else."""
    provider, model = parse_spec(spec)
    if provider in BUILTINS:
        builtin = BUILTINS[provider]
        if provider in DECIDERS and model.endswith(MOVING_ALIASES):
            raise ValueError(f"judge {spec!r} names a moving alias; name the versioned model, "
                             f"e.g. {provider}:jev-1.14.0, so the verdicts stay tied to it")
        return ResolvedJudge(builtin.adapter, model, None, builtin.key_name)
    named = {endpoint.name: endpoint for endpoint in endpoints}
    if provider in named:
        endpoint = named[provider]
        return ResolvedJudge(endpoint.adapter, model, endpoint, endpoint.key_ref)
    if provider == "custom":
        raise ValueError("custom: is recorded for a completer given in code "
                         "(JudgeConfig.completer with completer_name); it cannot be typed")
    if provider in RESERVED_NAMES:
        raise ValueError(f"provider {provider!r} is reserved and cannot judge in this build")
    known = sorted(BUILTINS) + sorted(named)
    raise ValueError(f"unknown provider {provider!r} in {spec!r}; known: {', '.join(known)}")


def check_endpoints(endpoints: Sequence[Endpoint]) -> None:
    """Refuse two endpoints sharing a name: which one judged would depend on list order."""
    names = [endpoint.name for endpoint in endpoints]
    duplicated = sorted({name for name in names if names.count(name) > 1})
    if duplicated:
        raise ValueError(f"endpoint names must be unique; repeated: {', '.join(duplicated)}")
