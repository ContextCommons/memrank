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
"""What a run's ``--judge provider:model`` names, settled before the agent starts.

A built-in provider (``anthropic``, ``openai``) resolves on its own. Any other provider is the
name of one of the organisation's endpoints (``memrank endpoints``), read from the API with its
base URL, so every teammate's run resolves the same one; an unknown name is refused, naming the
endpoints the organisation does have. Then the judge's model is checked with the key the run will
judge with (:func:`memrank.judging.providers.check.model_refusal`), so a wrong model or a refused
key stops the run before it has asked the agent anything.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from memrank.accounts.org_endpoints import EndpointError, list_endpoints, resolve_endpoint
from memrank.errors import ActionRequired
from memrank.judging.providers import (
    BUILTINS,
    RESERVED_NAMES,
    Endpoint,
    ResolvedJudge,
    parse_spec,
    provider_label,
    resolve,
)
from memrank.outcome import Step

#: ``(target, key, models_listing) -> refusal or None``: the model check, injectable for tests.
ModelCheck = Callable[[ResolvedJudge, str, bool], "str | None"]


def _endpoint(row: dict[str, Any]) -> Endpoint:
    return Endpoint(name=row["name"], adapter=row["adapter"], api_base=row["api_base"],
                    key_ref=row["key_ref"], fingerprint=row["fingerprint"])


def _no_such_endpoint(http: Any, org: str, spec: str, name: str) -> ActionRequired:
    names = [row["name"] for row in list_endpoints(http, org)]
    known = ", ".join(sorted(BUILTINS)) + (f"; {org}'s endpoints: {', '.join(names)}" if names
                                           else f"; {org} has no endpoints")
    return ActionRequired(
        f"--judge {spec} names {name!r}, which is neither a built-in provider nor one of "
        f"{org}'s endpoints ({known}).",
        steps=(Step("List them:", (f"memrank endpoints ls --org {org}",)),
               Step("Or register it:",
                    (f"memrank endpoints add {name} --adapter openai --base-url <URL> "
                     f"--key-ref <KEY_NAME> --org {org}",))))


def resolve_run_judge(http: Any, org: str, spec: str) -> tuple[ResolvedJudge, bool]:
    """``spec`` resolved for a run in ``org``, and whether its endpoint has a models listing.

    Raises:
        ActionRequired: The provider is neither built in nor one of ``org``'s endpoints.
        ValueError: The spec is malformed, or names a reserved provider that cannot judge.
    """
    provider, _ = parse_spec(spec)
    if provider in BUILTINS or provider in RESERVED_NAMES:
        return resolve(spec), True  # a reserved name is refused here, with resolve's words
    try:
        row = resolve_endpoint(http, org, provider)
    except EndpointError as exc:
        if exc.status != 404:
            raise
        raise _no_such_endpoint(http, org, spec, provider) from exc
    endpoint = _endpoint(row)
    return resolve(spec, (endpoint,)), bool(row["models_listing"])


def check_model(target: ResolvedJudge, key: str, models_listing: bool, org: str,
                check: ModelCheck) -> None:
    """Refuse the run when ``key`` does not reach ``target``'s model; nothing has run yet."""
    refusal = check(target, key, models_listing)
    if refusal is None:
        return
    who = provider_label(target.provider)
    raise ActionRequired(
        f"{who[0].upper()}{who[1:]} will not judge with {target.model!r} and {org}'s "
        f"{target.key_name}: {refusal}. Nothing has run.",
        steps=(Step("Name a model it serves with --judge, or save a different key:",
                    (f"memrank secrets set {target.key_name} --org {org}",)),))
