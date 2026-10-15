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
"""Provider adapters for the judge and the reader, modelled on LiteLLM's three layers.

A ``provider:model`` spec selects an adapter (:mod:`spec`); the adapter normalises the wire format
and the errors (:mod:`anthropic`, :mod:`openai`, :class:`ProviderError`); the endpoint and key come
from configuration -- a built-in entry or an :class:`Endpoint` -- never from the model string.
"""

from __future__ import annotations

from memrank.judging.providers.anthropic import anthropic_send
from memrank.judging.providers.errors import ProviderError
from memrank.judging.providers.openai import openai_send
from memrank.judging.providers.spec import (
    BUILTINS,
    DECIDERS,
    RESERVED_NAMES,
    Endpoint,
    ResolvedJudge,
    check_endpoints,
    parse_spec,
    provider_label,
    reach_label,
    resolve,
)
from memrank.judging.providers.types import Adapter, Send

#: The adapter each resolved spec that completes text is sent through. A decider (``typesafe``) has
#: none: it judges through :mod:`memrank.judging.jev`.
ADAPTERS: dict[str, Adapter] = {"anthropic": anthropic_send, "openai": openai_send}

__all__ = ["ADAPTERS", "BUILTINS", "DECIDERS", "RESERVED_NAMES", "Adapter", "Endpoint", "ProviderError",
           "ResolvedJudge", "Send", "check_endpoints", "parse_spec", "provider_label",
           "reach_label", "resolve"]
