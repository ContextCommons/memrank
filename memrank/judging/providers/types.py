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
"""The adapter boundary: what an adapter is given, and what it hands back."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Protocol

from memrank.judging.providers.spec import ResolvedJudge

#: One call: ``(model, system, user, sampling) -> text``. ``sampling`` is sent exactly as given;
#: an adapter never drops a parameter its provider rejects.
Send = Callable[[str, str, str, Mapping[str, Any]], str]


class Adapter(Protocol):
    """Builds a :data:`Send` for a resolved spec and its key.

    An adapter normalises the wire format and the errors (:class:`ProviderError`) and nothing
    else: no parameter dropping, no routing to another deployment.
    """

    def __call__(self, target: ResolvedJudge, api_key: str, *, max_tokens: int) -> Send: ...
