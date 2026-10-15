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
"""The ``anthropic`` adapter: the Messages API through the ``anthropic`` SDK.

The SDK already retries transient failures twice, so this adapter adds no retry of its own; it
only names the endpoint explicitly and turns a refusal into :class:`ProviderError`. The endpoint is
always passed, so the SDK's own ``ANTHROPIC_BASE_URL`` never redirects ``anthropic:`` silently: a
stand-in is a named endpoint, which the record then names.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from memrank.judging.providers.errors import ProviderError
from memrank.judging.providers.spec import ResolvedJudge
from memrank.judging.providers.types import Send


def _refusal(exc: Any, provider: str) -> ProviderError:
    from memrank import rate_limit

    body = exc.body if isinstance(exc.body, dict) else {}
    said = body.get("error")
    message = str(said.get("message")) if isinstance(said, dict) else exc.message
    return ProviderError(message, provider=provider, status_code=exc.status_code,
                         request_id=getattr(exc, "request_id", None),
                         retry_after=rate_limit.retry_after_seconds(exc))


def anthropic_send(target: ResolvedJudge, api_key: str, *, max_tokens: int) -> Send:
    """A sender for ``target`` keyed by ``api_key``; refusals raise :class:`ProviderError`."""
    from memrank.errors import optional_import

    anthropic = optional_import("anthropic", None)
    client = anthropic.Anthropic(api_key=api_key, base_url=target.api_base)

    def send(model: str, system: str, user: str, sampling: Mapping[str, Any]) -> str:
        try:
            # No system prompt at all when there is none: the official readers are one user
            # message.
            msg = client.messages.create(
                model=model, max_tokens=max_tokens, **({"system": system} if system else {}),
                messages=[{"role": "user", "content": user}], **sampling)
        except anthropic.APIStatusError as exc:
            raise _refusal(exc, target.provider) from exc
        return "".join(b.text for b in msg.content if isinstance(b, anthropic.types.TextBlock))

    return send
