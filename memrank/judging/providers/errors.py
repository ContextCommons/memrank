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
"""The one error every judge adapter raises when its provider refuses or fails a call."""

from __future__ import annotations

from memrank.errors import MemrankError


class ProviderError(MemrankError):
    """A provider's answer that is not a completion, normalised across adapters.

    ``status_code`` is where :func:`memrank.rate_limit.is_rate_limited` looks, so a 429 from any
    adapter reaches the shared gate. ``quota_exhausted`` marks a 429 that waiting cannot fix --
    OpenAI's ``insufficient_quota`` -- so the gate refuses it at once instead of spending its
    deadline retrying a key with no credit.

    Attributes:
        provider: The provider name the spec used (a built-in's, or the endpoint's).
        status_code: The HTTP status; 200 when a reply came back with no text in it.
        message: What the provider said, verbatim.
        request_id: The provider's request id, when it sent one.
        retry_after: Seconds the provider asked us to wait, when it said.
        quota_exhausted: Whether the account is out of credit rather than over a rate.
    """

    def __init__(self, message: str, *, provider: str, status_code: int,
                 request_id: str | None = None, retry_after: float | None = None,
                 quota_exhausted: bool = False) -> None:
        request = f" (request {request_id})" if request_id else ""
        super().__init__(f"{provider} answered {status_code}: {message}{request}")
        self.provider = provider
        self.status_code = status_code
        self.message = message
        self.request_id = request_id
        self.retry_after = retry_after
        self.quota_exhausted = quota_exhausted
