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
"""The CLI's HTTP client for the memrank API, whose connection failures explain themselves.

A refused connection to the API reached ``runner._exit_with``'s last branch as "internal error:
ConnectError ... this is a bug in memrank" -- memrank blaming itself for an address nothing was
listening on. The CLI knows everything needed to say what happened instead: which API it tried,
where that address came from (the environment, the settings file, or the built-in default), and
that it is the connection, not the run, that failed.

The same chokepoint as :func:`memrank.adapters.errors.engine_client`, for the same reason: every
client to the API is built here, so a new call site gets the message by constructing its client.
"""

from __future__ import annotations

import httpx

from memrank import config, settings
from memrank.adapters.errors import safe_location
from memrank.errors import MemrankError

#: Where ``api.url`` came from, in the words the message uses.
_SOURCES = {settings.ENV: "from MEMRANK_API_URL in the environment",
            settings.FILE: "from api.url in {path}",
            settings.DEFAULT: "the built-in default"}


class ApiUnreachable(MemrankError):
    """Nothing answered at the memrank API address, or it did not answer in time."""


def url_source() -> str:
    """Where this CLI's API address was read from, as a phrase."""
    _, source = settings.resolve("api.url")
    return _SOURCES[source].format(path=settings.store_path())


def unreachable(request: httpx.Request, exc: httpx.TransportError) -> ApiUnreachable:
    """The explanation for a request to the API that got no answer."""
    what = ("did not answer in time" if isinstance(exc, httpx.TimeoutException)
            else "nothing answered there")
    configured = safe_location(httpx.URL(config.memrank_api_url()))
    return ApiUnreachable(
        f"cannot reach the memrank API at {configured} ({url_source()}): "
        f"{what}. This is the connection between this machine and the API, not your run: "
        "check that address and this machine's network, then retry.\n"
        f"  the underlying error: {request.method} {request.url.path}: "
        f"{type(exc).__name__}: {exc}")


class _ExplainingTransport(httpx.BaseTransport):
    """Turns a request that reached no API into :class:`ApiUnreachable`; nothing else."""

    def __init__(self, inner: httpx.BaseTransport) -> None:
        self._inner = inner

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        try:
            return self._inner.handle_request(request)
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            raise unreachable(request, exc) from exc

    def close(self) -> None:
        self._inner.close()


def api_client(*, timeout: float, token: str | None = None,
               transport: httpx.BaseTransport | None = None) -> httpx.Client:
    """An ``httpx.Client`` bound to the configured API, carrying ``token`` when given."""
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    inner = transport or httpx.HTTPTransport()
    return httpx.Client(base_url=config.memrank_api_url(), timeout=timeout, headers=headers,
                        transport=_ExplainingTransport(inner))

