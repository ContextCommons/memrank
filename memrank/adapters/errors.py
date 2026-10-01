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
# Used by the current path (connect/http, api_client): it sits in a deprecated package but is not
# deprecated itself, and it must not warn.
"""When an engine refuses a request, say what it said.

Every HTTP adapter guarded its calls with a bare ``response.raise_for_status()``. httpx's
``HTTPStatusError`` carries the status line and the URL and **not the body** -- which is the only
place an engine explains itself. That exception then reached ``runner._exit_with``'s last branch
and printed::

    error: internal error: HTTPStatusError: Client error '400 Bad Request' for url
    'http://localhost:17350/v1/memories/ingest'
    note: this is a bug in memrank; re-run with MEMRANK_DEBUG=1 for a traceback

Both halves are wrong. The reason was discarded at the one moment it existed, and an engine
rejecting our payload was reported as a defect in the harness -- sending the reader to read memrank's
source rather than the engine's answer. On 2026-08-13 that cost a BEAM run three hours of completed
ingest and left nothing to diagnose it with. :class:`memrank.runner.RateLimitExhausted` records the
identical lesson one status code over: a 429 that printed "this is a bug in memrank" "sends the
reader hunting the wrong thing entirely".

``adapters/native.py`` already got this right and is the model: it never calls
``raise_for_status``, checks the status by hand, and puts the translator's own error text in the
message. This module generalises that to the four adapters that talk HTTP to a vendor engine.

WHY THE BODY IS REDACTED AND CAPPED. An engine's error body is not ours and can echo anything it
was sent -- retrieved memory content, or a request header including the bearer token it just
rejected. ``memrank.errors`` states the standing policy that error paths must not leak what
``submit`` decrypted, which is why there is no locals-rendering traceback mode; the same rule
applies to a body we did not write.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from memrank.errors import MemrankError
from memrank.secrets.names import is_secret_key

#: How much of an engine's body to quote. Enough to carry a whole nested failure chain -- an engine
#: reporting its own provider's refusal, which is the shape every real rejection has had -- short of
#: pasting a rejected document back. The old 400 cut mid-sentence through the innermost reason.
BODY_LIMIT = 2000

#: How much of :data:`BODY_LIMIT` is spent on the END of an over-long body. Engines nest their
#: failures outermost-first, so the part that names what actually went wrong -- the limit that was
#: exceeded, the field that was rejected -- is the part a head-only truncation throws away.
TAIL_LIMIT = 700

#: What replaces a value whose key looks like a credential. Same token ``receipt._redact_secrets``
#: writes, so one grep finds every redaction in an artifact or a message.
REDACTED = "[redacted]"


class EngineRejected(MemrankError):
    """An engine refused a request, and said why.

    A ``MemrankError`` because it is not a bug in the harness: the run is over, but the reader
    should be looking at the engine. Carries the status separately from the message so a retry
    policy can ask whether trying again could possibly help -- a 5xx or a timeout may pass on the
    second attempt, a 400 will be refused identically forever.
    """

    def __init__(self, message: str, *, status: int, method: str, url: str) -> None:
        super().__init__(message)
        self.status = status
        self.method = method
        self.url = url

    @property
    def is_transient(self) -> bool:
        """Whether a second attempt could plausibly succeed.

        5xx only. A 4xx is the engine stating that this request is unacceptable, and re-sending it
        buys nothing but the cost of sending it -- which on a 390 s/doc engine is measured in hours.
        429 is deliberately NOT here: rate limiting has its own gate and its own retry
        (:class:`memrank.runner.RateLimitExhausted`), and treating it as a generic transient fault
        would retry around a limiter built to be waited on.
        """
        return self.status >= 500


class EngineTimedOut(MemrankError):
    """An engine did not answer within the time this adapter was willing to wait.

    Distinct from :class:`EngineRejected` because nothing was rejected: there is no response, no
    status and no body -- the engine may still be working. That distinction is the whole point of
    the class. A bare ``httpx.ReadTimeout`` reached ``runner._exit_with``'s last branch and printed
    "internal error: ReadTimeout: timed out / this is a bug in memrank", which is wrong twice over
    and cost a beam:100k-mini run on 2026-08-13: an engine running a small-language-model extractor
    needed ~301s per document, against a 60s default inherited from an adapter written for an
    engine that takes 43s.

    A timeout is the most diagnosable failure the harness has -- it set the limit, it knows which
    engine it asked, and it knows the variable that changes it -- so the message states all three
    and names the variable rather than describing it.
    """

    def __init__(self, engine: str, timeout_s: float, env_var: str, request: httpx.Request) -> None:
        super().__init__(
            f"{engine} did not answer within {timeout_s:g}s: {request.method} "
            f"{request.url.path}. It was not refusing -- no response arrived at all, and it may "
            f"still be working. Raise the limit with {env_var}=<seconds>, or give it less work "
            f"per request.")
        self.engine = engine
        self.timeout_s = timeout_s
        self.env_var = env_var

    @property
    def is_transient(self) -> bool:
        """Always False. A second identical request waits the same insufficient time.

        Deliberately not retried: on an engine that needs 301s per document, a retry policy that
        believes a timeout is transient spends the timeout again for the same outcome -- and the
        thing that would actually help, a longer limit, is the one thing a retry does not change.
        """
        return False


def unreachable_remedy(base_url_env: str | None) -> str:
    """What to do about an engine that is not answering, in this installation's own terms.

    Two remedies, named at the moment of failure: point memrank at the engine that IS running, or
    have memrank start one. "No engine is running" is a fact about the machine, and the reader
    should not have to already know how memrank is told where to look.

    Every path named here exists in an installed copy. The message this replaced sent the reader
    to a shell script in the maintainers' own checkout -- which the person who most needs this
    message, an outsider whose engine memrank cannot reach, does not have (ATO-2125).
    """
    settings = (f"the {base_url_env} environment variable or the adapter's `base_url` argument"
                if base_url_env else "the adapter's `base_url` argument")
    return (f"memrank sent that request and nothing answered. Point it somewhere else with "
            f"{settings}, or have memrank start a disposable engine of its own by re-running "
            f"with `--on local`.")


def safe_location(url: httpx.URL) -> str:
    """``url`` as scheme, host, port and path -- never its userinfo or query.

    A base URL is operator-supplied and can carry a credential in either place
    (``http://user:token@host`` or ``?api_key=...``); an error message is printed, logged and
    pasted into tickets, so it quotes only the part that says where memrank looked.
    """
    # httpx rebuilds the URL itself, so an IPv6 host keeps its brackets (http://[::1]:8888).
    return str(url.copy_with(username=None, password=None, query=None, fragment=None))


class EngineUnreachable(MemrankError):
    """Nothing answered at the address memrank was given for an engine.

    A bare ``httpx.ConnectError`` reached ``runner._exit_with``'s last branch and printed
    "internal error: ConnectError: [Errno 61] Connection refused / this is a bug in memrank" for
    ``memrank submit mem0 locomo --on none`` with no mem0 running -- telling the reader to file a
    bug against the harness for a service they had not started. A refused connection is a fact
    about the machine, and the harness knows everything needed to say which one: the engine, the
    address it tried, and the variable that moves it.
    """

    def __init__(self, engine: str, request: httpx.Request, base_url_env: str | None,
                 reason: str) -> None:
        location = safe_location(request.url)
        super().__init__(
            f"engine {engine!r} is not answering at {location}: {reason}\n"
            f"  {unreachable_remedy(base_url_env)}")
        self.engine = engine
        self.url = location
        self.base_url_env = base_url_env


class TimeoutTranslatingTransport(httpx.BaseTransport):
    """The one place an engine's timeout becomes a sentence naming its knob.

    It is also where a refused connection becomes :class:`EngineUnreachable`, for the same reason:
    one chokepoint every engine client already passes through. Only ``httpx.ConnectError`` is
    translated -- nothing was sent, so there is nothing else it can mean. Any other exception,
    including a programmer error in an adapter, passes through untouched.

    A transport rather than a try/except at each call site: there are ~15 request calls across five
    adapters, and a guard that must be repeated is a guard one sibling will be missing. Wrapping
    the transport means an adapter gets this by CONSTRUCTING ITS CLIENT, which
    ``test_every_engine_facing_adapter_builds_its_client_through_the_helper`` enumerates -- a sixth
    adapter that builds a bare ``httpx.Client`` fails there rather than shipping the old message.

    WRAPS an inner transport rather than subclassing ``HTTPTransport``, so the socket layer is
    injectable. Subclassing made the only honest test a real connection to a port that never
    answers -- a timing-dependent test, which this project does not write, and the first attempt at
    one patched over ``handle_request`` and silently tested nothing.
    """

    def __init__(self, inner: httpx.BaseTransport, *, engine: str, timeout_s: float,
                 env_var: str, base_url_env: str | None = None) -> None:
        self._inner = inner
        self._engine = engine
        self._timeout_s = timeout_s
        self._env_var = env_var
        self._base_url_env = base_url_env

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        try:
            return self._inner.handle_request(request)
        except httpx.TimeoutException as exc:
            raise EngineTimedOut(self._engine, self._timeout_s, self._env_var, request) from exc
        except httpx.ConnectError as exc:
            raise EngineUnreachable(self._engine, request, self._base_url_env, str(exc)) from exc

    def close(self) -> None:
        self._inner.close()


def engine_client(*, engine: str, timeout_s: float, env_var: str,
                  base_url_env: str | None = None,
                  transport: httpx.BaseTransport | None = None, **kwargs: Any) -> httpx.Client:
    """An ``httpx.Client`` for talking to a vendor engine, whose failures explain themselves.

    Every adapter that speaks HTTP to an engine builds its client here. ``kwargs`` are httpx's own
    (``base_url``, ``headers``, ...); ``timeout`` is set from ``timeout_s`` so the limit the message
    quotes cannot drift from the limit that fires. ``base_url_env`` is the variable that moves the
    engine's address, named when nothing answers there. ``transport`` exists for tests to supply a
    socket layer that fails on demand.
    """
    return httpx.Client(
        timeout=timeout_s,
        transport=TimeoutTranslatingTransport(transport or httpx.HTTPTransport(), engine=engine,
                                              timeout_s=timeout_s, env_var=env_var,
                                              base_url_env=base_url_env),
        **kwargs)


class ConfigurationNotTaken(MemrankError):
    """An engine is up, but not configured the way the target declares.

    Its own class rather than a bare ``RuntimeError``, which reached the "this is a bug in memrank"
    branch and sent the reader to the harness for what is a live-engine mismatch. Refusing here is
    the point: a run that ingested anyway would record a configuration the engine does not have,
    and the number would be attributed to the wrong system.
    """


def _redact(value: Any) -> Any:
    """``value`` with anything that looks like a credential replaced, recursively."""
    if isinstance(value, dict):
        return {k: (REDACTED if is_secret_key(str(k)) else _redact(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact(v) for v in value]
    return value


def _capped(body: str) -> str:
    """``body`` within :data:`BODY_LIMIT`, keeping both ends and saying what it dropped.

    Keeping only the head is what produced ``exceeds the server limit of 163`` -- a number cut in
    half at the one moment it was the whole finding. Head and tail together cost no more of the
    body than a single window would, so this buys the innermost reason without widening what an
    engine can echo back at us.
    """
    if len(body) <= BODY_LIMIT:
        return body
    head = body[:BODY_LIMIT - TAIL_LIMIT]
    tail = body[-TAIL_LIMIT:]
    return f"{head}[... {len(body) - BODY_LIMIT} chars elided ...]{tail}"


def body_excerpt(response: httpx.Response) -> str:
    """The engine's own explanation, redacted and capped.

    Prefers a JSON ``error`` field because that is where every engine here puts its message, falls
    back to the whole JSON body, then to raw text. ``(empty body)`` rather than an empty string:
    "the engine said nothing" is itself the finding when a proxy or a sidecar is the one answering.
    """
    try:
        body = response.json()
    except ValueError:
        return _capped(response.text.strip()) or "(empty body)"
    if isinstance(body, dict) and body.get("error"):
        return _capped(str(_redact({"error": body["error"]})["error"]))
    return _capped(json.dumps(_redact(body), default=str))


def raise_for_status(response: httpx.Response) -> httpx.Response:
    """``response`` unchanged, or :class:`EngineRejected` naming what the engine said.

    The one replacement for ``httpx.Response.raise_for_status`` across every adapter that talks to
    a vendor engine. Returns the response so it can be used inline where the old call sat.
    """
    if response.status_code < 400:
        return response
    request = response.request
    raise EngineRejected(
        f"engine rejected {request.method} {request.url.path} "
        f"({response.status_code}): {body_excerpt(response)}",
        status=response.status_code, method=request.method, url=str(request.url))


def safe_json(response: httpx.Response) -> dict[str, Any]:
    """An engine's response body as a dict, or ``{}`` when it is not JSON.

    One copy of what was four byte-identical ``_safe_json`` static methods (atomicmemory,
    supermemory, mem0, hindsight). A non-dict JSON value is wrapped under ``results`` because two
    engines answer a search with a bare list.

    Only ever reached on a SUCCESS: the checked request above raises before this on a 4xx, so an
    empty dict here means "the engine returned 200 and no JSON", never "the engine refused".
    """
    try:
        data = response.json()
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {"results": data}
