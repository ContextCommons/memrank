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
"""When an engine refuses, the message is the engine's, and it is not called a memrank bug.

Every HTTP adapter used a bare `raise_for_status()`, whose exception carries the status line and
the URL and not the body -- the only place an engine explains itself. On 2026-08-13 a BEAM run died
on `400 Bad Request for url .../v1/memories/ingest` with no reason recoverable anywhere, having
already completed three hours of ingest.

No adapter had a single test for its error path: every fake stubbed `raise_for_status` as a no-op,
so the branch was tested out of existence. This is that branch.
"""
from __future__ import annotations

import httpx
import pytest

from memrank.adapters import errors as adapter_errors


def _response(status: int, *, json_body: object = None, text: str = "") -> httpx.Response:
    """A real httpx.Response bound to a real request -- `raise_for_status` reads both."""
    request = httpx.Request("POST", "http://localhost:17350/v1/memories/ingest")
    if json_body is not None:
        return httpx.Response(status, json=json_body, request=request)
    return httpx.Response(status, text=text, request=request)


def test_the_engines_own_words_reach_the_message():
    """The whole point: a 400 body says which field was wrong, and that used to be discarded."""
    with pytest.raises(adapter_errors.EngineRejected) as caught:
        adapter_errors.raise_for_status(
            _response(400, json_body={"error": "conversation exceeds 32000 tokens"}))

    assert "conversation exceeds 32000 tokens" in str(caught.value)


def test_the_message_names_the_engine_not_the_harness():
    """`internal error ... this is a bug in memrank` sent the reader to the wrong source tree."""
    with pytest.raises(adapter_errors.EngineRejected) as caught:
        adapter_errors.raise_for_status(_response(400, json_body={"error": "bad"}))

    message = str(caught.value)
    assert message.startswith("engine rejected POST /v1/memories/ingest (400)")
    assert "bug in memrank" not in message


def test_a_refusal_is_a_clean_error_not_an_internal_one():
    """`MemrankError` is what routes it through the CLI boundary as a one-line refusal."""
    from memrank.errors import MemrankError

    assert issubclass(adapter_errors.EngineRejected, MemrankError)


@pytest.mark.parametrize("status,transient", [
    (400, False), (404, False), (422, False),
    # Rate limiting has its own gate and its own waiting; treating it as a generic transient
    # fault would retry around a limiter built to be waited on.
    (429, False),
    (500, True), (502, True), (503, True),
])
def test_only_a_server_fault_is_worth_a_second_attempt(status, transient):
    """What the per-unit retry branches on. A 400 refused once is refused identically forever,
    and on a 390 s/doc engine each pointless attempt costs real hours."""
    with pytest.raises(adapter_errors.EngineRejected) as caught:
        adapter_errors.raise_for_status(_response(status, json_body={"error": "x"}))

    assert caught.value.is_transient is transient


def test_a_success_passes_straight_through():
    response = _response(200, json_body={"ok": True})

    assert adapter_errors.raise_for_status(response) is response


def test_a_body_that_is_not_json_still_reaches_the_reader():
    """Proxies and sidecars answer with HTML, and that fact IS the finding."""
    with pytest.raises(adapter_errors.EngineRejected, match="502 Bad Gateway"):
        adapter_errors.raise_for_status(_response(502, text="<html>502 Bad Gateway</html>"))


def test_an_engine_that_says_nothing_says_so():
    with pytest.raises(adapter_errors.EngineRejected, match=r"\(empty body\)"):
        adapter_errors.raise_for_status(_response(400, text="   "))


def test_a_credential_echoed_back_is_not_reprinted():
    """An engine's error body is not ours and may quote the request that carried a token."""
    with pytest.raises(adapter_errors.EngineRejected) as caught:
        adapter_errors.raise_for_status(
            _response(401, json_body={"detail": "rejected", "api_key": "sk-live-secret"}))

    assert "sk-live-secret" not in str(caught.value)
    assert adapter_errors.REDACTED in str(caught.value)


def test_a_long_body_is_capped_rather_than_pasted_back():
    """A rejected document must not be echoed whole into a terminal."""
    with pytest.raises(adapter_errors.EngineRejected) as caught:
        adapter_errors.raise_for_status(_response(400, text="x" * 10_000))

    assert len(str(caught.value)) < adapter_errors.BODY_LIMIT + 200


def test_a_capped_body_keeps_the_end_where_the_reason_is():
    """Engines nest their failures, so the innermost reason sits at the very end of the body.

    The head-only cap this replaces printed ``exceeds the server limit of 163`` -- the number that
    was the entire finding, cut in half.
    """
    reason = "prompt is 22596 tokens, which exceeds the server limit of 16384"
    with pytest.raises(adapter_errors.EngineRejected) as caught:
        adapter_errors.raise_for_status(
            _response(400, json_body={"error": "llm: extraction failed: " + "x" * 10_000 + reason}))

    assert reason in str(caught.value)
    assert "chars elided" in str(caught.value)


def test_a_body_within_the_cap_is_quoted_whole():
    """The elision marker must not appear on a body that was never truncated."""
    with pytest.raises(adapter_errors.EngineRejected) as caught:
        adapter_errors.raise_for_status(_response(400, json_body={"error": "y" * 1_500}))

    assert "elided" not in str(caught.value)
    assert "y" * 1_500 in str(caught.value)


class _RefusingClient:
    """An engine that is healthy until asked to ingest, then answers with the 400 that ended the
    2026-08-13 BEAM run. Only the ingest path refuses -- prepare's reset must still succeed, or the
    test would prove the wrong call surfaces the reason."""

    def __init__(self) -> None:
        self.base_url = "http://localhost:17350"

    def post(self, path: str, **kwargs) -> httpx.Response:
        request = httpx.Request("POST", f"{self.base_url}{path}")
        if path.endswith("/ingest"):
            return httpx.Response(
                400, json={"error": "conversation exceeds 32000 tokens"}, request=request)
        return httpx.Response(200, json={}, request=request)


def test_the_reason_survives_the_whole_adapter_path(monkeypatch):
    """End to end through a real adapter, not just the helper: this is the exact call that failed
    on unit 4's third document and reported only a status code."""
    from memrank.adapters.atomicmemory import AtomicMemory
    from memrank.core import Document

    adapter = AtomicMemory(base_url="http://localhost:17350")
    monkeypatch.setattr(adapter, "_http", lambda: _RefusingClient())
    adapter.prepare("unit-4")

    with pytest.raises(adapter_errors.EngineRejected, match="exceeds 32000 tokens"):
        adapter.ingest([Document(id="d3", content="a conversation", user_id="u")])


# ---------------------------------------------------------------------------- #
# A timeout: no response, no status, no body -- and still not a memrank bug
# ---------------------------------------------------------------------------- #

class _TimingOutSocket(httpx.BaseTransport):
    """The socket layer, failing deterministically. No port, no clock, no sleep."""

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)


def _timing_out_client(engine: str, timeout_s: float, env_var: str) -> httpx.Client:
    """A real client through the REAL wrapper, with only the socket layer replaced."""
    return adapter_errors.engine_client(
        engine=engine, timeout_s=timeout_s, env_var=env_var,
        transport=_TimingOutSocket(), base_url="http://localhost:8080")


def test_a_timeout_names_the_engine_the_limit_and_the_variable():
    """The failure this exists for: `internal error: ReadTimeout: timed out / this is a bug in
    memrank`, on a run where the engine was working fine and the harness was impatient."""
    client = _timing_out_client("myengine", 60.0, "MYENGINE_TIMEOUT_S")

    with pytest.raises(adapter_errors.EngineTimedOut) as caught:
        client.post("/v1/memories/ingest", json={})

    message = str(caught.value)
    assert "myengine" in message
    assert "60s" in message
    assert "MYENGINE_TIMEOUT_S" in message


def test_a_timeout_is_not_reported_as_a_refusal():
    """`EngineRejected` carries a status; a timeout has none, and conflating them would invite a
    retry policy to read `is_transient` off a response that never existed."""
    client = _timing_out_client("hindsight", 900.0, "HINDSIGHT_TIMEOUT_S")

    with pytest.raises(adapter_errors.EngineTimedOut) as caught:
        client.post("/v1/default/banks/b/memories", json={})

    assert not isinstance(caught.value, adapter_errors.EngineRejected)
    assert caught.value.is_transient is False, "retrying waits the same insufficient time"


def test_a_timeout_is_a_clean_error_not_an_internal_one():
    """The half of the old message that sent readers into memrank's source."""
    from memrank.errors import MemrankError

    client = _timing_out_client("mem0", 60.0, "MEM0_TIMEOUT_S")

    with pytest.raises(MemrankError):
        client.post("/v1/memories/", json={})


# ---------------------------------------------------------------------------- #
# A refused connection: nothing is listening -- a fact about the machine
# ---------------------------------------------------------------------------- #

class _RefusingSocket(httpx.BaseTransport):
    """The socket layer refusing deterministically. No port is opened or probed."""

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("[Errno 61] Connection refused", request=request)


def _refused_client(base_url: str, base_url_env: str | None = "MEM0_HTTP_URL") -> httpx.Client:
    """A real client through the REAL wrapper, with only the socket layer replaced."""
    return adapter_errors.engine_client(
        engine="mem0", timeout_s=60.0, env_var="MEM0_TIMEOUT_S", base_url_env=base_url_env,
        transport=_RefusingSocket(), base_url=base_url)


def test_a_refused_connection_names_the_engine_the_address_and_the_variable():
    """The failure this exists for: `internal error: ConnectError ... this is a bug in memrank`
    for a mem0 nobody had started."""
    from memrank.errors import MemrankError

    with pytest.raises(adapter_errors.EngineUnreachable) as caught:
        _refused_client("http://localhost:8888").get("/configure")

    message = str(caught.value)
    assert isinstance(caught.value, MemrankError)
    assert "'mem0'" in message
    assert "http://localhost:8888/configure" in message
    assert "MEM0_HTTP_URL" in message
    assert "--on local" in message


def test_a_refused_connection_never_quotes_a_credential_in_the_address():
    """A base URL can carry userinfo or a query key; the message says where, never with what."""
    client = _refused_client("http://alice:s3cret@engine.internal:8888?api_key=zzz")

    with pytest.raises(adapter_errors.EngineUnreachable) as caught:
        client.get("/configure")

    message = str(caught.value)
    assert "engine.internal:8888" in message
    for leaked in ("alice", "s3cret", "api_key", "zzz"):
        assert leaked not in message


def test_an_ipv6_address_is_quoted_as_a_usable_url():
    with pytest.raises(adapter_errors.EngineUnreachable) as caught:
        _refused_client("http://u:p@[::1]:8888").get("/configure")

    assert "http://[::1]:8888/configure" in str(caught.value)


def test_an_adapter_without_an_address_variable_names_its_argument():
    with pytest.raises(adapter_errors.EngineUnreachable, match="`base_url` argument"):
        _refused_client("http://localhost:8888", base_url_env=None).get("/configure")


def test_a_programmer_error_in_the_socket_layer_is_not_translated():
    """Only a refused connection means 'nothing is listening'; anything else stays a bug."""
    class _Broken(httpx.BaseTransport):
        def handle_request(self, request: httpx.Request) -> httpx.Response:
            raise TypeError("an adapter bug")

    client = adapter_errors.engine_client(
        engine="mem0", timeout_s=60.0, env_var="MEM0_TIMEOUT_S", transport=_Broken(),
        base_url="http://localhost:8888")

    with pytest.raises(TypeError, match="an adapter bug"):
        client.get("/configure")


def test_every_engine_facing_adapter_builds_its_client_through_the_helper():
    """The chokepoint, enumerated rather than trusted.

    A guard repeated at ~15 call sites is a guard one sibling is missing, so it lives in the
    transport -- which an adapter gets by constructing its client here. A sixth adapter reaching for
    a bare `httpx.Client` fails on the day it is added rather than shipping the old message.
    """
    import pathlib

    adapters = pathlib.Path(adapter_errors.__file__).parent
    offenders = [path.name for path in sorted(adapters.glob("*.py"))
                 if "httpx.Client(" in path.read_text(encoding="utf-8")
                 and path.name != "errors.py"]

    assert offenders == [], (
        f"{offenders} build an httpx client directly, so their timeouts will be reported as "
        f"'a bug in memrank'; use adapter_errors.engine_client()")
