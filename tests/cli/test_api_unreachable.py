"""An API that does not answer is named, with where its address came from -- never "a bug"."""

from __future__ import annotations

import httpx
import pytest

from memrank import runner, settings
from memrank.api_client import ApiUnreachable, api_client
from memrank.placement import run_api_client


def refusing(exc: type[httpx.TransportError]) -> httpx.MockTransport:
    def handle(request: httpx.Request) -> httpx.Response:
        raise exc("[Errno 61] Connection refused", request=request)

    return httpx.MockTransport(handle)


#: (case, how the address is set, the failure, what the message must say)
CASES = [
    ("env-refused", "env", httpx.ConnectError,
     ("from MEMRANK_API_URL in the environment", "nothing answered there", "ConnectError")),
    ("file-timeout", "file", httpx.ReadTimeout,
     ("from api.url in ", "did not answer in time", "ReadTimeout")),
    ("default-refused", "default", httpx.ConnectError,
     ("the built-in default", "https://api.memrank.ai")),
]


@pytest.mark.parametrize(("where", "failure", "says"), [c[1:] for c in CASES],
                         ids=[c[0] for c in CASES])
def test_an_unreachable_api_names_the_address_and_its_source(monkeypatch, tmp_path, where,
                                                             failure, says):
    monkeypatch.setenv("MEMRANK_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv("MEMRANK_API_URL", raising=False)
    if where == "env":
        monkeypatch.setenv("MEMRANK_API_URL", "http://127.0.0.1:9")
    elif where == "file":
        settings.put("api.url", "http://127.0.0.1:9")
    http = api_client(timeout=1, token="mrk_x", transport=refusing(failure))
    with pytest.raises(ApiUnreachable) as unreached:
        run_api_client.list_runs(http, "acme", limit=1)
    message = str(unreached.value)
    assert message.startswith("cannot reach the memrank API at ")
    assert "GET /orgs/acme/runs: " in message and "/orgs/acme/runs (" not in message
    assert all(part in message for part in says), message
    assert "bug" not in message and "mrk_x" not in message


def test_it_renders_as_a_failure_not_an_internal_error(monkeypatch, capsys):
    exc = ApiUnreachable("cannot reach the memrank API at http://127.0.0.1:9 (...)")
    code = runner._exit_with(exc).code
    shown = capsys.readouterr().err
    assert code == 1 and shown.startswith("error: cannot reach") and "internal" not in shown


def test_a_5xx_is_the_apis_side_and_keeps_what_it_said():
    def failing(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"detail": {"code": "db_down", "message": "pool empty"}})

    http = httpx.Client(base_url="http://api", transport=httpx.MockTransport(failing))
    with pytest.raises(run_api_client.RunApiError) as failed:
        run_api_client.list_runs(http, "acme", limit=1)
    lead, said = str(failed.value).split("\n", 1)
    assert "failed on its side (503); nothing in your command caused it" in lead
    assert said == "  what the API said: pool empty" and failed.value.code == "db_down"
