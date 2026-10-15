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
"""``memrank endpoints add / ls / rm`` (ATO-2503).

The API is faked at the ``_org_client`` seam, as ``tests/secrets/test_org_cli.py`` does. What is
pinned: the routes called and in what order, that a refused key saves nothing, and that the
base URL never reaches the terminal.
"""
from __future__ import annotations

import httpx
import pytest
from typer.testing import CliRunner

from memrank.accounts import org_endpoints
from memrank.cli import endpoints as endpoints_cli
from memrank.errors import ActionRequired
from memrank.runner import app

runner = CliRunner()

_BASE = "https://router.internal.example/v1"
_SAVED = {"name": "router", "adapter": "openai", "key_ref": "ATOMIC_ROUTER_API_KEY",
          "models_listing": True, "id": "e1", "revision": 1, "fingerprint": "e1@1"}
_ADD = ["endpoints", "add", "router", "--adapter", "openai", "--base-url", _BASE,
        "--key-ref", "ATOMIC_ROUTER_API_KEY", "--org", "acme"]


class _Response:
    def __init__(self, status_code: int, body):
        self.status_code = status_code
        self._body = body
        self.text = str(body)

    def json(self):
        return self._body


class _FakeApi:
    """A context-manager client answering by ``(method, path)``; records every call."""

    def __init__(self, answers: dict):
        self.answers = answers
        self.calls: list = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def _answer(self, method, path, body=None):
        self.calls.append((method, path, body))
        return self.answers[(method, path)]

    def get(self, path):
        return self._answer("GET", path)

    def post(self, path, json):
        return self._answer("POST", path, json)

    def delete(self, path):
        return self._answer("DELETE", path)


@pytest.fixture
def api(monkeypatch):
    fake = _FakeApi({
        ("GET", "/orgs/acme/secrets"): _Response(200, []),
        ("POST", "/orgs/acme/secrets"): _Response(200, {"stored": "ATOMIC_ROUTER_API_KEY"}),
        ("POST", "/orgs/acme/endpoints"): _Response(200, _SAVED),
        ("GET", "/orgs/acme/endpoints"): _Response(200, [_SAVED]),
        ("DELETE", "/orgs/acme/endpoints/router"): _Response(200, {"deleted": "router"}),
    })
    monkeypatch.setattr(endpoints_cli, "_org_client", lambda: fake)
    return fake


@pytest.fixture
def listing(monkeypatch):
    """The ``/models`` check, answering ``state['answer']`` (a bool, or an error to raise)."""
    state: dict = {"answer": True, "keys": []}

    def check(*, name, adapter, api_base, key):
        state["keys"].append(key)
        if isinstance(state["answer"], Exception):
            raise state["answer"]
        return state["answer"]

    monkeypatch.setattr(org_endpoints, "has_models_listing", check)
    return state


def test_add_asks_for_a_missing_key_checks_it_then_saves_key_and_endpoint(api, listing):
    result = runner.invoke(app, _ADD, input="sk-router\n")

    assert result.exit_code == 0, result.output
    assert listing["keys"] == ["sk-router"]
    assert [(m, p) for m, p, _ in api.calls] == [
        ("GET", "/orgs/acme/secrets"), ("POST", "/orgs/acme/secrets"),
        ("POST", "/orgs/acme/endpoints")]
    assert api.calls[2][2] == {"name": "router", "adapter": "openai", "api_base": _BASE,
                               "key_ref": "ATOMIC_ROUTER_API_KEY", "models_listing": True}
    assert "e1@1" in result.output
    assert _BASE not in result.output
    assert "sk-router" not in result.output


def test_add_with_a_saved_key_uses_it_without_asking(api, listing):
    api.answers[("GET", "/orgs/acme/secrets")] = _Response(
        200, [{"name": "ATOMIC_ROUTER_API_KEY", "backend": "local"}])
    api.answers[("GET", "/orgs/acme/secrets/resolved")] = _Response(
        200, {"ATOMIC_ROUTER_API_KEY": "sk-saved"})
    listing["answer"] = False

    result = runner.invoke(app, _ADD)

    assert result.exit_code == 0, result.output
    assert listing["keys"] == ["sk-saved"]
    assert ("POST", "/orgs/acme/secrets") not in [(m, p) for m, p, _ in api.calls]
    assert api.calls[-1][2]["models_listing"] is False
    assert "no models listing" in result.output


def test_a_refused_key_saves_nothing(api, listing):
    listing["answer"] = org_endpoints.EndpointError(
        "endpoint 'router' refused the key (401); nothing was saved", status=401)

    result = runner.invoke(app, _ADD, input="sk-wrong\n")

    assert isinstance(result.exception, org_endpoints.EndpointError)
    assert "refused the key" in str(result.exception)
    assert [(m, p) for m, p, _ in api.calls] == [("GET", "/orgs/acme/secrets")]


@pytest.mark.parametrize("swap", [("router", "openai"), (_BASE, "router.internal.example"),
                                  ("ATOMIC_ROUTER_API_KEY", "lower-case")])
def test_an_invalid_entry_is_refused_before_anything_is_asked(api, listing, swap):
    result = runner.invoke(app, [swap[1] if arg == swap[0] else arg for arg in _ADD])

    assert result.exit_code == 2
    assert api.calls == [] and listing["keys"] == []
    assert _BASE not in result.output


def test_ls_shows_the_fingerprint_and_never_a_base_url(api):
    result = runner.invoke(app, ["endpoints", "ls", "--org", "acme"])

    assert result.exit_code == 0, result.output
    assert "router" in result.output and "e1@1" in result.output
    assert "ATOMIC_ROUTER_API_KEY" in result.output


def test_rm_deletes_the_endpoint(api):
    result = runner.invoke(app, ["endpoints", "rm", "router", "--org", "acme"])

    assert result.exit_code == 0, result.output
    assert api.calls == [("DELETE", "/orgs/acme/endpoints/router", None)]


def test_a_member_is_told_to_ask_an_owner(api):
    api.answers[("DELETE", "/orgs/acme/endpoints/router")] = _Response(403, {"detail": "no"})

    result = runner.invoke(app, ["endpoints", "rm", "router", "--org", "acme"])

    assert isinstance(result.exception, ActionRequired)
    assert "Only an owner of acme" in str(result.exception)


def _transport(status: int, seen: list):
    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(status, json={"data": []})
    return httpx.MockTransport(handle)


@pytest.mark.parametrize("status,listed", [(200, True), (404, False), (405, False)])
def test_the_models_check_reads_the_listing(status, listed):
    seen: list = []
    answer = org_endpoints.has_models_listing(name="router", adapter="openai", api_base=_BASE,
                                              key="sk", transport=_transport(status, seen))
    assert answer is listed
    assert str(seen[0].url) == f"{_BASE}/models"
    assert seen[0].headers["authorization"] == "Bearer sk"


def test_an_anthropic_adapter_checks_its_own_listing():
    seen: list = []
    org_endpoints.has_models_listing(name="stub", adapter="anthropic", api_base=_BASE,
                                     key="sk", transport=_transport(200, seen))
    assert str(seen[0].url) == f"{_BASE}/v1/models"
    assert seen[0].headers["x-api-key"] == "sk"


@pytest.mark.parametrize("status", [401, 403, 500])
def test_a_refusal_or_failure_raises_without_the_url(status):
    with pytest.raises(org_endpoints.EndpointError) as raised:
        org_endpoints.has_models_listing(name="router", adapter="openai", api_base=_BASE,
                                         key="sk", transport=_transport(status, []))
    assert raised.value.status == status
    assert _BASE not in str(raised.value) and "router" in str(raised.value)
