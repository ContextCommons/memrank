"""A run's --judge resolves against the organisation's endpoints, with that endpoint's key."""

from __future__ import annotations

import httpx
import pytest

from memrank.errors import ActionRequired
from memrank.loop import upload as up
from memrank.placement import run_api_client
from tests.loop.conftest import WHOAMI

#: The private host an endpoint sits on, which nothing a run prints may name.
HOST = "llm.internal.example"
ROUTER = {"name": "router", "adapter": "openai", "key_ref": "ATOMIC_ROUTER_API_KEY",
          "models_listing": True, "id": "e1", "revision": 3, "fingerprint": "e1@3",
          "api_base": f"https://{HOST}/v1"}


def org_api(endpoints: dict[str, dict], secrets: dict[str, str]) -> tuple[httpx.Client, list]:
    """The org's API: its endpoints by name and its saved secrets; every path asked, in order."""
    asked: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        asked.append(path)
        if path == "/whoami":
            return httpx.Response(200, json=WHOAMI)
        if path.endswith("/resolved") and "/endpoints/" in path:
            name = path.split("/")[-2]
            return httpx.Response(200, json=endpoints[name]) if name in endpoints else \
                httpx.Response(404, json={"detail": f"org has no endpoint named {name!r}"})
        if path.endswith("/endpoints"):
            return httpx.Response(200, json=[{k: v for k, v in row.items() if k != "api_base"}
                                             for row in endpoints.values()])
        if path.endswith("/secrets/resolved"):
            return httpx.Response(200, json=secrets)
        return httpx.Response(200, json={"runs": []})

    return httpx.Client(base_url="http://api", transport=httpx.MockTransport(handle)), asked


@pytest.fixture
def signed_in(monkeypatch):
    def with_org(endpoints: dict[str, dict], secrets: dict[str, str]) -> list[str]:
        http, asked = org_api(endpoints, secrets)
        monkeypatch.setattr(run_api_client, "authenticated_client", lambda: http)
        monkeypatch.setattr(up.settings, "get", lambda key: "acme")
        return asked

    return with_org


def test_an_endpoint_judge_resolves_from_the_org_with_its_own_key(signed_in):
    signed_in({"router": ROUTER}, {"ATOMIC_ROUTER_API_KEY": "rk", "ANTHROPIC_API_KEY": "ak"})
    checked: list[tuple] = []
    hosted = up.connect(judge="router:bedrock/claude-haiku-4-5",
                        check=lambda target, key, listing: checked.append((key, listing)))
    assert hosted.judge_key == "rk" and checked == [("rk", True)]
    judge = hosted.judge
    assert judge is not None and judge.endpoint is not None
    assert (judge.adapter, judge.model, judge.key_name, judge.fingerprint) == (
        "openai", "bedrock/claude-haiku-4-5", "ATOMIC_ROUTER_API_KEY", "e1@3")


def test_an_unknown_endpoint_is_refused_naming_the_orgs_endpoints_before_any_key(signed_in):
    asked = signed_in({"router": ROUTER}, {})
    with pytest.raises(ActionRequired) as refused:
        up.connect(judge="nosuch:model", check=lambda *args: None)
    assert "acme's endpoints: router" in refused.value.statement
    assert not any(path.endswith("/secrets/resolved") for path in asked)
    assert HOST not in str(refused.value)


def test_a_model_the_key_cannot_reach_stops_the_run_before_it_starts(signed_in):
    signed_in({"router": ROUTER}, {"ATOMIC_ROUTER_API_KEY": "rk"})
    with pytest.raises(ActionRequired) as refused:
        up.connect(judge="router:nope",
                   check=lambda *args: "its models listing does not include 'nope'")
    said = refused.value.statement
    assert said.startswith("The endpoint 'router' will not judge with 'nope'")
    assert "Nothing has run" in said and HOST not in str(refused.value)
