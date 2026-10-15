"""The judge's model is checked before a run, for free wherever the provider allows it."""

from __future__ import annotations

import httpx
import pytest

from memrank.judging.providers import ADAPTERS, Endpoint, ProviderError, resolve
from memrank.judging.providers.check import JudgeUnreachable, model_refusal

HOST = "llm.internal.example"
ROUTER = Endpoint(name="router", adapter="openai", api_base=f"https://{HOST}/v1",
                  key_ref="ROUTER_KEY", fingerprint="e1@1")


def answering(status: int, body: dict | None = None) -> tuple[httpx.MockTransport, list]:
    seen: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(status, json=body or {})

    return httpx.MockTransport(handle), seen


@pytest.mark.parametrize(("spec", "url", "auth"), [
    ("anthropic:claude-haiku-4-5", "https://api.anthropic.com/v1/models/claude-haiku-4-5",
     ("x-api-key", "k")),
    ("openai:gpt-5", "https://api.openai.com/v1/models/gpt-5", ("authorization", "Bearer k")),
], ids=["anthropic", "openai"])
def test_a_built_in_provider_is_asked_for_the_one_model(spec, url, auth):
    transport, seen = answering(200, {"id": "x"})
    assert model_refusal(resolve(spec), "k", transport=transport) is None
    assert str(seen[0].url) == url and seen[0].method == "GET"
    assert seen[0].headers[auth[0]] == auth[1]


@pytest.mark.parametrize(("status", "said"), [
    (401, "the key was not accepted (401)"),
    (403, "the key was not accepted (403)"),
    (404, "it has no model 'gpt-5'"),
])
def test_a_refused_key_or_unknown_model_is_an_answer(status, said):
    transport, _ = answering(status)
    assert model_refusal(resolve("openai:gpt-5"), "k", transport=transport) == said


def test_an_outage_is_raised_not_answered():
    transport, _ = answering(503)
    with pytest.raises(ProviderError):
        model_refusal(resolve("openai:gpt-5"), "k", transport=transport)


@pytest.mark.parametrize(("listed", "said"), [
    (["bedrock/claude-haiku-4-5"], None),
    (["bedrock/other"], "its models listing does not include 'bedrock/claude-haiku-4-5'"),
], ids=["listed", "not-listed"])
def test_an_endpoint_must_list_the_model(listed, said):
    transport, seen = answering(200, {"data": [{"id": model} for model in listed]})
    target = resolve("router:bedrock/claude-haiku-4-5", (ROUTER,))
    assert model_refusal(target, "k", transport=transport) == said
    assert str(seen[0].url) == f"https://{HOST}/v1/models"


def test_an_endpoint_without_a_listing_is_checked_with_one_token(monkeypatch):
    sent: list[int] = []

    def openai(target, api_key, *, max_tokens):
        sent.append(max_tokens)
        return lambda model, system, user, sampling: "OK"

    monkeypatch.setitem(ADAPTERS, "openai", openai)
    target = resolve("router:bedrock/claude-haiku-4-5", (ROUTER,))
    assert model_refusal(target, "k", models_listing=False) is None and sent == [1]


def test_an_unreachable_endpoint_is_raised_without_its_url():
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"cannot connect to {HOST}", request=request)

    target = resolve("router:bedrock/claude-haiku-4-5", (ROUTER,))
    with pytest.raises(JudgeUnreachable) as raised:
        model_refusal(target, "k", transport=httpx.MockTransport(refuse))
    assert HOST not in str(raised.value) and "the endpoint 'router'" in str(raised.value)
