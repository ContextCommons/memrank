"""Login is required, a judged run needs the org's saved key, and an upload is one PUT."""

from __future__ import annotations

import json

import httpx
import pytest

from memrank.accounts.run_secrets import OrgSecretsError
from memrank.errors import ActionRequired
from memrank.loop import upload as up
from memrank.loop.judge_key import JudgeKeyMissing
from memrank.placement import run_api_client
from memrank.placement.run_api_client import RunApiError
from tests.loop.conftest import ACCOUNT, WHOAMI
from tests.service.test_scoring import recorded

RUN = "20260927-010203__locomo__abcdef"


def test_no_session_is_refused_with_the_login_command(monkeypatch):
    def no_session():
        raise RunApiError("not signed in -- run `memrank auth login`", code="no_session")

    monkeypatch.setattr(run_api_client, "authenticated_client", no_session)
    with pytest.raises(up.LoginRequired, match="(?s)needs a login.*memrank auth login"):
        up.connect(judge=False)


def test_no_organisation_is_refused(monkeypatch):
    monkeypatch.setattr(run_api_client, "authenticated_client", lambda: object())
    monkeypatch.setattr(up.settings, "get", lambda key: None)
    with pytest.raises(up.LoginRequired, match="No organisation"):
        up.connect(judge=False)


def api(secrets: dict[str, str] | int = 200,
        bodies: list[dict] | None = None) -> tuple[httpx.Client, list[str]]:
    """The runs API: ``secrets`` is the org's saved credentials, or a refusal's status;
    ``bodies`` collects every PUT's body."""
    calls: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        calls.append(f"{request.method} {request.url.path}")
        if request.method == "PUT" and bodies is not None:
            bodies.append(json.loads(request.content))
        if request.url.path.endswith("/answers"):
            return httpx.Response(200, json={"answers": "answers.json"})
        if request.url.path.endswith("/secrets/resolved"):
            return httpx.Response(secrets, json={}) if isinstance(secrets, int) \
                else httpx.Response(200, json=secrets)
        if request.url.path == "/whoami":
            return httpx.Response(200, json=WHOAMI)
        if request.method == "PUT":
            assert json.loads(request.content)["record"]["kind"] == "agent"
            return httpx.Response(200, json={"id": RUN, "url": "U"})
        return httpx.Response(200, json={"runs": []})

    return httpx.Client(base_url="http://api", transport=httpx.MockTransport(handle)), calls


@pytest.fixture
def signed_in(monkeypatch):
    def with_api(secrets: dict[str, str] | int) -> list[str]:
        http, calls = api(secrets)
        monkeypatch.setattr(run_api_client, "authenticated_client", lambda: http)
        monkeypatch.setattr(up.settings, "get", lambda key: "acme")
        return calls

    return with_api


@pytest.mark.parametrize(("secrets", "judge", "key"), [
    ({"ANTHROPIC_API_KEY": "org-key"}, True, "org-key"),
    ({}, False, None),
], ids=["judged-takes-the-orgs-key", "no-judge-reads-no-secret"])
def test_connect_takes_the_judge_key_from_the_orgs_saved_secrets(signed_in, secrets, judge, key):
    calls = signed_in(secrets)
    assert up.connect(judge=judge).judge_key == key
    assert ("GET /orgs/acme/secrets/resolved" in calls) is judge


@pytest.mark.parametrize(("secrets", "refusal", "message"), [
    ({}, JudgeKeyMissing, "(?s)memrank secrets set ANTHROPIC_API_KEY --org acme.*--no-judge"),
    (403, ActionRequired, "(?s)only an owner of acme.*--org acme.*--no-judge"),
    (401, ActionRequired, "(?s)not accepted.*memrank auth login"),
], ids=["no-key-saved", "role-may-not-read-secrets", "session-rejected"])
def test_a_judged_run_is_refused_before_it_starts(signed_in, secrets, refusal, message):
    signed_in(secrets)
    with pytest.raises(refusal, match=message):
        up.connect(judge=True)


def test_a_failing_secrets_api_is_an_error_not_a_step(signed_in):
    signed_in(503)
    with pytest.raises(OrgSecretsError) as raised:
        up.connect(judge=True)
    assert not isinstance(raised.value, ActionRequired)


def test_an_upload_puts_the_answers_then_the_summary_that_points_at_them():
    result = recorded().model_copy(update={"run_id": RUN})
    bodies: list[dict] = []
    http, calls = api(bodies=bodies)
    url = up.upload(up.Hosted(http=http, org="acme", account=ACCOUNT), result, lambda line: None)
    assert url == "U"
    assert calls == [f"PUT /orgs/acme/runs/{RUN}/answers", f"PUT /orgs/acme/runs/{RUN}"]
    answers, synced = bodies
    assert answers == {"cases": [c.model_dump(mode="json") for c in result.cases]}
    summary = synced["record"]["result"]
    assert synced["record"]["answers"] == "answers.json"
    assert [c["questions"] for c in summary["cases"]] == [[] for _ in result.cases]
    assert summary["score"] == result.score.model_dump(mode="json")
