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
"""``memrank secrets ls / rm --org`` -- the org vault reached through the CLI.

The API is faked at the ``_org_client`` seam (the house style: a fake client, not respx),
so these tests pin the CLI's side of the contract: which routes it calls, what it prints,
and that a value string can never appear in an org listing.
"""
from __future__ import annotations

import pytest
from typer.testing import CliRunner

from memrank.cli import secrets as secrets_cli
from memrank.runner import app

runner = CliRunner()


class _Response:
    def __init__(self, status_code: int, body):
        self.status_code = status_code
        self._body = body
        self.text = str(body)

    def json(self):
        return self._body


class _FakeClient:
    """Context-manager client whose canned response is set per test."""

    def __init__(self, response: _Response, calls: list):
        self._response = response
        self._calls = calls

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get(self, path):
        self._calls.append(("GET", path))
        return self._response

    def delete(self, path):
        self._calls.append(("DELETE", path))
        return self._response


@pytest.fixture
def org_api(monkeypatch):
    """Install a fake org client; tests set ``state['response']`` and read ``state['calls']``."""
    state = {"response": _Response(200, []), "calls": []}
    monkeypatch.setattr(secrets_cli, "_org_client",
                        lambda: _FakeClient(state["response"], state["calls"]))
    return state


def test_wallet_ls_still_lists_the_wallet(wallet):
    wallet.put("OPENAI_API_KEY", "sk-wallet-value")
    result = runner.invoke(app, ["secrets", "ls"])
    assert result.exit_code == 0
    assert "OPENAI_API_KEY" in result.stdout
    assert "sk-wallet-value" not in result.stdout  # masked, as before the rename


def test_org_ls_prints_names_and_backends_never_values(org_api):
    org_api["response"] = _Response(200, [{"name": "ANTHROPIC_API_KEY", "backend": "ssm"},
                                          {"name": "VOYAGE_API_KEY", "backend": "local"}])
    result = runner.invoke(app, ["secrets", "ls", "--org", "acme"])
    assert result.exit_code == 0
    # Backend is its own column now rather than a bracketed suffix; the claim under test is
    # what the row CARRIES -- a name and where it lives, and no value.
    rows = {line.split()[0]: line.split()[1] for line in result.stdout.splitlines()[1:]}
    assert rows == {"ANTHROPIC_API_KEY": "ssm", "VOYAGE_API_KEY": "local"}
    assert org_api["calls"] == [("GET", "/orgs/acme/secrets")]


def test_org_ls_with_an_empty_vault_names_the_fix(org_api):
    """An empty listing prints nothing on stdout -- the hint is narration, so `| wc -l` is 0."""
    result = runner.invoke(app, ["secrets", "ls", "--org", "acme"])
    assert result.exit_code == 0
    assert result.stdout == ""
    assert "secrets set" in result.stderr and "--org acme" in result.stderr


def test_logged_out_org_ls_is_a_sign_in_step_not_an_error(monkeypatch, capsys):
    from memrank import runner as memrank_runner
    from memrank.placement import run_api_client

    def refuse():
        raise run_api_client.RunApiError("not signed in -- run `memrank auth login`",
                                         code="no_session")

    monkeypatch.setattr(run_api_client, "authenticated_client", refuse)
    monkeypatch.setattr("sys.argv", ["memrank", "secrets", "ls", "--org", "acme"])
    with pytest.raises(SystemExit) as exited:
        memrank_runner.main()
    shown = capsys.readouterr().err
    assert exited.value.code == 1
    assert "\n  memrank auth login" in shown and "error:" not in shown


def test_org_rm_reports_removal_and_the_ssm_retention_note(org_api):
    org_api["response"] = _Response(200, {"deleted": "ANTHROPIC_API_KEY", "backend": "ssm"})
    result = runner.invoke(app, ["secrets", "rm", "ANTHROPIC_API_KEY", "--org", "acme"])
    assert result.exit_code == 0
    assert "removed ANTHROPIC_API_KEY for org acme" in result.stderr
    assert "retained" in result.output  # the SSM parameter outlives the reference -- say so
    assert org_api["calls"] == [("DELETE", "/orgs/acme/secrets/ANTHROPIC_API_KEY")]


def test_org_rm_of_a_local_backed_secret_has_no_ssm_note(org_api):
    org_api["response"] = _Response(200, {"deleted": "VOYAGE_API_KEY", "backend": "local"})
    result = runner.invoke(app, ["secrets", "rm", "VOYAGE_API_KEY", "--org", "acme"])
    assert result.exit_code == 0
    assert "retained" not in result.output


def test_org_rm_of_an_absent_name_fails_with_the_servers_message(org_api):
    org_api["response"] = _Response(404, {"detail": "org has no secret named 'NOPE'"})
    result = runner.invoke(app, ["secrets", "rm", "NOPE", "--org", "acme"])
    assert result.exit_code == 1
    assert "org has no secret named" in result.output


@pytest.mark.parametrize("argv", [["secrets", "import-env", "--org", "acme"],
                                  ["secrets", "path", "--org", "acme"]])
def test_wallet_medium_utilities_refuse_org(argv):
    """import-env and path are utilities of the wallet's storage medium, not org verbs."""
    result = runner.invoke(app, argv)
    assert result.exit_code == 2
    assert "No such option" in result.output
