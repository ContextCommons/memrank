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
"""``memrank auth`` at the CLI boundary -- the noun-space and its logged-out behaviour.

The credential store is stubbed rather than mocked away: these tests exercise the real
command, and never touch the developer's OS keyring (the FakeKeyring reasoning in
tests/internal/accounts/test_credentials.py).
"""
from __future__ import annotations

import pytest
import typer
from typer.testing import CliRunner

from memrank import settings
from memrank.cli import auth as auth_cli
from memrank.runner import app

runner = CliRunner()


class SignedOutStore:
    """A credential store holding nothing, however the real one would have been configured."""

    def load(self) -> None:
        return None


def test_auth_groups_the_three_verbs():
    """Identity is one noun-space: login, logout, status."""
    result = runner.invoke(app, ["auth", "--help"])
    assert result.exit_code == 0
    for verb in ("login", "logout", "status"):
        assert verb in result.output


def test_status_signed_out_points_at_login(monkeypatch):
    """Signed out is an answer, not a crash -- and it names the command that fixes it."""
    monkeypatch.setattr(auth_cli, "CredentialStore", SignedOutStore)
    result = runner.invoke(app, ["auth", "status"])
    assert result.exit_code == 1
    assert "auth login" in result.output


def test_status_is_reachable_only_under_auth():
    """`memrank status <id>` still means run status -- the flat verb was not stolen."""
    result = runner.invoke(app, ["status", "--help"])
    assert result.exit_code == 0
    assert "run" in result.output.lower()


# --- login ends configured, not merely authenticated -------------------------------------- #

class SignedInStore:
    """A credential store already holding a token."""

    def load(self) -> str:
        return "tok"

    def save(self, token: str) -> None:
        return None


def _whoami(orgs, expires_at="2026-09-01T00:00:00+00:00", default=None):
    """A stand-in httpx client whose /whoami answers with these memberships.

    ``default`` is the slug the server declares as the caller's personal org -- the field the
    CLI configures itself from. It defaults to the first membership, which is what a real
    deployment answers for a user with exactly one org.
    """
    personal = default if default is not None else (orgs[0] if orgs else None)

    class _Response:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"login": "ada", "email": "ada@e.com", "expires_at": expires_at,
                    "orgs": [{"slug": s, "name": s.title(), "role": "member",
                              "is_default": s == personal} for s in orgs]}

    class _Client:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def get(self, path):
            return _Response()

    return lambda token: _Client()


@pytest.fixture
def settings_store(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMRANK_CONFIG_DIR", str(tmp_path))
    for spec in settings.SETTINGS:
        monkeypatch.delenv(spec.env, raising=False)
    return tmp_path


@pytest.fixture
def never_asks(monkeypatch):
    """Any prompt at all during login is the defect these tests exist to prevent."""
    def _explode(*args, **kwargs):
        raise AssertionError("login asked a question")

    monkeypatch.setattr(typer, "prompt", _explode)
    monkeypatch.setattr(typer, "confirm", _explode)


def test_a_single_membership_is_configured_without_asking(settings_store, never_asks,
                                                          monkeypatch):
    """The golden path's whole point: after login, submitting needs no routing flags."""
    monkeypatch.setattr(auth_cli, "_authenticated_client", _whoami(["atomicstrata"]))
    auth_cli._configure_after_login("tok")

    assert settings.get("defaults.org") == "atomicstrata"
    assert settings.get("defaults.on") == "cloud"


def test_several_memberships_take_the_server_declared_default(settings_store, never_asks,
                                                              monkeypatch):
    """Belonging to a team org as well as your personal one is not a question to answer."""
    monkeypatch.setattr(auth_cli, "_authenticated_client",
                        _whoami(["acme", "globex"], default="globex"))
    auth_cli._configure_after_login("tok")

    assert settings.get("defaults.org") == "globex"


def test_a_second_login_leaves_an_existing_default_alone(settings_store, never_asks,
                                                         monkeypatch):
    """Only the first login configures: re-authenticating must not undo a deliberate choice."""
    settings.put("defaults.org", "acme")
    monkeypatch.setattr(auth_cli, "_authenticated_client",
                        _whoami(["acme", "globex"], default="globex"))
    auth_cli._configure_after_login("tok")

    assert settings.get("defaults.org") == "acme"


def test_an_env_default_is_reported_as_the_one_that_wins(settings_store, never_asks,
                                                         monkeypatch, capsys):
    """MEMRANK_ORG shadows the file, so writing one would configure a value nothing reads."""
    monkeypatch.setenv("MEMRANK_ORG", "from-env")
    monkeypatch.setattr(auth_cli, "_authenticated_client", _whoami(["globex"]))
    auth_cli._configure_after_login("tok")

    # style.say narrates on stderr, so stdout stays a machine-readable channel.
    narration = capsys.readouterr().err
    assert "from-env" in narration and "MEMRANK_ORG" in narration


def test_no_membership_configures_no_org_and_names_the_fix(settings_store, never_asks,
                                                           monkeypatch, capsys):
    """Signing in with no org is a real state; it must be said, not silently half-configured."""
    monkeypatch.setattr(auth_cli, "_authenticated_client", _whoami([]))
    auth_cli._configure_after_login("tok")

    assert settings.get("defaults.org") is None
    assert settings.get("defaults.on") == "cloud"
    assert "config set defaults.org" in capsys.readouterr().err


def test_status_reports_session_expiry_and_the_active_default(settings_store, monkeypatch):
    monkeypatch.setattr(auth_cli, "CredentialStore", SignedInStore)
    monkeypatch.setattr(auth_cli, "_authenticated_client", _whoami(["acme", "globex"]))
    settings.put("defaults.org", "globex")

    result = runner.invoke(app, ["auth", "status"])
    assert result.exit_code == 0
    assert "2026-09-01" in result.output
    # The default org is a marked ROW now, not a "(default)" suffix -- the marker is what says
    # which one a submission would run under, and it must land on that org and no other.
    rows = {line.split()[-2]: line for line in result.output.splitlines() if "Globex" in line
            or "Acme" in line}
    assert rows["globex"].startswith("▸") and not rows["acme"].startswith("▸")


def test_status_flags_a_default_org_you_cannot_act_on(settings_store, monkeypatch):
    """Otherwise the mismatch surfaces as a 403 at submission, far from its cause."""
    monkeypatch.setattr(auth_cli, "CredentialStore", SignedInStore)
    monkeypatch.setattr(auth_cli, "_authenticated_client", _whoami(["acme"]))
    settings.put("defaults.org", "left-over")

    result = runner.invoke(app, ["auth", "status"])
    assert "left-over" in result.output
    assert "not a member" in result.output


class RefusingStore:
    """A credential store whose backend will not answer -- a denied Keychain prompt."""

    def load(self):
        from memrank.accounts.credentials import CredentialError

        raise CredentialError("the credential store refused to answer (denied)")


def test_a_refused_store_is_one_line_not_a_traceback(monkeypatch):
    """The defect this fixes: a typed error was introduced with no handler, so `auth status`
    printed sixty lines of keyring internals instead of the sentence it carries."""
    monkeypatch.setattr(auth_cli, "CredentialStore", RefusingStore)

    result = runner.invoke(app, ["auth", "status"])

    assert result.exit_code == 1
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert "credential store refused" in result.output
    assert "Traceback" not in result.output
