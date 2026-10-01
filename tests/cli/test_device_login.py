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
"""The client half of a browserless login: what it prints, and when it stops.

``sleep`` is injected and counted rather than waited on. A polling loop tested against a real
clock is the timing-dependent test this repo does not allow, and it would also take the
server's full ten-minute flow lifetime to assert the give-up case once.

The refusal tests matter as much as the happy path. A loop that treats every non-success as
"keep waiting" spins until the flow expires and then reports a timeout, hiding whatever the
server actually said -- which is precisely the failure the browserless flow exists to end.
"""
from __future__ import annotations

import pytest

from memrank.accounts import login_flow
from memrank.accounts.loopback import LoginError
from memrank.cli import auth as auth_cli

_TOKEN = {"token": "mrk_x", "token_id": "1", "expires_at": "2026-09-12T00:00:00+00:00"}


class _Response:
    def __init__(self, status_code: int, body: dict) -> None:
        self.status_code = status_code
        self._body = body

    @property
    def text(self) -> str:
        return str(self._body)

    def json(self) -> dict:
        return self._body


class FakeApi:
    """An API that answers ``pending`` a set number of times, then whatever comes next.

    Records every device code it was polled with, so a test can assert the client presents the
    one it was issued rather than anything it saw in the URL.
    """

    def __init__(self, *, pending: int, then: _Response | None = None,
                 expires_in: int = 60, interval: int = 5) -> None:
        self._pending = pending
        self._then = then if then is not None else _Response(200, _TOKEN)
        self._flow = {"device_code": "mrk_dev", "verification_uri": "https://gui/signin?flow=st",
                      "expires_in": expires_in, "interval": interval}
        self.polled_with: list[str] = []

    def post(self, path: str, json: dict) -> _Response:
        if path == "/auth/device/code":
            return _Response(200, self._flow)
        self.polled_with.append(json["device_code"])
        if self._pending > 0:
            self._pending -= 1
            return _Response(202, {"status": "authorization_pending"})
        return self._then


def _run(api, announced=None, slept=None):
    return login_flow.perform_device_login(
        api, announce=(announced if announced is not None else []).append,
        sleep=(slept if slept is not None else []).append)


def test_it_polls_until_the_browser_half_is_approved():
    api = FakeApi(pending=3)

    assert _run(api) == _TOKEN
    assert len(api.polled_with) == 4


def test_it_prints_the_url_a_human_must_open():
    """The whole point on a headless box: something to copy out of the terminal."""
    announced = []

    _run(FakeApi(pending=0), announced=announced)

    assert announced == ["https://gui/signin?flow=st"]


def test_it_waits_the_interval_the_server_asked_for():
    """The server sizes the poll's rate limit against this, so the client may not choose it."""
    slept = []

    _run(FakeApi(pending=2, interval=7), slept=slept)

    assert slept == [7, 7]


def test_it_presents_the_device_code_it_was_issued():
    api = FakeApi(pending=1)

    _run(api)

    assert set(api.polled_with) == {"mrk_dev"}


def test_it_gives_up_once_the_flow_can_no_longer_be_approved():
    """Bounded by the server's advertised lifetime, not by an unbounded wait."""
    api = FakeApi(pending=999, expires_in=20, interval=5)

    with pytest.raises(LoginError, match="timed out"):
        _run(api)

    assert len(api.polled_with) == 5


def test_a_refusal_stops_the_loop_instead_of_being_treated_as_pending():
    api = FakeApi(pending=1, then=_Response(400, {"detail": "code_verifier does not match"}))

    with pytest.raises(LoginError, match="code_verifier"):
        _run(api)


def test_a_flow_that_cannot_even_begin_says_so():
    class Refusing:
        def post(self, path, json):
            return _Response(400, {"detail": "no web sign-in page"})

    with pytest.raises(LoginError, match="no web sign-in page"):
        _run(Refusing())


def test_a_machine_with_no_browser_takes_the_browserless_path(monkeypatch):
    """Nobody over SSH should have to already know that --no-browser exists."""
    monkeypatch.setattr(auth_cli, "_has_browser", lambda: False)
    taken = []
    monkeypatch.setattr(auth_cli.login_flow, "perform_device_login",
                        lambda http, **kw: taken.append("device") or _TOKEN)
    monkeypatch.setattr(auth_cli.login_flow, "perform_login",
                        lambda http, **kw: pytest.fail("opened a browser that does not exist"))
    monkeypatch.setattr(auth_cli, "_client", lambda: _NullClient())
    monkeypatch.setattr(auth_cli, "CredentialStore", _NullStore)
    monkeypatch.setattr(auth_cli, "_configure_after_login", lambda token: None)

    auth_cli.login(no_browser=False)

    assert taken == ["device"]


def test_the_flag_forces_it_even_where_a_browser_exists(monkeypatch):
    """For the case detection cannot see: a browser that is present but will not open."""
    monkeypatch.setattr(auth_cli, "_has_browser", lambda: True)
    taken = []
    monkeypatch.setattr(auth_cli.login_flow, "perform_device_login",
                        lambda http, **kw: taken.append("device") or _TOKEN)
    monkeypatch.setattr(auth_cli, "_client", lambda: _NullClient())
    monkeypatch.setattr(auth_cli, "CredentialStore", _NullStore)
    monkeypatch.setattr(auth_cli, "_configure_after_login", lambda token: None)

    auth_cli.login(no_browser=True)

    assert taken == ["device"]


class _NullClient:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _NullStore:
    def save(self, token: str) -> None:
        return None


def test_the_browser_login_prints_its_link_as_well_as_opening_it(monkeypatch, capsys):
    """A coding agent cannot see the browser window, so it needs the link to hand on."""
    opened = []
    monkeypatch.setattr(auth_cli.webbrowser, "open", opened.append)

    auth_cli._open_and_announce("http://api/auth/authorize?state=s")

    assert opened == ["http://api/auth/authorize?state=s"]
    assert "http://api/auth/authorize?state=s" in capsys.readouterr().err
