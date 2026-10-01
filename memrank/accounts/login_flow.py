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
"""The client half of the PKCE login, as one function with its collaborators injected.

Kept separate from the Typer command so the flow can be tested against a real API, a real
socket, and real PKCE -- with only the browser and the upstream provider scripted. A flow
buried inside a CLI command is a flow that only ever gets tested by hand.

Order of operations matters and is deliberate: the loopback port is opened *before* the
browser is sent anywhere, so a user who authorizes instantly cannot arrive before anything
is listening.
"""
from __future__ import annotations

import secrets
import urllib.parse

from memrank.accounts import loopback, pkce


def perform_login(http, *, open_browser) -> dict[str, str]:
    """Run the browser login and redeem the result for a CLI token.

    Args:
        http: An ``httpx.Client``-shaped object bound to the API (the real CLI passes an
            ``httpx.Client``; tests pass a ``TestClient``, which has the same surface).
        open_browser: Called with the authorization URL. The real CLI hands this to
            ``webbrowser.open``.

    Returns:
        The API's token response: ``token``, ``token_id``, and ``expires_at``.

    Raises:
        loopback.LoginError: If the browser half failed, was refused, or never returned.
        RuntimeError: If the API refused to redeem the code.
    """
    verifier = pkce.make_verifier()
    # A CSRF value distinct from the PKCE verifier. They answer different questions: state
    # proves this callback belongs to the flow we started, the verifier proves we are the
    # client that started it.
    state = secrets.token_urlsafe(16)

    with loopback.LoopbackReceiver(state=state) as receiver:
        query = urllib.parse.urlencode({
            "redirect_uri": receiver.redirect_uri,
            "code_challenge": pkce.challenge_for(verifier),
            "state": state,
        })
        open_browser(f"{str(http.base_url).rstrip('/')}/auth/authorize?{query}")
        code = receiver.wait()

    response = http.post("/auth/token", json={"code": code, "code_verifier": verifier})
    if response.status_code != 200:
        raise RuntimeError(f"could not complete login: {response.status_code} {response.text}")
    return response.json()


def perform_device_login(http, *, announce, sleep) -> dict[str, str]:
    """Run a login for a machine with no browser, and poll until it is approved.

    The mirror image of :func:`perform_login`: no port is opened, because nothing will call
    back. The user carries the URL to a browser somewhere else, and this polls until the server
    says the browser half is done.

    Args:
        http: An ``httpx.Client``-shaped object bound to the API, as in :func:`perform_login`.
        announce: Called with the URL a human must open. Injected for the same reason
            ``open_browser`` is -- so the flow is testable without a terminal.
        sleep: Called with a number of seconds between polls. Injected so tests drive the loop
            by call count rather than by elapsed time; nothing here may depend on a wall clock.

    Returns:
        The API's token response: ``token``, ``token_id``, and ``expires_at``.

    Raises:
        loopback.LoginError: If the flow expired before anyone approved it, or the server
            refused to finish it. Shares the exception with the loopback flow because the
            caller's handling is identical -- the difference between the two is not the user's
            problem to sort out.
    """
    verifier = pkce.make_verifier()
    started = http.post("/auth/device/code",
                        json={"code_challenge": pkce.challenge_for(verifier)})
    if started.status_code != 200:
        raise loopback.LoginError(
            f"could not begin the login: {started.status_code} {started.text}")
    flow = started.json()
    announce(flow["verification_uri"])

    # Bounded by the server's own advertised lifetime rather than by a clock read here: the
    # flow expires server-side, and a client that kept polling past it would only collect the
    # same refusal forever. One extra attempt so the last poll lands after the final sleep.
    interval = flow["interval"]
    for attempt in range(int(flow["expires_in"] / interval) + 1):
        if attempt:
            sleep(interval)
        polled = http.post("/auth/device/token",
                           json={"device_code": flow["device_code"], "code_verifier": verifier})
        if polled.status_code == 200:
            return polled.json()
        if polled.status_code != 202:
            raise loopback.LoginError(f"login failed: {polled.text}")
    raise loopback.LoginError("timed out waiting for the login to be approved in a browser")
