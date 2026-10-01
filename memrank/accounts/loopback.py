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
"""The CLI's one-shot loopback callback server for the PKCE login flow.

RFC 8252 endorses this shape -- a native app receives its authorization code on a loopback
redirect -- because the response never leaves the machine. Two details carry that guarantee
and both are easy to get wrong:

- **Bind 127.0.0.1, never 0.0.0.0.** Listening on all interfaces would let anything on the
  local network receive the code. This is the most common mistake in the pattern.
- **Port 0.** The kernel assigns a free port, so two concurrent logins cannot collide and
  nothing needs a fixed port reserved. The API accepts any loopback port precisely so this
  works, which is why the CLI talks to our API rather than to GitHub directly.

The port is open only for the duration of one login and closes on exit from the context
manager, whether the flow succeeded or not.
"""
from __future__ import annotations

import http.server
import socketserver
import threading
import urllib.parse

from memrank.errors import MemrankError

#: A generous safety valve so a browser that never returns cannot hang the CLI forever.
#: Nothing depends on this for correctness -- the flow is driven by the callback request.
WAIT_TIMEOUT_SECONDS = 300

_SUCCESS_PAGE = b"""<!doctype html><html><body style="font-family:system-ui;padding:3rem">
<h2>You're signed in to memrank.</h2><p>You can close this tab and return to your terminal.</p>
</body></html>"""


#: How often ``shutdown()`` is noticed. The default 0.5s is a visible pause at the end of an
#: otherwise instant login.
_SHUTDOWN_POLL_SECONDS = 0.05


class LoginError(MemrankError):
    """Raised when the browser half of the login failed. Never swallowed."""


class _QuietHTTPServer(http.server.HTTPServer):
    """An HTTPServer that does not resolve its own hostname when binding.

    ``HTTPServer.server_bind`` calls ``socket.getfqdn()`` to populate ``server_name``. On a
    machine whose DNS is slow or unreachable -- a laptop on a captive-portal network, a VPN
    mid-handshake -- that blocks for tens of seconds before the browser is ever opened, and
    the user sees ``memrank auth login`` hang with no output. Nothing here uses ``server_name``,
    so the lookup is pure cost.
    """

    def server_bind(self) -> None:
        """Bind without the reverse-DNS round trip."""
        # Deliberately skips HTTPServer.server_bind and calls its parent's. Reached through
        # `socketserver` itself: `http.server` imports the module but does not re-export it.
        socketserver.TCPServer.server_bind(self)
        self.server_name = "127.0.0.1"
        self.server_port = self.server_address[1]


class _CallbackHandler(http.server.BaseHTTPRequestHandler):
    """Handles exactly one callback, then signals the waiting CLI."""

    def do_GET(self) -> None:  # noqa: N802 -- name fixed by BaseHTTPRequestHandler
        """Accept the authorization code iff the state matches the one we issued."""
        query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        receiver = self.server.receiver  # type: ignore[attr-defined]
        state = query.get("state", [None])[0]
        if state != receiver.state:
            # Refused *before* anything is recorded: an attacker-initiated flow must not be
            # able to plant its code in a CLI the user started for a different login.
            self.send_error(400, "state mismatch")
            return
        if "error" in query:
            receiver.failure = query["error"][0]
            self.send_error(400, f"login failed: {receiver.failure}")
            receiver.done.set()
            return
        receiver.code = query.get("code", [None])[0]
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(_SUCCESS_PAGE)
        receiver.done.set()

    def log_message(self, *args) -> None:
        """Silence the default stderr access log -- this is a CLI, not a server."""


class LoopbackReceiver:
    """A one-shot ``http://127.0.0.1:<kernel-assigned>/callback`` listener."""

    def __init__(self, *, state: str) -> None:
        """Bind the socket. ``state`` is the CSRF value this flow will accept."""
        self.state = state
        self.code: str | None = None
        self.failure: str | None = None
        self.done = threading.Event()
        self._server = _QuietHTTPServer(("127.0.0.1", 0), _CallbackHandler)
        self._server.receiver = self  # type: ignore[attr-defined]
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            kwargs={"poll_interval": _SHUTDOWN_POLL_SECONDS}, daemon=True)
        self._thread.start()

    @property
    def redirect_uri(self) -> str:
        """The address to hand the authorization server for this flow."""
        return f"http://127.0.0.1:{self._server.server_address[1]}/callback"

    def wait(self) -> str:
        """Block until the browser delivers a code, then return it.

        Raises:
            LoginError: If the provider reported an error, or the browser never returned.
        """
        if not self.done.wait(timeout=WAIT_TIMEOUT_SECONDS):
            raise LoginError("timed out waiting for the browser to complete the login")
        if self.failure:
            raise LoginError(f"login failed: {self.failure}")
        if not self.code:
            raise LoginError("login completed without an authorization code")
        return self.code

    def close(self) -> None:
        """Stop listening. The port is not left open past the flow that needed it."""
        self._server.shutdown()
        self._server.server_close()

    def __enter__(self) -> LoopbackReceiver:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
