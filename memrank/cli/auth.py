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
"""``memrank auth login`` / ``logout`` / ``status``.

A thin shell around :mod:`memrank.accounts.login_flow`. Identity is one noun-space rather
than three flat verbs, and ``status`` -- not a bare ``whoami`` -- is the name that can grow to
own credential validity, which is the plane's own business to diagnose.

Its own module rather than folded into ``runner.py``, which is already past the project's
400-line ceiling; ``secrets_cli`` and ``targets_cli`` set the same precedent.
"""
from __future__ import annotations

import time
import webbrowser

import httpx
import typer

from memrank import settings
from memrank.accounts import login_flow
from memrank.accounts.credentials import CredentialError, CredentialStore
from memrank.accounts.loopback import LoginError
from memrank.api_client import api_client
from memrank.term import detail, style, table

app = typer.Typer(help="Who you are signed in as, and which orgs you can act on.")


def _client() -> httpx.Client:
    """An HTTP client bound to the configured API."""
    return api_client(timeout=30)


def _authenticated_client(token: str) -> httpx.Client:
    """An HTTP client carrying the caller's bearer token."""
    return api_client(timeout=30, token=token)


def _default_org(orgs: list[dict]) -> str | None:
    """The org the server declares as this user's default -- their personal org.

    The server answers this because it is the only party that can: every user is given a
    personal org when their account is created, and `/whoami` marks it. Asking the user to
    pick would be the routing decision this phase exists to remove, wearing a different hat --
    and the moment they belong to one team org as well, "pick one" would be everyone's login.

    Read with ``.get`` so a deployment predating the field configures nothing and says so,
    rather than dying on a KeyError.
    """
    for org in orgs:
        if org.get("is_default"):
            return org["slug"]
    return None


def _no_default_to_configure(login: str) -> None:
    """Say what is missing and the two commands that answer it.

    Reached with no memberships at all, or -- a state that should not exist -- memberships none
    of which the server calls default. Both leave the machine signed in but unable to submit,
    so neither may be silent.
    """
    # Names the thing to send, not the command that answers it. The granting command is
    # operator tooling that does not ship, so quoting it told the one person who cannot run it
    # exactly what to type (ATO-2125). What the reader needs is the login to send and the org
    # slug to ask for.
    style.say(f"  {style.caution('!')} no default org -- submissions need one. "
              f"Ask an operator to grant your GitHub login ({login}) membership of an org, "
              f"then run:\n"
              f"      memrank config set defaults.org <slug>\n"
              f"      Already a member? The same command names the org you want to submit to.")


def _configure_after_login(token: str) -> None:
    """Turn a session into a working configuration: a default org, and cloud placement.

    Signing in and being able to submit are different things, and the gap between them is
    exactly the flags the golden path must not have. `defaults.on` is written HERE rather than
    shipped, so a machine that never signed in stays local and nothing fails for lack of a login.

    The default org is configured on the FIRST login only. A later `auth login` -- a rotated
    token, a lapsed session -- must not quietly undo a default the user chose since, so an org
    that already answers is reported and left alone.
    """
    with _authenticated_client(token) as http:
        response = http.get("/whoami")
    response.raise_for_status()
    body = response.json()
    settings.put("defaults.on", "cloud")
    configured, source = settings.resolve("defaults.org")
    if configured is not None:
        # An env value shadows the file everywhere, so naming it here saves the user wondering
        # why `config set` later appears to do nothing -- the same trap `config set` warns about.
        via = " (from MEMRANK_ORG)" if source == settings.ENV else ""
        style.say(f"  {style.good('✓')} default org: {configured}{via} -- unchanged")
    elif (org := _default_org(body["orgs"])) is not None:
        settings.put("defaults.org", org)
        style.say(f"  {style.good('✓')} default org: {org}")
    else:
        _no_default_to_configure(body["login"])
    style.say(f"  {style.good('✓')} defaults: on=cloud")


def _stored_token() -> str | None:
    """The stored session, or a clean refusal when the store itself would not answer.

    A CredentialError carries a sentence written for a person; letting it escape prints sixty
    lines of keyring internals instead. Every entry point that reads the credential funnels
    through here so there is one place that has to remember.
    """
    try:
        return CredentialStore().load()
    except CredentialError as exc:
        style.error(str(exc))
        raise typer.Exit(code=1) from exc


def _has_browser() -> bool:
    """Whether this machine can open a browser at all.

    ``webbrowser.get()`` raises when it can find nothing to launch, which is the honest test --
    ``webbrowser.open`` answers the same question by returning ``False`` only after the flow
    has already committed to waiting for a callback.
    """
    try:
        webbrowser.get()
    except webbrowser.Error:
        return False
    return True


def _announce_verification_url(url: str) -> None:
    """Print the URL a browserless login must be approved at."""
    style.say("No browser here. Open this on any device to sign in:\n"
              f"\n      {url}\n")


def _open_and_announce(url: str) -> None:
    """Open the sign-in page here, and print its link for when the browser does not come up.

    The link matters to a coding agent running the login for a person: the agent cannot see
    whether a browser window appeared, so it hands the printed link on instead. The page calls
    back to a port on this computer, so the link only works in a browser on this computer.
    """
    style.say("Opening your browser to sign in. If it did not open, open this link in a "
              f"browser on this computer:\n\n      {url}\n\nWaiting for you to sign in...")
    webbrowser.open(url)


def _browserless_login(http) -> dict[str, str]:
    """Sign in by URL and polling, for a machine with no browser of its own."""
    return login_flow.perform_device_login(
        http, announce=_announce_verification_url, sleep=time.sleep)


@app.command()
def login(no_browser: bool = typer.Option(
        False, "--no-browser",
        help="Print a URL to open elsewhere instead of launching a browser here.")) -> None:
    """Sign in through the browser, and configure this machine to submit.

    Falls back to the browserless flow on its own when there is no browser to open, rather
    than requiring ``--no-browser``: over SSH, the person who needs the flag is the person who
    has no way to know it exists. The flag remains for the case the detection cannot see --
    a browser that is present but unusable, such as an X display that will not open.
    """
    store = CredentialStore()
    browserless = no_browser or not _has_browser()
    try:
        with _client() as http:
            result = (_browserless_login(http) if browserless
                      else login_flow.perform_login(http, open_browser=_open_and_announce))
    except LoginError as exc:
        style.error(str(exc))
        raise typer.Exit(code=1) from exc
    store.save(result["token"])
    style.say(f"  {style.good('✓')} signed in -- token expires {result['expires_at']}")
    _configure_after_login(result["token"])


@app.command()
def logout() -> None:
    """Forget the stored CLI token on this machine.

    Local only, deliberately: this clears the credential here, it does not revoke it
    server-side. Use ``manage-users.py token revoke`` for a token you believe is compromised
    -- and say so rather than letting the user assume otherwise.
    """
    CredentialStore().clear()
    style.say("Signed out locally. To revoke the token itself, use `token revoke`.")


@app.command()
def status() -> None:
    """Show who you are signed in as, and which orgs you can act on."""
    token = _stored_token()
    if token is None:
        style.say("not signed in (run `memrank auth login`)")
        raise typer.Exit(code=1)
    with _authenticated_client(token) as http:
        response = http.get("/whoami")
    if response.status_code == 401:
        style.say("your session has expired -- run `memrank auth login` again")
        raise typer.Exit(code=1)
    response.raise_for_status()
    body = response.json()
    fields = [("account", f"{style.accent(body['login'])} {style.unit('<' + body['email'] + '>')}")]
    # Reaching a 200 IS the validity check -- the expiry below says when that stops being true.
    # Absent when talking to a deployment older than this field; omitted rather than guessed.
    if body.get("expires_at"):
        fields.append(("session", f"{style.good('valid')} "
                                  f"{style.unit('expires ' + body['expires_at'])}"))
    detail.emit(detail.panel("signed in", fields))
    default_org = settings.get("defaults.org")
    if body["orgs"]:
        # The default is a MARKED ROW rather than a suffix: which org a submission runs under is
        # the question this listing is asked, and a parenthetical at the end of one line is the
        # easiest thing on the screen to miss.
        table.emit((table.Column("", width=1),
                    table.Column("ORG", styler=style.accent),
                    table.Column("NAME")),
                   [[table.Cell("▸", style.accent) if org["slug"] == default_org else "",
                     org["slug"], org["name"]] for org in body["orgs"]],
                   title="Orgs")
    if not body["orgs"]:
        style.say("  (no orgs yet -- ask an operator to grant you one)")
    if default_org and default_org not in {o["slug"] for o in body["orgs"]}:
        # A default naming an org you cannot act on fails at submission, far from its cause.
        style.say(f"  {style.caution('!')} defaults.org is {default_org!r}, "
                  f"which you are not a member of -- "
                  f"`memrank config set defaults.org <slug>`")
