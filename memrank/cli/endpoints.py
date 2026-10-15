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
"""`memrank endpoints` -- the LLM endpoints an organisation judges through by name.

Register an OpenAI-compatible (or Anthropic-compatible) endpoint once, and a run names it as
``<endpoint>:<model>``. The endpoint lives in the organisation, so every teammate's run resolves
the same one; its base URL stays there, and results record ``<id>@<revision>`` instead of it.
"""
from __future__ import annotations

import typer

from memrank import settings
from memrank.accounts import org_endpoints
from memrank.accounts.run_secrets import (
    api_message,
    load_org_secrets,
    save_org_secret,
    session_rejected,
)
from memrank.cli.secrets import _org_client
from memrank.errors import ActionRequired
from memrank.term import style, table

endpoints_app = typer.Typer(help="Name the LLM endpoints your organisation judges through.")

#: What a user runs to choose the organisation, when neither --org nor a default names one.
_SET_ORG = "memrank config set defaults.org <slug>"

_COLUMNS = (
    table.Column("NAME", width=16, styler=style.accent),
    table.Column("ADAPTER", width=10),
    table.Column("KEY", width=26),
    table.Column("MODELS", width=8),
    table.Column("RECORDED AS", styler=style.unit),
)

_ORG_HELP = "the organisation (default: defaults.org)"


def _org(org: str | None) -> str:
    """``--org``, else the default organisation runs are recorded in; refuse without either."""
    chosen = org or settings.get("defaults.org")
    if not chosen:
        raise ActionRequired("No organisation named. Pass --org, or set your default:", _SET_ORG)
    return chosen


def _refuse_if_invalid(name: str, base_url: str, key_ref: str) -> None:
    """Check the entry before anything is asked or saved; the API applies the same rules."""
    for problem in (org_endpoints.name_problem(name), org_endpoints.base_url_problem(base_url),
                    org_endpoints.key_ref_problem(key_ref)):
        if problem:
            raise typer.BadParameter(problem)


def _saved_secret_names(http, org: str) -> set[str]:
    """The names of ``org``'s saved keys, so ``add`` asks only for a key that is missing."""
    response = http.get(f"/orgs/{org}/secrets")
    if response.status_code == 401:
        raise session_rejected()
    if response.status_code >= 400:
        raise org_endpoints.EndpointError(
            f"could not read {org!r}'s saved keys: {api_message(response)}",
            status=response.status_code)
    return {row["name"] for row in response.json()}


def _key_for(http, org: str, key_ref: str) -> tuple[str, bool]:
    """The key ``key_ref`` names, and whether it was typed here (so is not saved yet)."""
    if key_ref in _saved_secret_names(http, org):
        return load_org_secrets(http, org)[key_ref], False
    return typer.prompt(f"{key_ref} (paste key; saved to {org})", hide_input=True), True


@endpoints_app.command("add")
def add_endpoint(
    name: str = typer.Argument(..., help="what runs call it: --judge <name>:<model>"),
    adapter: str = typer.Option(..., "--adapter", help="the API it speaks: openai | anthropic"),
    base_url: str = typer.Option(..., "--base-url", help="its base URL, e.g. https://host/v1"),
    key_ref: str = typer.Option(..., "--key-ref", help="the org secret holding its key"),
    org: str = typer.Option(None, "--org", help=_ORG_HELP),
) -> None:
    """Register an endpoint, or replace the one of that name.

    Checks the key against the endpoint's models listing first; a refused key saves nothing.
    """
    if adapter not in org_endpoints.ADAPTERS:
        raise typer.BadParameter(f"--adapter is one of {', '.join(org_endpoints.ADAPTERS)}")
    _refuse_if_invalid(name, base_url, key_ref)
    org = _org(org)
    with _org_client() as http:
        key, typed = _key_for(http, org, key_ref)
        listing = org_endpoints.has_models_listing(name=name, adapter=adapter,
                                                   api_base=base_url.rstrip("/"), key=key)
        if typed:
            save_org_secret(http, org, key_ref, key)
        saved = org_endpoints.save_endpoint(http, org, {
            "name": name, "adapter": adapter, "api_base": base_url, "key_ref": key_ref,
            "models_listing": listing})
    if typed:
        style.say(f"stored {key_ref} for org {org}")
    style.say(f"registered {name} ({adapter}) for org {org} -- judge with "
              f"--judge {name}:<model>; results record it as {saved['fingerprint']}")
    if not listing:
        style.note(f"{name} has no models listing, so a run checks its model with one "
                   f"1-token completion")


@endpoints_app.command("ls")
def list_endpoints(org: str = typer.Option(None, "--org", help=_ORG_HELP)) -> None:
    """List the organisation's endpoints. Base URLs are never shown."""
    org = _org(org)
    with _org_client() as http:
        rows = org_endpoints.list_endpoints(http, org)
    if not rows:
        style.say(f"  (no endpoints -- `memrank endpoints add <NAME> --org {org} ...`)")
        return
    table.emit(_COLUMNS, [[row["name"], row["adapter"], row["key_ref"],
                           "listed" if row["models_listing"] else "none", row["fingerprint"]]
                          for row in rows], title=f"{org} endpoints")


@endpoints_app.command("rm")
def rm_endpoint(
    name: str = typer.Argument(..., help="the endpoint to remove"),
    org: str = typer.Option(None, "--org", help=_ORG_HELP),
) -> None:
    """Remove an endpoint. Its key stays saved; `memrank secrets rm --org` removes that."""
    org = _org(org)
    with _org_client() as http:
        org_endpoints.delete_endpoint(http, org, name)
    style.say(f"removed {name} for org {org} -- runs can no longer judge through it")
