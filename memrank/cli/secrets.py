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
"""`memrank secrets` -- the credentials wallet, in its two scopes.

Store the API keys you have once (:mod:`memrank.secrets.wallet`), or in an org's vault through the
API. What a given target *requires* is answered by `targets show <ref>`, which prints ✔/✘ per
required secret: the target owns that fact, so the catalog is where it is read. The
`memrank preflight` command that used to answer it here is retired -- a run refuses loudly with
the same information at the moment it matters, which cannot go stale between the check and the
run.
"""

from __future__ import annotations

import os
from pathlib import Path

import typer

from memrank.accounts.run_secrets import SIGN_IN, api_message, save_org_secret
from memrank.errors import ActionRequired
from memrank.secrets import wallet
from memrank.term import style, table

secrets_app = typer.Typer(help="Store API keys (a local credentials wallet).")


def _mask(value: str) -> str:
    """Mask a secret for display; ``${VAR}`` references are shown verbatim (not secret)."""
    if value.startswith("${") and value.endswith("}"):
        return value
    if len(value) <= 8:
        return "•" * len(value)
    return f"{value[:3]}…{value[-4:]}"


@secrets_app.command("set")
def set_secret(
    name: str = typer.Argument(..., help="secret env-var name, e.g. OPENAI_API_KEY"),
    value: str = typer.Option(None, "--value", help="value, non-interactive (else prompted, hidden)"),
    ref: str = typer.Option(None, "--ref", help="store a ${ENV_VAR} reference instead of a literal"),
    org: str = typer.Option(None, "--org", help="store for this org instead of the local wallet; "
                                                "cloud runs spend the org's own keys"),
) -> None:
    """Store an API key -- in the local wallet, or for an org with ``--org``.

    Two destinations because they serve two things. The wallet is this machine's, for local
    runs. An org's key lives in the platform and is what a CLOUD run spends: it is written to
    Parameter Store and the task definition carries only its ARN, so the value reaches the
    container without the API ever holding it again.
    """
    if ref and org:
        raise typer.BadParameter("--ref stores a pointer to a variable on THIS machine, which a "
                                 "cloud task cannot read; pass the value itself for --org")
    if ref:
        stored = ref if ref.startswith("${") else f"${{{ref}}}"
    elif value is not None:
        stored = value
    else:
        stored = typer.prompt(f"{name} (paste key)", hide_input=True)
    if org:
        _put_org_secret(org, name, stored)
        return
    wallet.put(name, stored)
    style.say(f"stored {name} -> {wallet.store_path()}")


def _org_client():
    """The authenticated API client for org-vault verbs, or a loud exit naming the fix.

    One chokepoint for token loading, base URL and refusal wording --
    ``run_api_client.authenticated_client`` already distinguishes "not signed in" from
    "your session store would not answer", and every org verb here should say it the
    same way ``runs ls --org`` does.
    """
    from memrank.placement import run_api_client

    try:
        return run_api_client.authenticated_client()
    except run_api_client.RunApiError as exc:
        if exc.code == "no_session":
            raise ActionRequired("You are not signed in. An organisation's keys are reached "
                                 "through your memrank login:", SIGN_IN) from exc
        style.error(str(exc))
        raise typer.Exit(1) from exc


def _put_org_secret(org: str, name: str, value: str) -> None:
    """Store ``name`` for ``org`` through the API, by the path ``memrank run`` saves one too.

    A refusal raises (see :func:`save_org_secret`) -- a credential that did not store must not
    read as though it did, or the next submission fails for a reason already known here.
    """
    with _org_client() as http:
        save_org_secret(http, org, name, value)
    style.say(f"stored {name} for org {org} -- runs recorded in {org} will spend this key")


#: The wallet listing. NAME is what a reader looks up, VALUE is masked because a terminal is a
#: shoulder-surfable place, and SOURCE answers the question the mask provokes: which copy of
#: this key would a run actually use?
_WALLET_COLUMNS = (
    table.Column("NAME", width=24, styler=style.accent),
    table.Column("VALUE", width=14),
    table.Column("SOURCE"),
)

#: An org's vault has no values to show -- the route does not return them -- so it is two columns
#: rather than a wallet listing with a permanently empty one.
_VAULT_COLUMNS = (
    table.Column("NAME", width=24, styler=style.accent),
    table.Column("BACKEND", styler=style.unit),
)


@secrets_app.command("ls")
def list_secrets(
    org: str = typer.Option(None, "--org", help="list this org's vault (names only) "
                                                "instead of the local wallet"),
) -> None:
    """List stored secrets -- the wallet (values masked), or an org's vault with ``--org``."""
    if org:
        _list_org_secrets(org)
        return
    pooled = set(wallet.names())
    rows: list[list] = []
    for name in sorted(pooled):
        raw = wallet.get(name) or ""
        src = "env (overrides)" if name in os.environ else "wallet"
        # An environment value SHADOWS the stored one, so the source is the finding, not a
        # footnote -- coloured as an advisory because the secret being used is not the one here.
        rows.append([name, _mask(raw),
                     table.Cell(src, style.caution if name in os.environ else style.unit)])
    # Keys resolvable ONLY from the environment are invisible to the wallet but very much in play.
    # Showing them is the difference between "what have I stored" and "what would a run find".
    for name in sorted(set(_IMPORTABLE) & set(os.environ) - pooled):
        rows.append([name, table.Cell("(not stored)", style.dim),
                     table.Cell("env only", style.unit)])
    if rows:
        table.emit(_WALLET_COLUMNS, rows, title="Secrets")
    if not pooled and not (set(_IMPORTABLE) & set(os.environ)):
        style.say("  (no credentials -- `memrank secrets set <NAME>` "
                  "or `memrank secrets import-env`)")


def _list_org_secrets(org: str) -> None:
    """Print an org's credential names and backends. Never values -- the route has none."""
    with _org_client() as http:
        response = http.get(f"/orgs/{org}/secrets")
    if response.status_code >= 400:
        style.error(f"could not list secrets for {org!r}: {api_message(response)}")
        raise typer.Exit(1)
    rows = response.json()
    if not rows:
        style.say(f"  (no org keys -- `memrank secrets set <NAME> --org {org}`)")
        return
    table.emit(_VAULT_COLUMNS, [[row["name"], row["backend"]] for row in rows],
               title=f"{org} vault")


# Only these reach config.secret(); everything else in a .env file is non-secret config
# (DATABASE_URL, Arena settings, flags) or a MEM0_* declaration the manifest cross-check reads
# from the environment on purpose. Importing those would put non-secrets in the wallet.
_IMPORTABLE = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "VOYAGE_API_KEY", "GEMINI_API_KEY",
               "GROQ_API_KEY", "HINDSIGHT_API_KEY", "ATOMICMEMORY_API_KEY")


def _dotenv_secrets(path: Path) -> dict[str, str]:
    """The credential lines of a .env file. Parsed here rather than loaded, deliberately --
    ``config.secret()`` no longer reads .env, and importing must not resurrect that path."""
    found: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, _, value = stripped.partition("=")
        name, value = name.strip(), value.strip().strip('"').strip("'")
        if name in _IMPORTABLE and value:
            found[name] = value
    return found


@secrets_app.command("import-env")
def import_env(
    path: Path = typer.Option(Path(".env"), "--from", help="file to read"),
    overwrite: bool = typer.Option(False, "--overwrite", help="replace keys already in the wallet"),
) -> None:
    """Copy API keys out of a ``.env`` file into the wallet, once.

    ``config.secret()`` deliberately no longer reads ``.env``: a file that populates the process
    behind your back makes a credential look present when it is genuinely absent in CI and in the
    cloud. This is the one-time migration, so the change costs an invocation rather than a hunt.

    The ``.env`` file is left alone -- the non-secret configuration in it is still read.
    """
    if not path.exists():
        style.error(f"{path} does not exist")
        raise typer.Exit(1)
    found = _dotenv_secrets(path)
    if not found:
        style.say(f"no importable credentials in {path} "
                  f"(looked for: {', '.join(_IMPORTABLE)})")
        return
    stored = set(wallet.names())
    for name, value in sorted(found.items()):
        if name in stored and not overwrite:
            style.say(f"  skipped {name:24} (already in the wallet; --overwrite to replace)")
            continue
        wallet.put(name, value)
        style.say(f"  imported {name:24} -> {wallet.store_path()}")


@secrets_app.command("rm")
def rm_secret(
    name: str = typer.Argument(..., help="secret to remove"),
    org: str = typer.Option(None, "--org", help="remove from this org's vault "
                                                "instead of the local wallet"),
) -> None:
    """Remove a secret -- from the wallet, or from an org's vault with ``--org``."""
    if org:
        _rm_org_secret(org, name)
        return
    style.say(f"removed {name}" if wallet.delete(name) else f"{name} was not stored")


def _rm_org_secret(org: str, name: str) -> None:
    """Delete ``name`` from ``org``'s vault through the API; say what actually happened."""
    with _org_client() as http:
        response = http.delete(f"/orgs/{org}/secrets/{name}")
    if response.status_code >= 400:
        style.error(f"could not remove {name} for {org!r}: {api_message(response)}")
        raise typer.Exit(1)
    style.say(f"removed {name} for org {org} -- cloud runs can no longer spend it")
    if response.json().get("backend") == "ssm":
        style.note("the platform's reference is gone; the parameter in AWS SSM is retained "
                   "(the next `secrets set` of this name overwrites it)")


@secrets_app.command("path")
def secrets_path() -> None:
    """Print the wallet file path."""
    style.out(str(wallet.store_path()))
