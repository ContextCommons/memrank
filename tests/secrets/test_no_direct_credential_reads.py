"""Every credential read must go through memrank's one resolver.

This guard exists because per-surface discipline already failed: `judge_client` and two adapters
each read an API key straight from ``os.environ``, so a key held in the wallet was invisible to them
and a run died mid-way on a credential ``preflight`` had already reported as present. Fixing the
three call sites is not enough -- the fourth one someone adds next month is the actual risk, which is
what this enumerates.
"""
from __future__ import annotations

import re
from pathlib import Path

from memrank.secrets.names import _SECRET_EXACT, _SECRET_SUBSTRINGS

PACKAGE = Path(__file__).resolve().parents[2] / "memrank"

# Files allowed to read credentials from the environment: the resolver itself, and the CLI line
# that reports WHICH source won (a membership test, not a credential read).
ALLOWED = {"config.py", "secrets_cli.py"}

# Credential-shaped env vars that the wallet does not and must not hold, exempted BY NAME rather
# than by file -- an exempt filename would excuse every future read in that module too.
#
# The rule this guard enforces is "a key stored with `memrank secrets set` must be visible to the
# code that needs it". That rule is about the wallet's contents: per-org provider keys, resolved
# through `config.secret()`. MEMRANK_RUN_TOKEN is not one. It is minted by the API at launch and
# injected into the harness container's environment for the length of one run
# (`memrank/api/run_tokens.py`), names a run rather than a provider, and could not be stored in a
# wallet by anyone -- there is nobody to store it and nothing to store it for. Resolving it through
# the wallet would mean looking up a per-run secret in a per-org store, and finding nothing.
_NOT_WALLET_SECRETS = {"MEMRANK_RUN_TOKEN"}

# `os.environ.get("NAME")` / `os.environ["NAME"]` / `"NAME" in os.environ`
_ENV_READ = re.compile(r"""os\.environ(?:\.get)?\s*[\(\[]\s*["']([A-Z0-9_]+)["']""")


# Suffixes that mark an ENV VAR as a credential. Deliberately stricter than receipt.py's config-key
# heuristics, which omit a bare "token" so that legitimate config fields like `token_budget` and
# `tokens_per_query` are not redacted. Environment variables live in a different namespace: there is
# no MEM0_TOKEN_BUDGET env var, while ANYTHING ending in _TOKEN/_KEY/_SECRET is a credential.
_CREDENTIAL_SUFFIXES = ("_KEY", "_TOKEN", "_SECRET", "_PASSWORD")


def _looks_like_a_credential(var: str) -> bool:
    """Credential env-var names, distinguished from DECLARATIONS.

    Declarations such as ``MEM0_EMBEDDER_MODEL`` must keep reading the environment directly: the M2
    cross-check compares them against the manifest to prove the engine process saw what memrank
    claims, so resolving them through the wallet would defeat the check entirely.
    """
    if var in _NOT_WALLET_SECRETS:
        return False
    flat = var.replace("_", "").lower()
    return (var.endswith(_CREDENTIAL_SUFFIXES)
            or any(s in flat for s in _SECRET_SUBSTRINGS)
            or var.lower() in _SECRET_EXACT)


def _offenders() -> list[str]:
    found = []
    for path in sorted(PACKAGE.rglob("*.py")):
        if path.name in ALLOWED:
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for var in _ENV_READ.findall(line):
                if _looks_like_a_credential(var):
                    found.append(f"{path.relative_to(PACKAGE)}:{number}: {var}")
    return found


def test_no_module_reads_a_credential_from_the_environment():
    offenders = _offenders()
    assert not offenders, (
        "these read a credential directly instead of via config.secret(), so a key stored with "
        "`memrank secrets set` would be invisible to them:\n  " + "\n  ".join(offenders))


def test_the_guard_recognises_credential_names():
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "HINDSIGHT_API_KEY", "SUPERMEMORY_TOKEN"):
        assert _looks_like_a_credential(var), var


def test_the_guard_does_not_flag_a_platform_injected_run_token():
    """The exemption is asserted, so removing the name breaks a test rather than a deployment."""
    assert not _looks_like_a_credential("MEMRANK_RUN_TOKEN")
    # ...and the exemption is one name, not a suffix: a real credential ending the same way is
    # still caught.
    assert _looks_like_a_credential("MEMRANK_OTHER_TOKEN")


def test_the_guard_does_not_flag_declarations():
    """The distinction that makes this guard usable rather than noisy."""
    for var in ("MEM0_EMBEDDER_MODEL", "MEM0_EMBEDDING_DIMS", "MEM0_LLM_PROVIDER",
                "MEMRANK_CONFIG_DIR", "MEMRANK_RUNS_DIR", "MEM0_HTTP_URL"):
        assert not _looks_like_a_credential(var), var


def test_the_guard_would_actually_catch_a_regression(tmp_path, monkeypatch):
    """A guard that cannot fail is not a guard."""
    offending = tmp_path / "memrank" / "new_surface.py"
    offending.parent.mkdir()
    offending.write_text('import os\nkey = os.environ.get("ANTHROPIC_API_KEY")\n', encoding="utf-8")
    monkeypatch.setattr("tests.secrets.test_no_direct_credential_reads.PACKAGE", offending.parent)
    assert _offenders(), "the enumeration missed a planted credential read"
