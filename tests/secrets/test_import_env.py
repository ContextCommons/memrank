"""Migrating credentials out of a `.env` file into the wallet.

`config.secret()` no longer reads `.env`, so this is the one-time move that makes the change cost
an invocation rather than a hunt. It parses the file rather than loading it -- importing must not
resurrect the very path that was removed.
"""
from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from memrank.cli.secrets import secrets_app

runner = CliRunner()

DOTENV = """\
# a comment
ANTHROPIC_API_KEY=sk-ant-real
OPENAI_API_KEY="sk-openai-quoted"
DATABASE_URL=postgres://user:pw@host/db
MEM0_EMBEDDER_MODEL=BAAI/bge-small-en-v1.5
ARENA_TOKEN_SECRET=not-ours

EMPTY_KEY=
"""


def _invoke(tmp_path: Path, monkeypatch, *args):
    monkeypatch.setenv("MEMRANK_CONFIG_DIR", str(tmp_path / "cfg"))
    env_file = tmp_path / ".env"
    env_file.write_text(DOTENV, encoding="utf-8")
    return runner.invoke(secrets_app, ["import-env", "--from", str(env_file), *args])


def test_credentials_are_imported(tmp_path, monkeypatch):
    from memrank.secrets import wallet

    assert _invoke(tmp_path, monkeypatch).exit_code == 0
    assert wallet.get("ANTHROPIC_API_KEY") == "sk-ant-real"
    assert wallet.get("OPENAI_API_KEY") == "sk-openai-quoted"      # quotes stripped


def test_non_secret_config_is_not_imported(tmp_path, monkeypatch):
    """DATABASE_URL and the MEM0_* declarations are not secrets in the wallet's model -- the
    declarations in particular MUST stay env-readable for the manifest cross-check."""
    from memrank.secrets import wallet

    _invoke(tmp_path, monkeypatch)
    stored = set(wallet.names())
    assert "DATABASE_URL" not in stored
    assert "MEM0_EMBEDDER_MODEL" not in stored
    assert "ARENA_TOKEN_SECRET" not in stored                            # consumed by Deno, not us


def test_a_blank_value_is_skipped(tmp_path, monkeypatch):
    from memrank.secrets import wallet

    _invoke(tmp_path, monkeypatch)
    assert "EMPTY_KEY" not in set(wallet.names())


def test_an_existing_key_is_not_clobbered(tmp_path, monkeypatch):
    """Re-running the migration must not overwrite a key deliberately set to something else."""
    from memrank.secrets import wallet

    monkeypatch.setenv("MEMRANK_CONFIG_DIR", str(tmp_path / "cfg"))
    wallet.put("ANTHROPIC_API_KEY", "deliberately-different")
    result = _invoke(tmp_path, monkeypatch)
    assert "skipped" in result.output
    assert wallet.get("ANTHROPIC_API_KEY") == "deliberately-different"


def test_overwrite_is_explicit(tmp_path, monkeypatch):
    from memrank.secrets import wallet

    monkeypatch.setenv("MEMRANK_CONFIG_DIR", str(tmp_path / "cfg"))
    wallet.put("ANTHROPIC_API_KEY", "old")
    _invoke(tmp_path, monkeypatch, "--overwrite")
    assert wallet.get("ANTHROPIC_API_KEY") == "sk-ant-real"


def test_a_missing_file_is_an_error(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMRANK_CONFIG_DIR", str(tmp_path / "cfg"))
    result = runner.invoke(secrets_app, ["import-env", "--from", str(tmp_path / "nope.env")])
    assert result.exit_code == 1
    assert "does not exist" in result.output


def test_importing_does_not_reload_dotenv_into_the_process(tmp_path, monkeypatch):
    """The whole point: the migration must not put .env back on the resolution path."""
    from memrank import config

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    _invoke(tmp_path, monkeypatch)
    assert "ANTHROPIC_API_KEY" not in __import__("os").environ
    assert config.secret("ANTHROPIC_API_KEY") == "sk-ant-real"           # from the wallet only
