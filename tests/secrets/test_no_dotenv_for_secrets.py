"""Secrets come from the environment or memrank's own wallet -- never from a `.env` file.

`.env` is *implicit*: a file that populates the process behind your back. `export` and ECS's
SSM-into-container-env injection are *deliberate*. Only the implicit one is a problem, and it is a
serious one -- it makes a credential bug invisible exactly where it is cheap to find, and visible
only where it is expensive.

Two instances, both real:

- `memrank submit mem0:bge-tei demo` demanded ANTHROPIC_API_KEY it never uses. Wrong locally for
  weeks, unnoticed because `.env` answered. It surfaced only when an ECS task, with no `.env` and
  no terminal, refused to start.
- The ECS harness container has no judge key, so `--on cloud --judge` cannot work. Still invisible
  locally, for the same reason.

These tests pin the resolution order so neither can recur silently.
"""
from __future__ import annotations

import inspect
from pathlib import Path

from memrank import config


def test_secret_does_not_load_dotenv():
    """The guard, written against the source: re-adding the call has to be a deliberate act."""
    source = inspect.getsource(config.secret)
    assert "_ensure_dotenv" not in source, (
        "config.secret() must not load .env: a key sitting in a file would silently satisfy a "
        "requirement that is genuinely unmet in CI and in the cloud, which is how a broken "
        "credential preflight survived weeks of local runs.")


def test_resolve_ref_does_not_load_dotenv():
    """A wallet `${VAR}` reference resolves against the REAL environment, not a file."""
    assert "_ensure_dotenv" not in inspect.getsource(config._resolve_ref)


def test_a_dotenv_in_the_working_directory_is_ignored(monkeypatch, tmp_path: Path):
    """The end-to-end property. Written with a real .env on disk, because that is the situation."""
    monkeypatch.setenv("MEMRANK_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    (tmp_path / ".env").write_text("ANTHROPIC_API_KEY=from-dotenv\n", encoding="utf-8")
    monkeypatch.setattr(config, "_DOTENV_LOADED", False)   # a fresh process would try to load it

    assert config.secret("ANTHROPIC_API_KEY") is None


def test_the_process_environment_still_wins(monkeypatch, tmp_path: Path):
    """Not a compromise: ECS resolves SSM INTO the container env and CI injects the same way."""
    monkeypatch.setenv("MEMRANK_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-environment")
    assert config.secret("ANTHROPIC_API_KEY") == "from-environment"


def test_the_wallet_answers_when_the_environment_does_not(monkeypatch, tmp_path: Path):
    from memrank.secrets import wallet

    monkeypatch.setenv("MEMRANK_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    wallet.put("ANTHROPIC_API_KEY", "from-wallet")
    assert config.secret("ANTHROPIC_API_KEY") == "from-wallet"


def test_non_secret_config_still_reads_dotenv():
    """Deliberately unchanged: DATABASE_URL, Arena settings and flags are not secrets, and
    removing .env for them would break local dev, dbmate and compose for no gain."""
    assert "_ensure_dotenv" in inspect.getsource(config.database_url)
    assert "_ensure_dotenv" in inspect.getsource(config.mlflow_enabled)
