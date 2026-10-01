"""Shared fixtures for the credentials-wallet + requirements tests.

Isolates every test to a tmp wallet dir and a clean environment for the API-key vars we assert on,
so a real repo ``.env`` never leaks into the resolver. No network, no timing.
"""
from __future__ import annotations

from pathlib import Path

import pytest

_ISOLATED_VARS = (
    "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "VOYAGE_API_KEY", "GEMINI_API_KEY", "MY_REF_KEY",
)


@pytest.fixture
def wallet(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("MEMRANK_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.chdir(tmp_path)  # no .env here -> dotenv load is a no-op
    for var in _ISOLATED_VARS:
        monkeypatch.delenv(var, raising=False)
    # Imported here, not at module scope, so the env isolation above is already in place -- and
    # aliased because the fixture itself is named `wallet`.
    from memrank.secrets import wallet as module
    return module
