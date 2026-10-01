"""Resolving a missing credential once, then never again."""
from __future__ import annotations

import pytest

from memrank import config
from memrank.config import ConfigError


def test_nothing_missing_is_a_no_op(wallet, monkeypatch):
    wallet.put("ANTHROPIC_API_KEY", "sk-stored")
    monkeypatch.setattr(config, "_stdin_is_interactive", lambda: False)
    config.ensure_secrets(["ANTHROPIC_API_KEY"])          # must not raise, must not prompt


def test_empty_request_is_a_no_op(wallet):
    config.ensure_secrets([])


def test_missing_key_is_prompted_and_stored(wallet, monkeypatch):
    monkeypatch.setattr(config, "_stdin_is_interactive", lambda: True)
    monkeypatch.setattr(config, "_prompt_for_secret", lambda name: "sk-typed")
    config.ensure_secrets(["ANTHROPIC_API_KEY"])
    assert wallet.get("ANTHROPIC_API_KEY") == "sk-typed"   # persisted for every future run
    assert config.secret("ANTHROPIC_API_KEY") == "sk-typed"


def test_a_second_run_does_not_prompt_again(wallet, monkeypatch):
    monkeypatch.setattr(config, "_stdin_is_interactive", lambda: True)
    monkeypatch.setattr(config, "_prompt_for_secret", lambda name: "sk-typed")
    config.ensure_secrets(["ANTHROPIC_API_KEY"])

    def _fail(name):
        raise AssertionError(f"prompted again for {name}")

    monkeypatch.setattr(config, "_prompt_for_secret", _fail)
    config.ensure_secrets(["ANTHROPIC_API_KEY"])


def test_non_interactive_refuses_rather_than_hanging(wallet, monkeypatch):
    monkeypatch.setattr(config, "_stdin_is_interactive", lambda: False)
    with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY"):
        config.ensure_secrets(["ANTHROPIC_API_KEY"])


def test_non_interactive_lists_every_missing_key_at_once(wallet, monkeypatch):
    monkeypatch.setattr(config, "_stdin_is_interactive", lambda: False)
    with pytest.raises(ConfigError) as exc:
        config.ensure_secrets(["ANTHROPIC_API_KEY", "VOYAGE_API_KEY"])
    assert "ANTHROPIC_API_KEY" in str(exc.value)
    assert "VOYAGE_API_KEY" in str(exc.value)


def test_an_empty_answer_is_rejected(wallet, monkeypatch):
    """A blank paste must not be stored as a valid credential."""
    monkeypatch.setattr(config, "_stdin_is_interactive", lambda: True)
    monkeypatch.setattr(config, "_prompt_for_secret", lambda name: "   ")
    with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY"):
        config.ensure_secrets(["ANTHROPIC_API_KEY"])
    assert wallet.get("ANTHROPIC_API_KEY") is None


def test_duplicate_names_are_asked_once(wallet, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(config, "_stdin_is_interactive", lambda: True)
    monkeypatch.setattr(config, "_prompt_for_secret", lambda name: calls.append(name) or "sk-x")
    config.ensure_secrets(["ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY"])
    assert calls == ["ANTHROPIC_API_KEY"]


def test_storing_a_secret_still_says_where_it_went(wallet, monkeypatch, capsys):
    """The one line `style.say` prints here, asserted because its import moved.

    `memrank/config.py` imported `memrank.term.style` at module level to serve this single
    call, which put typer into every library caller's process (ATO-2133). The import is now
    local to the call, and the risk of that change is a line that quietly stops appearing --
    an operator who pasted a key and was told nothing would not know it was stored, or where.
    """
    monkeypatch.setattr(config, "_stdin_is_interactive", lambda: True)
    monkeypatch.setattr(config, "_prompt_for_secret", lambda name: "sk-typed")

    config.ensure_secrets(["ANTHROPIC_API_KEY"])

    said = capsys.readouterr()
    assert f"stored ANTHROPIC_API_KEY -> {wallet.store_path()}" in said.out + said.err
