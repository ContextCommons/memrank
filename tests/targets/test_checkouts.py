"""The per-machine checkout store, and the `rootFrom: link` binding it answers.

A source target evaluates a working tree whose location no committed file can state: memrank does
not know the repository holding the descriptor, that repository's engine does not know it is being
evaluated, and the checkout may be anywhere. These tests pin the three properties that makes the
store worth having -- it is keyed per target so two arms cannot collide, an unlinked target is
reportable rather than fatal, and argv beats the store.
"""
from __future__ import annotations

import pytest

from memrank.errors import MemrankError
from memrank.targets import checkouts
from memrank.targets import manifest as m

SOURCE = {
    "schema_version": 1,
    "name": "myengine:slm",
    "kind": "stack",
    "interface": {"adapter": "myengine", "transport": "http"},
    "launch": {"command": "cargo run -- --bind 127.0.0.1:{port}", "requires": ["Cargo.toml"]},
    "network": {"port": 8080, "readiness": {"path": "/health"}},
}


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path, monkeypatch):
    """Never touch the operator's real ~/.config/memrank while testing."""
    monkeypatch.setenv("MEMRANK_CONFIG_DIR", str(tmp_path))


def _linked(binding, name="myengine:slm"):
    return m.from_dict({**SOURCE, "name": name, "binding": binding})


def test_a_stored_path_comes_back_absolute(tmp_path):
    stored = checkouts.put("myengine:slm", str(tmp_path))
    assert stored == str(tmp_path.resolve())
    assert checkouts.resolve("myengine:slm") == str(tmp_path.resolve())


def test_an_unlinked_target_resolves_to_nothing_rather_than_raising():
    assert checkouts.resolve("myengine:slm") is None
    assert _linked({"kind": "source", "rootFrom": "link"}).binding.root is None


def test_two_arms_of_one_repo_do_not_collide(tmp_path):
    """The reason the key is the target ref and not the repository."""
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    checkouts.put("myengine:slm", str(a))
    checkouts.put("myengine:slm@featx", str(b))

    assert _linked({"kind": "source", "rootFrom": "link"}).binding.root == str(a)
    assert _linked({"kind": "source", "rootFrom": "link"},
                   name="myengine:slm@featx").binding.root == str(b)


def test_a_stored_env_reference_resolves_through_config(monkeypatch, tmp_path):
    """The wallet's value contract, borrowed: a link may name a variable instead of a path."""
    monkeypatch.setenv("MYENGINE_ROOT", str(tmp_path))
    assert checkouts.put("myengine:slm", "${MYENGINE_ROOT}") == "${MYENGINE_ROOT}"
    assert checkouts.resolve("myengine:slm") == str(tmp_path)


def test_a_reference_to_an_unset_variable_refuses_by_name(monkeypatch):
    monkeypatch.delenv("MYENGINE_ROOT", raising=False)
    checkouts.put("myengine:slm", "${MYENGINE_ROOT}")
    with pytest.raises(MemrankError, match="MYENGINE_ROOT"):
        checkouts.resolve("myengine:slm")


def test_declaring_both_root_and_rootFrom_is_refused():
    """Refused rather than ranked: a precedence rule is invisible to whoever reads the file."""
    with pytest.raises(m.ManifestError, match="both root and rootFrom"):
        _linked({"kind": "source", "root": "/work/myengine", "rootFrom": "link"})


def test_an_unknown_rootFrom_names_what_is_known():
    with pytest.raises(m.ManifestError, match="known: link"):
        _linked({"kind": "source", "rootFrom": "wallet"})


def test_unlinking_reports_whether_there_was_a_link(tmp_path):
    checkouts.put("myengine:slm", str(tmp_path))
    assert checkouts.delete("myengine:slm") is True
    assert checkouts.delete("myengine:slm") is False
