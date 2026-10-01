"""Descriptors read from `targets.path` -- the directories an operator owns.

This is what lets a target file live beside the translator it launches, so the pair can be copied,
shared or committed as one thing. The property under test throughout is that the SAME bytes work
wherever the directory ends up: today every user hand-writes their own copy with their own absolute
path, and nothing is shareable.
"""

from __future__ import annotations

import os

import pytest

from memrank.targets import catalog as c
from memrank.targets.manifest import ManifestError
from tests import withheld

DESCRIPTOR = """\
schema_version: 1
name: myengine:dev
kind: stack
interface: {adapter: native, transport: translator}
binding: {kind: source}
launch: {command: "python translators/myengine.py --port {port}"}
network: {port: 8099, readiness: {path: /memrank/v1/describe}}
"""


def evals_dir(tmp_path, monkeypatch, name="evals", body=DESCRIPTOR, target="myengine"):
    """A directory of descriptors, put on targets.path."""
    directory = tmp_path / name / "targets"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{target}.yaml").write_text(body, encoding="utf-8")
    monkeypatch.setenv("MEMRANK_CONFIG_DIR", str(tmp_path / "config"))
    existing = os.environ.get("MEMRANK_TARGETS_PATH")
    monkeypatch.setenv("MEMRANK_TARGETS_PATH",
                       f"{existing}{os.pathsep}{directory}" if existing else str(directory))
    return directory


def test_a_descriptor_on_the_path_becomes_a_target(tmp_path, monkeypatch):
    evals_dir(tmp_path, monkeypatch)

    assert "myengine:dev" in c.list_targets()
    assert c.resolve_target("myengine:dev").adapter == "native"


def test_root_defaults_to_the_directory_the_descriptor_came_from(tmp_path, monkeypatch):
    """The whole point: no absolute path in the file, so the same bytes work anywhere."""
    directory = evals_dir(tmp_path, monkeypatch)

    got = c.resolve_target("myengine:dev")

    assert got.binding is not None
    assert got.binding.root == str(directory.resolve())


def test_the_same_descriptor_resolves_after_being_moved(tmp_path, monkeypatch):
    """Portability, stated as a test: copy the folder, and it still describes itself correctly."""
    first = evals_dir(tmp_path, monkeypatch, name="here")
    monkeypatch.delenv("MEMRANK_TARGETS_PATH")
    second = evals_dir(tmp_path, monkeypatch, name="there")

    assert (first / "myengine.yaml").read_text() == (second / "myengine.yaml").read_text()
    assert c.resolve_target("myengine:dev").binding.root == str(second.resolve())


def test_two_directories_claiming_one_name_is_refused(tmp_path, monkeypatch):
    """Peers on a search path have no honest precedence rule, and a quiet merge would run
    whichever sorted first while the receipt named the other."""
    evals_dir(tmp_path, monkeypatch, name="one")
    evals_dir(tmp_path, monkeypatch, name="two")

    with pytest.raises(ManifestError, match="defined in both"):
        c.list_targets()


def test_the_user_dir_still_wins_over_the_path(tmp_path, monkeypatch):
    """`${MEMRANK_CONFIG_DIR}/targets` stays the final local override, as it always was."""
    evals_dir(tmp_path, monkeypatch)
    user = tmp_path / "config" / "targets"
    user.mkdir(parents=True, exist_ok=True)
    (user / "myengine.yaml").write_text(
        "name: myengine:dev\nkind: in-process\nadapter: word-overlap\n", encoding="utf-8")

    assert c.resolve_target("myengine:dev").adapter == "word-overlap"


def test_an_unset_path_changes_nothing(tmp_path, monkeypatch):
    monkeypatch.delenv("MEMRANK_TARGETS_PATH", raising=False)
    monkeypatch.setenv("MEMRANK_CONFIG_DIR", str(tmp_path))

    assert c.path_dirs() == []
    withheld.require("mem0")
    assert "mem0" in c.list_targets()


def test_the_suite_never_sees_a_developers_own_targets():
    """Guards the import-time isolation in tests/conftest.py.

    `tests/targets/test_engine_env.py` parametrizes over `list_targets()` inside a decorator, which
    runs at COLLECTION -- before any fixture. Isolating only in a session fixture meant a developer
    who followed the documented setup (`memrank config set targets.path ~/evals`) would watch this
    suite fail on targets memrank has never heard of, with no env var exported and no obvious
    cause. Deliberately takes no fixture: it must observe the ambient state.
    """
    assert c.path_dirs() == []
    assert set(c.origins().values()) == {c.builtin_dir()}


def test_a_relative_entry_is_refused_rather_than_read_from_the_cwd(tmp_path, monkeypatch):
    """A relative entry points at a different directory from every folder memrank runs in, and
    a missing directory loads as zero targets -- so the failure mode is silent vanishing. Refused
    loudly instead, naming the entry and the fix."""
    monkeypatch.setenv("MEMRANK_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("MEMRANK_TARGETS_PATH", "./evals")

    with pytest.raises(ManifestError, match="relative"):
        c.path_dirs()


def test_a_path_set_from_one_directory_works_from_another(tmp_path, monkeypatch):
    """The reported bug: `config set targets.path ./dir` on one box, then every command run from
    any other folder found nothing. Set-time resolution makes the stored value cwd-independent."""
    from memrank import settings

    directory = tmp_path / "evals" / "targets"
    directory.mkdir(parents=True)
    (directory / "myengine.yaml").write_text(DESCRIPTOR, encoding="utf-8")
    monkeypatch.setenv("MEMRANK_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.delenv("MEMRANK_TARGETS_PATH", raising=False)

    monkeypatch.chdir(tmp_path / "evals")
    settings.put("targets.path", "targets")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    assert "myengine:dev" in c.list_targets()


def test_origins_names_where_each_target_came_from(tmp_path, monkeypatch):
    directory = evals_dir(tmp_path, monkeypatch)

    where = c.origins()

    assert where["myengine:dev"] == directory
    withheld.require("mem0")
    assert where["mem0"] == c.builtin_dir()
