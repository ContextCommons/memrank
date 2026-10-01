"""Native source bindings stay isolated from artifact-backed evaluator runs."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from memrank import runner
from memrank.placement.base import PlacementError
from memrank.placement.workspace import WorkspacePlacement, _command, _source_facts
from memrank.provenance.engine import build_provenance
from memrank.targets.manifest import (
    Engine,
    Manifest,
    Network,
    Readiness,
    SourceBinding,
    Workspace,
)


def _target(root: Path, command: str = "cargo run -- --port {port}") -> Manifest:
    return Manifest(name="atomicmemory", kind="stack", adapter="atomicmemory",
                    engine=Engine(port=8000), binding=SourceBinding("source", str(root)),
                    launch=Workspace(command, ("Cargo.toml",)),
                    network=Network(8000, Readiness("/health")))


def test_scalar_command_becomes_direct_exec_argv(tmp_path):
    target = _target(tmp_path, "cargo run -p engine -- --bind '127.0.0.1:{port}'")
    assert _command(target) == [
        "cargo", "run", "-p", "engine", "--", "--bind", "127.0.0.1:8000"]


@pytest.mark.parametrize("where", ["cloud", "none"])
def test_a_source_target_cannot_run_anywhere_but_local(tmp_path, monkeypatch, where):
    """The reproducibility gate: a directory on this machine is not a pinned artifact."""
    target = _target(tmp_path)
    monkeypatch.setattr("memrank.targets.resolve_target", lambda ref, overrides: target)
    with pytest.raises(Exception, match="require --on local"):
        runner._validate_source_target(
            explicit_on=where, on=where, factories=[("atomicmemory", lambda: None)], overrides=[])


def test_a_source_target_may_sweep_against_the_catalog(tmp_path, monkeypatch):
    """Comparing a freshly-translated engine against the catalog is the point of having one.

    Previously refused outright. Nothing was protected by that: a sweep tears each target down
    before the next begins, so two source targets never hold a port at once.
    """
    target = _target(tmp_path)
    monkeypatch.setattr("memrank.targets.resolve_target", lambda ref, overrides: target)
    factory = [("atomicmemory:dev", lambda: None)]

    where, got = runner._validate_source_target(
        explicit_on="local", on="local", factories=[*factory, *factory], overrides=[])

    assert where == "local"
    assert got == [target, target]


def test_named_source_target_selects_local_without_an_on_flag(tmp_path, monkeypatch):
    target = _target(tmp_path)
    monkeypatch.setattr("memrank.targets.resolve_target", lambda ref, overrides: target)
    where, got = runner._validate_source_target(
        explicit_on=None, on="cloud", factories=[("atomicmemory:dev", lambda: None)], overrides=[])
    assert where == "local"
    assert got == [target]


def test_check_names_a_wrong_checkout(tmp_path, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: f"/bin/{name}")
    rows = WorkspacePlacement(target=_target(tmp_path)).check()

    assert any(row.name == "Cargo.toml" and not row.ok for row in rows)


def _git(tmp_path: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)


def test_source_fingerprint_changes_for_dirty_and_untracked_files(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "Test")
    tracked = tmp_path / "engine.txt"
    tracked.write_text("one", encoding="utf-8")
    _git(tmp_path, "add", "engine.txt")
    _git(tmp_path, "commit", "-qm", "initial")
    _, clean, first = _source_facts(tmp_path)
    tracked.write_text("two", encoding="utf-8")
    (tmp_path / "new.txt").write_text("three", encoding="utf-8")
    _, dirty, second = _source_facts(tmp_path)

    assert clean is False and dirty is True
    assert first != second


def test_an_unversioned_directory_is_run_and_recorded_as_such(tmp_path):
    """Version control is a fact about the directory, not a precondition for running from it.

    Requiring `git init` on a scratch folder buys nothing -- a single-commit throwaway SHA
    identifies no more than a content hash would -- while blocking the case this placement exists
    for: evaluating something handed to you five minutes ago.
    """
    (tmp_path / "translator.py").write_text("print('hi')", encoding="utf-8")

    sha, dirty, delta = _source_facts(tmp_path)

    assert sha is None
    assert (dirty, delta) == (False, "")


def test_an_unversioned_run_omits_the_commit_rather_than_faking_one(tmp_path, monkeypatch):
    """An empty SHA would read as a value, and "clean" would be a claim about an untracked tree."""
    target = _target(tmp_path)
    placement = WorkspacePlacement(target=target)
    monkeypatch.setattr(placement, "check", lambda: [])
    monkeypatch.setattr(placement, "_ensure_port_free", lambda: None)
    monkeypatch.setattr(placement, "_start", lambda: None)
    monkeypatch.setattr(placement, "_wait_ready", lambda url: None)
    monkeypatch.setattr(
        "memrank.placement.workspace._source_facts", lambda source: (None, False, ""))

    env = placement.provision(target).adapter_env
    provenance = build_provenance("atomicmemory", env)
    properties = provenance["properties"]

    # No commit is invented. The harness env carries an empty string, which `_env` normalizes to
    # absent (engine_provenance.py:162) -- the same convention it already uses for an image with no
    # multi-arch index. What matters is the receipt, and the receipt says nothing it cannot support.
    assert not env.get("ATOMICMEMORY_ENGINE_SOURCE_SHA")
    assert "ATOMICMEMORY_ENGINE_SOURCE_DIRTY" not in env
    assert properties["memrank:workspace_unversioned"] == "true"
    assert "memrank:source_sha" not in properties
    assert "memrank:source_dirty" not in properties
    assert provenance["purl"] is None and provenance["generatedFrom"] is None
    # Still identifies WHAT was launched, and still cannot be published.
    assert len(env["ATOMICMEMORY_ENGINE_COMMAND_SHA256"]) == 64
    assert env["MEMRANK_EVIDENCE_CLASS"] == "development_observation"


def test_workspace_endpoint_is_directional_and_has_no_source_path(tmp_path, monkeypatch):
    target = _target(tmp_path)
    placement = WorkspacePlacement(target=target)
    monkeypatch.setattr(placement, "check", lambda: [])
    monkeypatch.setattr(placement, "_ensure_port_free", lambda: None)
    monkeypatch.setattr(placement, "_start", lambda: None)
    monkeypatch.setattr(placement, "_wait_ready", lambda url: None)
    monkeypatch.setattr(
        "memrank.placement.workspace._source_facts", lambda source: ("a" * 40, True, "b" * 64))

    endpoint = placement.provision(target)

    assert endpoint.adapter_env["MEMRANK_EVIDENCE_CLASS"] == "development_observation"
    assert endpoint.adapter_env["ATOMICMEMORY_ENGINE_SOURCE_DIRTY"] == "true"
    assert len(endpoint.adapter_env["ATOMICMEMORY_ENGINE_COMMAND_SHA256"]) == 64
    assert str(tmp_path) not in repr(endpoint)
    assert build_provenance("atomicmemory", endpoint.adapter_env)["type"] == "application"


def test_port_conflict_is_a_loud_refusal(monkeypatch, tmp_path):
    class BusySocket:
        def bind(self, address):
            raise OSError("busy")

        def close(self):
            pass

    monkeypatch.setattr("socket.socket", lambda *args: BusySocket())
    placement = WorkspacePlacement(target=_target(tmp_path))
    with pytest.raises(PlacementError, match="already in use"):
        placement._ensure_port_free()
