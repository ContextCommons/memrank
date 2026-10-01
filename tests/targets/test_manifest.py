"""Validation rules for target manifests."""
from __future__ import annotations

from pathlib import Path

import pytest

from memrank.targets import manifest as m

BASE = {"name": "mem0", "kind": "stack", "adapter": "mem0"}
SOURCE = {
    "schema_version": 1,
    "name": "myengine:dev",
    "kind": "stack",
    "interface": {"adapter": "myengine", "transport": "http"},
    "binding": {"kind": "source", "root": "/work/myengine"},
    "launch": {"command": "cargo run -- --bind 127.0.0.1:{port}",
               "requires": ["Cargo.toml"]},
    "network": {"port": 8080, "readiness": {"path": "/health"}},
    "components": {"llm": {"provider": "regex"}},
}


def test_minimal_stack_manifest_parses():
    got = m.from_dict(BASE)
    assert got.name == "mem0"
    assert got.kind == "stack"
    assert got.adapter == "mem0"
    assert got.depends == ()
    assert got.components == {}


def test_unknown_kind_is_rejected():
    with pytest.raises(m.ManifestError, match="unknown kind 'daemon'"):
        m.from_dict({**BASE, "kind": "daemon"})


def test_in_process_may_not_declare_depends():
    with pytest.raises(m.ManifestError, match="in-process.*may not declare depends"):
        m.from_dict({"name": "word-overlap", "kind": "in-process",
                     "adapter": "word-overlap", "depends": ["pgvector"]})


def test_embedder_model_without_dims_is_rejected():
    """dims never follows from the model name, so a silent stale value is the trap."""
    with pytest.raises(m.ManifestError, match=r"components\.embedder\.dims is required"):
        m.from_dict({**BASE, "components": {
            "embedder": {"provider": "voyage", "model": "voyage-4-large"}}})


def test_unknown_component_role_is_rejected():
    with pytest.raises(m.ManifestError, match="unknown component role 'reranker'"):
        m.from_dict({**BASE, "components": {"reranker": {"provider": "cohere"}}})


def test_transport_defaults_to_none_and_round_trips():
    assert m.from_dict(BASE).transport is None
    assert m.from_dict({**BASE, "transport": "http"}).transport == "http"


def test_unknown_transport_is_rejected():
    with pytest.raises(m.ManifestError, match="unknown transport 'grpc'"):
        m.from_dict({**BASE, "transport": "grpc"})


def test_components_round_trip():
    got = m.from_dict({**BASE, "depends": ["pgvector"], "components": {
        "embedder": {"provider": "voyage", "model": "voyage-4-large", "dims": 1024},
        "llm": {"provider": "anthropic", "model": "claude-sonnet-4-5"}}})
    assert got.depends == ("pgvector",)
    assert got.components["embedder"].dims == 1024
    assert got.components["llm"].provider == "anthropic"


def test_complete_source_target_parses_as_one_configuration():
    got = m.from_dict(SOURCE)

    assert got.binding == m.SourceBinding("source", "/work/myengine")
    assert got.launch == m.Workspace(
        "cargo run -- --bind 127.0.0.1:{port}", ("Cargo.toml",))
    assert got.network == m.Network(8080, m.Readiness("/health"))
    assert m.to_dict(got)["interface"] == {"adapter": "myengine", "transport": "http"}
    assert m.to_dict(got, include_local_binding=False)["binding"] == {"kind": "source"}
    assert "/work/myengine" not in repr(m.to_dict(got, include_local_binding=False))


def test_local_definition_digest_detects_binding_changes():
    first = m.definition_digest(m.from_dict(SOURCE))
    moved = m.from_dict({**SOURCE, "binding": {"kind": "source", "root": "/other/myengine"}})

    assert m.definition_digest(moved) != first


def test_definition_digest_covers_every_target_in_a_sweep():
    """Digesting only the first would leave the rest free to change between submit and execution."""
    one = m.from_dict(SOURCE)
    two = m.from_dict({**SOURCE, "name": "myengine:other"})

    assert m.definition_digest(one, two) != m.definition_digest(one)
    assert m.definition_digest(one, two) != m.definition_digest(two, one)


@pytest.mark.parametrize("root, expected", [
    (None, "/evals"),                       # omitted -> the descriptor's own directory
    (".", "/evals"),
    ("engines/myengine", "/evals/engines/myengine"),
    ("/somewhere/else", "/somewhere/else"),  # absolute stays absolute
])
def test_binding_root_resolves_against_the_descriptor_directory(root, expected):
    """What lets a descriptor sit beside its translator and be copied or shared verbatim: an
    absolute path baked into the file is true on exactly one machine."""
    binding = {"kind": "source"} if root is None else {"kind": "source", "root": root}

    got = m.from_dict({**SOURCE, "binding": binding}, base_dir=Path("/evals"))

    assert got.binding is not None
    assert got.binding.root == expected


def test_a_relative_root_needs_a_directory_to_resolve_against():
    """Built from a mapping rather than read from disk, there is nothing to resolve against -- and
    a cwd-relative guess would mean a different directory depending on where the command ran."""
    with pytest.raises(m.ManifestError, match="no file to resolve it against"):
        m.from_dict({**SOURCE, "binding": {"kind": "source", "root": "relative"}})


@pytest.mark.parametrize("change, message", [
    ({"schema_version": 2}, "schema_version"),
    ({"binding": {"kind": "source", "root": 7}}, "path string"),
    ({"network": {"port": 0, "readiness": {"path": "/health"}}}, "network.port"),
    ({"network": {"port": 8080, "readiness": {"path": "health"}}}, "must start"),
    ({"engine": {"artifact": "myengine"}}, "may not declare legacy"),
])
def test_invalid_source_target_is_rejected(change, message):
    with pytest.raises(m.ManifestError, match=message):
        m.from_dict({**SOURCE, **change})


def test_workspace_command_is_a_readable_scalar():
    got = m.from_dict({**BASE, "workspace": {
        "command": "cargo run -p engine -- --port {port}",
        "requires": ["Cargo.toml"],
    }})

    assert got.workspace is not None
    assert got.workspace.command.startswith("cargo run")
    assert got.workspace.requires == ("Cargo.toml",)


@pytest.mark.parametrize("requires", [None, []])
def test_workspace_requires_is_optional(requires):
    """`requires` buys an early wrong-checkout error; a launcher with no distinctive marker file
    legitimately has none. It used to be rejected in both these forms -- `raw.get(...) or ()` turned
    an omitted key AND an explicit empty list into a tuple, then refused them for not being a list.
    """
    workspace = {"command": "python translator.py --port {port}"}
    if requires is not None:
        workspace["requires"] = requires

    got = m.from_dict({**BASE, "workspace": workspace})

    assert got.workspace is not None
    assert got.workspace.requires == ()


@pytest.mark.parametrize("workspace, message", [
    ({"command": ["cargo", "run"]}, "non-empty string"),
    ({"command": "cargo run -- {host}"}, "unknown placeholder"),
    ({"command": "cargo run", "requires": ["../other"]}, "must stay within"),
])
def test_invalid_workspace_declarations_are_rejected(workspace, message):
    with pytest.raises(m.ManifestError, match=message):
        m.from_dict({**BASE, "workspace": workspace})
