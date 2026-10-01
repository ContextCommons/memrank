"""A target names the credentials it needs; memrank delivers them and interprets nothing.

The behaviour these pin is a deliberate removal of knowledge. Before, what a target needed was
DERIVED from its provider through a table memrank owned, so "one vendor, one key" was the only
expressible shape -- an engine wanting a user and a password, or a token under a name nobody had
added to the table, simply could not be described. Resolution was already name-agnostic
(`config.secret` takes any string); only the guessing was hardcoded.
"""

from __future__ import annotations

import pytest

from memrank.placement.local import engine_secrets
from memrank.targets.catalog import required_secrets_for
from memrank.targets.manifest import ManifestError, from_dict, to_dict

STACK = {"name": "t", "kind": "stack", "adapter": "atomicmemory"}


def target(**overrides):
    return from_dict({**STACK, **overrides})


def test_a_list_means_the_engine_reads_the_same_names():
    got = target(secrets=["LLM_API_KEY", "EMBEDDING_API_KEY"])

    assert got.secrets == {"LLM_API_KEY": ["LLM_API_KEY"],
                           "EMBEDDING_API_KEY": ["EMBEDDING_API_KEY"]}


def test_a_mapping_expresses_a_rename_without_memrank_holding_a_table():
    """The shape SECRET_ENV exists for: hindsight reads its LLM key under its own name."""
    got = target(secrets={"ANTHROPIC_API_KEY": "HINDSIGHT_API_LLM_API_KEY"})

    assert got.secrets == {"ANTHROPIC_API_KEY": ["HINDSIGHT_API_LLM_API_KEY"]}


def test_one_credential_may_feed_several_variables():
    """A mapping to a single string could not say this: both entries would need the same key."""
    got = target(secrets={"AM_SLM_TOKEN": ["LLM_API_KEY", "EMBEDDING_API_KEY"]})

    assert got.secrets == {"AM_SLM_TOKEN": ["LLM_API_KEY", "EMBEDDING_API_KEY"]}


def test_a_credential_that_is_not_one_key_is_expressible():
    """The case the provider table could not describe at all, and the reason this field exists."""
    got = target(secrets=["MYENGINE_USER", "MYENGINE_PASSWORD", "MYENGINE_TENANT"])

    assert sorted(got.secrets) == ["MYENGINE_PASSWORD", "MYENGINE_TENANT", "MYENGINE_USER"]


@pytest.mark.parametrize("raw, message", [
    ("LLM_API_KEY", "list of names or a mapping"),
    (["", "OK"], "non-empty env-var names"),
    ([None], "non-empty env-var names"),
    ({"NAME": ""}, "must name the variable"),
    ({"NAME": 7}, "must name the variable"),
    ({"NAME": ["OK", ""]}, "must name the variable"),
    ({"NAME": []}, "names no variable"),
])
def test_malformed_declarations_are_rejected(raw, message):
    with pytest.raises(ManifestError, match=message):
        target(secrets=raw)


def test_an_in_process_arm_may_not_declare_secrets():
    """There is no engine process to inject into -- the control arms run inside memrank itself."""
    with pytest.raises(ManifestError, match="no engine process"):
        from_dict({"name": "b", "kind": "in-process", "adapter": "word-overlap",
                   "secrets": ["ANYTHING"]})


def test_declared_secrets_add_to_what_providers_imply():
    """Union, not replacement: atomicmemory's anthropic LLM still implies its key."""
    needed = required_secrets_for(target(secrets=["LLM_API_KEY"]))

    assert "ANTHROPIC_API_KEY" in needed      # derived from the target's own components
    assert "LLM_API_KEY" in needed            # declared


def test_required_names_are_what_memrank_RESOLVES_not_what_the_engine_reads():
    """The wallet knows the left-hand name; the right-hand one only exists inside the engine."""
    needed = required_secrets_for(target(secrets={"ANTHROPIC_API_KEY": "SOMETHING_ELSE"}))

    assert "SOMETHING_ELSE" not in needed
    assert "ANTHROPIC_API_KEY" in needed


#: supermemory declares no launch providers, so nothing is DERIVED and these exercise only the
#: declared path. Using an engine with a derived key would trip the preflight on that key first.
KEYLESS = {"name": "t", "kind": "stack", "adapter": "supermemory"}


def test_values_are_resolved_and_injected_under_the_engines_names(monkeypatch):
    monkeypatch.setenv("MYENGINE_USER", "alice")
    monkeypatch.setenv("MYENGINE_PASSWORD", "s3cret")

    injected = engine_secrets(from_dict(
        {**KEYLESS, "secrets": {"MYENGINE_USER": "DB_USER", "MYENGINE_PASSWORD": "DB_PASS"}}))

    assert injected["DB_USER"] == "alice"
    assert injected["DB_PASS"] == "s3cret"


def test_one_resolved_value_reaches_every_variable_it_was_declared_for(monkeypatch):
    """One token, two roles: core reads it as LLM_API_KEY and again as EMBEDDING_API_KEY."""
    monkeypatch.setenv("AM_SLM_TOKEN", "edge-token")

    injected = engine_secrets(from_dict(
        {**KEYLESS, "secrets": {"AM_SLM_TOKEN": ["LLM_API_KEY", "EMBEDDING_API_KEY"]}}))

    assert injected["LLM_API_KEY"] == "edge-token"
    assert injected["EMBEDDING_API_KEY"] == "edge-token"


def test_an_unresolvable_secret_injects_nothing_rather_than_an_empty_value(monkeypatch):
    """An empty string is a value the engine would act on; absence is refused at preflight."""
    monkeypatch.delenv("NOT_ANYWHERE", raising=False)
    declared = from_dict({**KEYLESS, "secrets": ["NOT_ANYWHERE"]})

    assert "NOT_ANYWHERE" not in engine_secrets(declared)
    assert "NOT_ANYWHERE" in required_secrets_for(declared)


def test_a_manifest_carries_names_never_values(monkeypatch):
    """to_dict feeds receipts and the org sync, so a value here would leave the machine."""
    monkeypatch.setenv("LLM_API_KEY", "super-secret-token")

    serialized = to_dict(from_dict({
        "schema_version": 1, "name": "s", "kind": "stack",
        "interface": {"adapter": "native", "transport": "translator"},
        "binding": {"kind": "source", "root": "/tmp"},
        "launch": {"command": "run --port {port}"},
        "network": {"port": 9000, "readiness": {"path": "/health"}},
        "secrets": ["LLM_API_KEY"],
    }))

    assert serialized["secrets"] == {"LLM_API_KEY": ["LLM_API_KEY"]}
    assert "super-secret-token" not in repr(serialized)
