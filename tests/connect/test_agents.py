"""Agent names resolve only to files shipped in the package; anything else is a path."""

from __future__ import annotations

import pytest

from memrank import agents
from memrank.connect.base import ConnectorConfigError
from memrank.connect.spec import load_agent


def test_full_context_ships_and_starts_itself():
    spec = load_agent(agents.resolve("full-context"))
    assert spec.ref.name == "full-context" and spec.start is not None
    assert spec.start.argv[:3] == ["{python}", "-m", "memrank.runner"]
    assert spec.start.ready.endswith("/health")


def test_every_shipped_agent_loads():
    assert agents.shipped()
    for path in agents.shipped().values():
        load_agent(path)


def test_a_path_is_a_path(tmp_path):
    spec = tmp_path / "mine.yaml"
    spec.write_text("name: mine\n")
    assert agents.resolve(str(spec)) == spec


def test_anything_else_is_refused_naming_what_ships():
    with pytest.raises(ConnectorConfigError, match="full-context"):
        agents.resolve("mem0")
