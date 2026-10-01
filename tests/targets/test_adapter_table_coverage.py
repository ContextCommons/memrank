"""Every registered adapter must appear in every table keyed by adapter name.

Adding an engine means filling roughly eight name-keyed tables (docs/systems.md). Each one
raises loudly when its own entry is missing, but only at the moment it is read -- which for some
is a cloud render nobody runs locally, and for READINESS is a health check that never passes
while the harness waits for a task that will be killed.

Derived from REGISTRY rather than a hand-written list on purpose: a list would have to be edited by
the same person who just forgot the table. This is the enumeration test the chokepoint rule asks
for -- it fails when a new adapter lacks an entry, naming the table and the adapter.
"""

from __future__ import annotations

import pytest

from memrank.adapters import REGISTRY
from memrank.provenance.engine import PROVENANCE
from memrank.secrets.requirements import REQUIREMENTS
from memrank.targets.engine_env import (
    BASE_URL_ENV,
    ENGINE_COMMAND,
    ENGINE_ENV,
    ENGINE_SETTINGS,
    PROVENANCE_FIELDS,
    READINESS,
)

#: Tables that describe how to reach and launch an engine PROCESS. In-process arms have none.
_ENGINE_TABLES = {
    "ENGINE_ENV": ENGINE_ENV,
    "BASE_URL_ENV": BASE_URL_ENV,
    "ENGINE_SETTINGS": ENGINE_SETTINGS,
    "ENGINE_COMMAND": ENGINE_COMMAND,
    "READINESS": READINESS,
    "PROVENANCE_FIELDS": PROVENANCE_FIELDS,
    "PROVENANCE": PROVENANCE,
}


def _out_of_process() -> list[str]:
    """Adapters that drive a separate engine process, so the engine tables apply to them."""
    return sorted(name for name, cls in REGISTRY.items()
                  if getattr(cls, "transport", None) != "in-process")


@pytest.mark.parametrize("table_name", sorted(_ENGINE_TABLES))
def test_every_out_of_process_adapter_has_a_table_entry(table_name: str):
    table = _ENGINE_TABLES[table_name]
    missing = [name for name in _out_of_process() if name not in table]
    assert not missing, (
        f"{table_name} has no entry for {', '.join(missing)}. Absence is not a default: add the "
        f"entry (an empty dict or None is a positive statement) -- see docs/systems.md.")


def test_every_adapter_declares_its_launch_requirements():
    """Including the in-process arms: `none` and `icl` spend nothing, and saying so is the point."""
    missing = [name for name in sorted(REGISTRY) if name not in REQUIREMENTS]
    assert not missing, (
        f"REQUIREMENTS has no entry for {', '.join(missing)}; an engine with no declared "
        f"requirements cannot be preflighted, so a run fails at launch instead of at submission.")


def test_native_is_registered_but_names_no_vendor():
    """The whole point of the native adapter: it is the one entry that is not an engine.

    If this ever gains vendor-specific values, the translator contract has leaked back into
    memrank and an engine memrank has never seen is no longer expressible.
    """
    assert "native" in REGISTRY
    assert ENGINE_ENV["native"] == {}
    assert ENGINE_SETTINGS["native"] == {}
    assert ENGINE_COMMAND["native"] is None
    assert PROVENANCE["native"]["source_repo"] is None
    assert REQUIREMENTS["native"].providers == {}
