# Copyright 2026 AtomicStrata
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.
"""Turning PEP 610 back into an instruction the reader can actually type.

An upgrade instruction is only as good as its fit: handing a released install a
``git+https://...`` command replaces that release with a branch checkout, and handing a uv
tool environment a ``pip install`` upgrades a copy the ``memrank`` on PATH does not run. Both
are ways of answering a question nobody asked, so the shape of the install is asserted here
rather than assumed.

The metadata is faked, not the reader -- each test hands the real code the
``direct_url.json`` pip or uv would have written, and the real installed location.
"""
from __future__ import annotations

import importlib.metadata
import json

import pytest

from memrank.provenance import install

_VCS = {"url": "https://github.com/ContextCommons/memrank",
        "vcs_info": {"vcs": "git", "requested_revision": "dev",
                     "commit_id": "1537ebb0c0ffee0000000000000000000000dead"}}
_EDITABLE = {"url": "file:///home/dev/memrank", "dir_info": {"editable": True}}

_VENV = "/home/user/project/.venv/lib/python3.12/site-packages"
_UV_TOOL = "/home/user/.local/share/uv/tools/memrank/lib/python3.12/site-packages"


class _FakeDistribution:
    """Enough of ``importlib.metadata.Distribution`` for the record and the location."""

    def __init__(self, record: dict | None, location: str):
        self._record = record
        self._location = location

    def read_text(self, filename: str) -> str | None:
        return json.dumps(self._record) if self._record is not None else None

    def locate_file(self, path: str) -> str:
        return self._location


@pytest.fixture
def installed_as(monkeypatch):
    """Present memrank as installed from `record` (``None`` = a registry wheel), at `location`."""
    def _install(record: dict | None, location: str = _VENV):
        monkeypatch.delenv("UV_TOOL_DIR", raising=False)
        monkeypatch.setattr(importlib.metadata.Distribution, "from_name",
                            staticmethod(lambda name: _FakeDistribution(record, location)))
    return _install


def test_a_released_install_is_never_told_to_fetch_a_branch(installed_as):
    """The defect this exists to prevent: a wheel user handed a git install."""
    installed_as(None)

    instruction = install.upgrade_instruction()

    assert instruction == "pip install --upgrade memrank"
    assert "git+" not in instruction


def test_a_uv_tool_install_upgrades_the_environment_that_owns_the_binary(installed_as):
    """`pip install --upgrade` inside a tool environment upgrades a copy PATH does not run."""
    installed_as(None, location=_UV_TOOL)

    assert install.upgrade_instruction() == "uv tool upgrade memrank"


def test_uv_tool_dir_is_honoured_when_it_moves_the_tool_root(installed_as, monkeypatch):
    """uv's tool root is relocatable, and the path shape is only the default."""
    installed_as(None, location="/opt/tools/memrank/lib/python3.12/site-packages")
    monkeypatch.setenv("UV_TOOL_DIR", "/opt/tools")

    assert install.upgrade_instruction() == "uv tool upgrade memrank"


def test_a_source_install_is_upgraded_from_the_ref_it_chose(installed_as):
    """Never a ref named by us: an upgrade may not move somebody off their own branch."""
    installed_as(_VCS, location=_UV_TOOL)

    instruction = install.upgrade_instruction()

    assert instruction == ("uv tool install --force --refresh "
                           "git+https://github.com/ContextCommons/memrank@dev")


def test_a_source_install_outside_a_tool_environment_upgrades_with_pip(installed_as):
    """`--force-reinstall` because a moved branch resolves to the same cached version."""
    installed_as(_VCS)

    assert install.upgrade_instruction() == (
        "pip install --upgrade --force-reinstall "
        "git+https://github.com/ContextCommons/memrank@dev")


def test_an_editable_install_is_told_to_move_its_tree_not_to_reinstall(installed_as):
    """Its code IS the tree, so any reinstall instruction would be a no-op dressed as a fix."""
    installed_as(_EDITABLE)

    instruction = install.upgrade_instruction()

    assert "/home/dev/memrank" in instruction
    assert "pip install" not in instruction and "uv tool" not in instruction


def test_a_refusal_that_asks_for_an_upgrade_is_completed_with_this_install_s_own(installed_as):
    """The platform asks; the machine that knows how it was installed answers."""
    installed_as(None)

    completed = install.with_upgrade_instruction(f"stale CLI -- {install.UPGRADE_REQUEST}")

    assert completed == "stale CLI -- upgrade the memrank CLI: pip install --upgrade memrank"


def test_a_refusal_about_anything_else_is_passed_through_untouched(installed_as):
    """Most refusals are not about version skew and must not grow upgrade advice."""
    installed_as(None)

    assert install.with_upgrade_instruction("no quota left") == "no quota left"


def test_a_missing_extra_names_an_install_command_that_fits_the_reader(installed_as):
    installed_as(None)

    assert install.dependency_instruction("mem0ai", "mem0") == (
        "pip install --upgrade 'memrank[mem0]'")


def test_a_missing_extra_in_a_checkout_syncs_rather_than_installs(installed_as):
    """An editable install has a pyproject.toml beside it; a released one does not."""
    installed_as(_EDITABLE)

    assert install.dependency_instruction("mem0ai", "mem0") == "uv sync --extra mem0"


def test_a_missing_extra_in_a_tool_environment_carries_the_source_it_came_from(installed_as):
    """`uv tool install --with` re-resolves the tool, so it has to name the same source."""
    installed_as(_VCS, location=_UV_TOOL)

    assert install.dependency_instruction("mem0ai", "mem0") == (
        "uv tool install --force --with mem0ai "
        "git+https://github.com/ContextCommons/memrank@dev")


def test_a_missing_base_dependency_repairs_the_install_it_is_missing_from(installed_as):
    """No extra to name: the install is incomplete rather than partial."""
    installed_as(None)

    assert install.dependency_instruction("tiktoken") == "pip install --force-reinstall memrank"
