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
"""Reading PEP 610 back: the four install shapes a version line has to survive.

The metadata is faked, not the reader -- each test hands the real code the exact
``direct_url.json`` that pip or uv would have written for that kind of install, so the
parsing under test is the parsing that runs.
"""
from __future__ import annotations

import importlib.metadata
import json

import pytest

from memrank import __version__
from memrank.provenance import install

_VCS = {"url": "https://github.com/ContextCommons/memrank",
        "vcs_info": {"vcs": "git", "requested_revision": "dev",
                     "commit_id": "1537ebb0c0ffee0000000000000000000000dead"}}
_EDITABLE = {"url": "file:///home/dev/memrank", "dir_info": {"editable": True}}
_WHEEL_URL = {"url": "https://example.invalid/memrank-0.2.0-py3-none-any.whl", "dir_info": {}}


class _FakeDistribution:
    """Enough of ``importlib.metadata.Distribution`` for the one file we read."""

    def __init__(self, record: dict | None):
        self._record = record

    def read_text(self, filename: str) -> str | None:
        return json.dumps(self._record) if self._record is not None else None


@pytest.fixture
def installed_as(monkeypatch):
    """Present memrank as though installed from `record` (``None`` = no PEP 610 file)."""
    def _install(record: dict | None):
        monkeypatch.setattr(importlib.metadata.Distribution, "from_name",
                            staticmethod(lambda name: _FakeDistribution(record)))
    return _install


def test_a_git_install_is_identified_by_its_resolved_commit(installed_as):
    """The ref orients, the commit identifies -- `dev` alone cannot name a build."""
    installed_as(_VCS)

    origin = install.describe_origin()

    assert origin == ("git+https://github.com/ContextCommons/memrank@dev, commit 1537ebb")


def test_an_editable_install_says_which_tree_it_tracks(installed_as):
    """Its code is whatever is on disk, so the path IS the version."""
    installed_as(_EDITABLE)

    assert install.describe_origin() == "editable, file:///home/dev/memrank"


def test_a_plain_url_install_reports_the_url_it_came_from(installed_as):
    installed_as(_WHEEL_URL)

    assert install.describe_origin() == _WHEEL_URL["url"]


def test_an_install_with_no_direct_url_reports_no_origin(installed_as):
    """A wheel off an index has no PEP 610 record. A fact about the install, not an error."""
    installed_as(None)

    assert install.describe_origin() is None
    assert install.version_line() == __version__


def test_the_version_line_carries_both_the_version_and_the_origin(installed_as):
    installed_as(_VCS)

    line = install.version_line()

    assert line.startswith(__version__)
    assert "commit 1537ebb" in line


def test_corrupt_metadata_fails_loudly_rather_than_reporting_less(installed_as, monkeypatch):
    """A quietly shorter version line is how an unidentifiable build gets shipped."""
    monkeypatch.setattr(importlib.metadata.Distribution, "from_name",
                        staticmethod(lambda name: _Corrupt()))

    with pytest.raises(json.JSONDecodeError):
        install.describe_origin()


class _Corrupt:
    def read_text(self, filename: str) -> str:
        return "{not json"
