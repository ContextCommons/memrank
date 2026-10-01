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
"""``memrank config`` -- round-trip, and the source column that makes surprises explicable."""
from __future__ import annotations

import pytest
from typer.testing import CliRunner

from memrank import settings
from memrank.runner import app

runner = CliRunner()


@pytest.fixture
def store(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMRANK_CONFIG_DIR", str(tmp_path))
    for spec in settings.SETTINGS:
        monkeypatch.delenv(spec.env, raising=False)
    return tmp_path


def test_set_then_get_round_trips(store):
    assert runner.invoke(app, ["config", "set", "defaults.org", "acme"]).exit_code == 0
    result = runner.invoke(app, ["config", "get", "defaults.org"])
    assert result.exit_code == 0
    assert result.output.strip() == "acme"


def test_ls_shows_every_key_with_its_source(store):
    runner.invoke(app, ["config", "set", "defaults.org", "acme"])
    output = runner.invoke(app, ["config", "ls"]).output
    for spec in settings.SETTINGS:
        assert spec.key in output
    # Source is its own SOURCE column now rather than a parenthesised suffix.
    sources = {line.split()[2] for line in output.splitlines()[1:] if line.split()}
    assert "file" in sources           # the one just written
    assert "default" in sources        # the ones nothing has answered


def test_ls_attributes_an_environment_value_to_the_environment(store, monkeypatch):
    monkeypatch.setenv("MEMRANK_ORG", "from-the-env")
    output = runner.invoke(app, ["config", "ls"]).output
    row = next(line for line in output.splitlines() if "from-the-env" in line)
    assert row.split()[2] == "env"


def test_setting_a_key_the_environment_overrides_says_so(store, monkeypatch):
    """Storing a value that cannot take effect is exactly the confusion `ls` exists to prevent."""
    monkeypatch.setenv("MEMRANK_ORG", "wins")
    result = runner.invoke(app, ["config", "set", "defaults.org", "acme"])
    assert result.exit_code == 0
    assert "MEMRANK_ORG" in result.output


def test_an_unknown_key_is_refused_by_name(store):
    result = runner.invoke(app, ["config", "set", "defaults.orgg", "acme"])
    assert result.exit_code != 0
    assert "defaults.org" in result.output       # the known keys are listed
    assert not settings.store_path().exists()


def test_get_of_an_unanswered_key_exits_non_zero(store):
    """Scripts branch on this; printing an empty line and exiting 0 would read as success."""
    assert runner.invoke(app, ["config", "get", "defaults.org"]).exit_code == 1
