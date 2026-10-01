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
"""Standing preferences: precedence, provenance, and refusing to store nonsense."""
from __future__ import annotations

import os
import pathlib
import stat

import pytest

from memrank import settings


@pytest.fixture
def store(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMRANK_CONFIG_DIR", str(tmp_path))
    for spec in settings.SETTINGS:
        monkeypatch.delenv(spec.env, raising=False)
    return tmp_path


def test_an_unset_key_reports_its_shipped_default(store):
    assert settings.resolve("defaults.on") == ("none", settings.DEFAULT)


def test_placement_ships_local_not_cloud(store):
    """Logged out, everything must still work. A shipped cloud default would make a fresh
    clone's first submission fail on authentication -- `auth login` writes cloud instead."""
    assert settings.get("defaults.on") == "none"


def test_auto_sync_ships_on(store):
    """Signed in, a finished run reaches the org without being asked; signed out the hook is
    silent, so shipping this on costs local-first use nothing."""
    assert settings.get("sync.auto") == "true"


def test_a_stored_value_beats_the_default_and_says_so(store):
    settings.put("defaults.org", "acme")
    assert settings.resolve("defaults.org") == ("acme", settings.FILE)


def test_the_environment_beats_the_file_and_says_so(store, monkeypatch):
    """`MEMRANK_API_URL=... memrank ...` is how every runbook line works; it must keep winning."""
    settings.put("defaults.org", "acme")
    monkeypatch.setenv("MEMRANK_ORG", "other")
    assert settings.resolve("defaults.org") == ("other", settings.ENV)


def test_an_unknown_key_is_refused_rather_than_stored(store):
    """A key that stores happily but is never read is a preference that silently never applies."""
    with pytest.raises(settings.UnknownSetting):
        settings.put("defaults.orgg", "acme")
    with pytest.raises(settings.UnknownSetting):
        settings.resolve("nope")
    assert not settings.store_path().exists()


def test_the_file_is_private_to_its_owner(store):
    settings.put("defaults.org", "acme")
    mode = stat.S_IMODE(settings.store_path().stat().st_mode)
    assert mode == 0o600


def test_writes_accumulate_rather_than_replace(store):
    settings.put("defaults.org", "acme")
    settings.put("defaults.on", "cloud")
    assert settings.get("defaults.org") == "acme"
    assert settings.get("defaults.on") == "cloud"


def test_resolved_covers_every_key(store):
    assert [spec.key for spec, _, _ in settings.resolved()] == [s.key for s in settings.SETTINGS]


def test_a_path_setting_is_stored_absolute(store, monkeypatch, tmp_path):
    """A relative path means "this directory, from here" at the moment it is typed. Stored
    verbatim it would mean "wherever you happen to be standing", which is the bug this test
    pins: the value must be resolved against the cwd at set time, then never move again."""
    workdir = tmp_path / "work"
    workdir.mkdir()
    monkeypatch.chdir(workdir)

    stored = settings.put("targets.path", os.pathsep.join(["./evals", "~/evals"]))

    expected = os.pathsep.join(
        [str((workdir / "evals").resolve()), str(pathlib.Path("~/evals").expanduser().resolve())])
    assert stored == expected
    assert settings.resolve("targets.path") == (expected, settings.FILE)


def test_a_non_path_setting_is_stored_verbatim(store):
    assert settings.put("defaults.org", "./acme") == "./acme"
    assert settings.get("defaults.org") == "./acme"
