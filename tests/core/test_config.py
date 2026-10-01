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
"""Central config: app loads exactly one publisher DB URL; ingest URL is separate."""
import pytest

from memrank import config


def test_database_url_reads_env(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://pub:p@h:6543/db")
    assert config.database_url() == "postgresql://pub:p@h:6543/db"


def test_ingest_database_url_is_separate(monkeypatch):
    monkeypatch.setenv("INGEST_DATABASE_URL", "postgresql://ing:p@h:5432/db")
    assert config.ingest_database_url() == "postgresql://ing:p@h:5432/db"


def test_missing_urls_are_loud(monkeypatch):
    monkeypatch.setattr(config, "_DOTENV_LOADED", True)  # skip .env load in test
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("INGEST_DATABASE_URL", raising=False)
    with pytest.raises(config.ConfigError):
        config.database_url()
    with pytest.raises(config.ConfigError):
        config.ingest_database_url()


def test_artifact_bucket_returns_configured_name(monkeypatch):
    monkeypatch.setattr(config, "_DOTENV_LOADED", True)
    monkeypatch.setenv("LEADERBOARD_ARTIFACT_BUCKET", "my-bucket")
    assert config.artifact_bucket() == "my-bucket"


def test_artifact_bucket_names_missing_var(monkeypatch):
    monkeypatch.setattr(config, "_DOTENV_LOADED", True)
    monkeypatch.delenv("LEADERBOARD_ARTIFACT_BUCKET", raising=False)
    with pytest.raises(config.ConfigError) as exc_info:
        config.artifact_bucket()
    assert "LEADERBOARD_ARTIFACT_BUCKET" in str(exc_info.value)


def test_artifact_bucket_rejects_empty_value(monkeypatch):
    """An empty string is missing config, not a valid bucket -- must not pass silently."""
    monkeypatch.setattr(config, "_DOTENV_LOADED", True)
    monkeypatch.setenv("LEADERBOARD_ARTIFACT_BUCKET", "")
    with pytest.raises(config.ConfigError):
        config.artifact_bucket()


def test_fake_storage_enabled_true_only_on_exact_1(monkeypatch):
    monkeypatch.setattr(config, "_DOTENV_LOADED", True)
    monkeypatch.setenv("MEMRANK_FAKE_STORAGE", "1")
    assert config.fake_storage_enabled() is True


def test_fake_storage_enabled_false_for_other_values(monkeypatch):
    monkeypatch.setattr(config, "_DOTENV_LOADED", True)
    for val in ("true", "yes", "0", ""):
        monkeypatch.setenv("MEMRANK_FAKE_STORAGE", val)
        assert config.fake_storage_enabled() is False
    monkeypatch.delenv("MEMRANK_FAKE_STORAGE", raising=False)
    assert config.fake_storage_enabled() is False


def test_arena_beta_mode_true_only_on_exact_1(monkeypatch):
    monkeypatch.setattr(config, "_DOTENV_LOADED", True)
    monkeypatch.setenv("ARENA_BETA_MODE", "1")
    assert config.arena_beta_mode() is True


def test_arena_beta_mode_default_off(monkeypatch):
    """Beta mode is default-OFF: absent or any non-'1' value reads False."""
    monkeypatch.setattr(config, "_DOTENV_LOADED", True)
    for val in ("true", "yes", "0", ""):
        monkeypatch.setenv("ARENA_BETA_MODE", val)
        assert config.arena_beta_mode() is False
    monkeypatch.delenv("ARENA_BETA_MODE", raising=False)
    assert config.arena_beta_mode() is False
