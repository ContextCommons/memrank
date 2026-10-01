"""The config_hash carries the version of the scheme that produced it."""
from __future__ import annotations

from memrank.provenance.receipt import CONFIG_HASH_VERSION, Receipt, build_receipt


def _receipt(**overrides):
    kwargs = {"adapter_name": "word-overlap", "adapter_version": "0.1.0",
              "engine_version": "in-process", "benchmark_name": "demo",
              "dataset_version": "1", "seed": 42, "config": {"k": 10}}
    kwargs.update(overrides)
    return build_receipt(**kwargs)


def test_receipt_records_the_hash_scheme_version():
    assert _receipt().config_hash_version == CONFIG_HASH_VERSION


def test_version_is_a_positive_int():
    assert isinstance(CONFIG_HASH_VERSION, int)
    assert CONFIG_HASH_VERSION >= 1


def test_versioning_does_not_change_the_hash_it_versions():
    """The marker lives OUTSIDE config, so adding it cannot alter config_hash."""
    assert "config_hash_version" not in _receipt().config


def test_a_run_without_config_still_records_the_version():
    got = _receipt(config=None)
    assert got.config_hash is None
    assert got.config_hash_version == CONFIG_HASH_VERSION


def test_legacy_receipts_read_back_as_version_zero():
    """Artifacts written before this field existed must not be mistaken for current ones."""
    legacy = {"memrank_version": "0.1.0", "adapter_name": "word-overlap", "adapter_version": "0.1.0",
              "engine_version": "in-process", "benchmark_name": "demo", "dataset_version": "1",
              "seed": 42, "started_at": "2026-01-01T00:00:00+00:00", "config_hash": "abc"}
    assert Receipt.from_dict(legacy).config_hash_version == 0
