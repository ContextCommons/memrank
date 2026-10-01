"""Receipt schema-evolution safety: schema_version + tolerant from_dict."""
from __future__ import annotations

from memrank.provenance.receipt import SCHEMA_VERSION, Receipt, build_receipt


def _payload() -> dict:
    receipt = build_receipt(
        adapter_name="a", adapter_version="1", engine_version="x",
        benchmark_name="demo", dataset_version="v1", seed=42, config={"k": 1})
    return receipt.to_dict()


def test_a_fresh_receipt_stamps_the_current_schema():
    """Stamped by ``build_receipt``, not taken from the field default -- see the next test."""
    assert _payload()["schema_version"] == SCHEMA_VERSION


def test_from_dict_ignores_unknown_keys():
    payload = _payload()
    payload["future_field"] = "written by a newer memrank"
    receipt = Receipt.from_dict(payload)  # must not raise
    assert receipt.adapter_name == "a"


def test_from_dict_defaults_missing_optional_keys():
    """A record predating the field is schema 1, whatever the current version has moved to.

    The field default is what an ABSENT value turns out to mean, so it must not track
    ``SCHEMA_VERSION`` -- otherwise every legacy artifact claims the newest layout and a
    comparator can no longer tell them apart.
    """
    payload = _payload()
    del payload["schema_version"]  # a record predating the field
    del payload["config"]
    receipt = Receipt.from_dict(payload)
    assert receipt.schema_version == 1
    assert receipt.config == {}


def test_record_predating_engine_provenance_still_loads():
    payload = _payload()
    del payload["engine_provenance"]  # a record written before the facet existed
    receipt = Receipt.from_dict(payload)
    assert receipt.engine_provenance == {}
