"""Workspace observations may sync privately but never enter a public leaderboard.

The receipt is where that starts: it declares the evidence class, and the default must be the
publishable one. What the LEADERBOARD then does with a non-publishable receipt -- refuse to
normalize it, and say why -- is asserted in `tests/internal/leaderboard/test_development_evidence.py`,
because `memrank.leaderboard` is not in a public tree.
"""
from memrank.provenance.receipt import build_receipt


def test_receipt_defaults_to_publishable_artifact_evidence():
    receipt = build_receipt(adapter_name="x", adapter_version="1", engine_version="1",
                            benchmark_name="demo", dataset_version="1", seed=1)
    assert receipt.evidence == {"class": "reproducible_evidence", "publishable": True}
