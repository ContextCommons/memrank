"""`--unit` unit selection: filter loaded units by id / substring / index, fail loud."""

from __future__ import annotations

import pytest
import typer

from memrank.adapters.word_overlap import WordOverlap
from memrank.runner import _filter_units, run_cell
from tests.fakes import MultiUnitFakeBenchmark


def _units():
    return MultiUnitFakeBenchmark(n_units=3).load()  # unit_ids: u0, u1, u2


def test_filter_by_exact_id():
    assert [u.unit_id for u in _filter_units(_units(), ["u1"])] == ["u1"]


def test_filter_by_index_all_digits():
    # all-digits selector is a 0-based index
    assert [u.unit_id for u in _filter_units(_units(), ["2"])] == ["u2"]


def test_filter_multiple_preserves_original_order_and_dedups():
    got = _filter_units(_units(), ["u2", "u0", "u2"])
    assert [u.unit_id for u in got] == ["u0", "u2"]


def test_filter_no_match_fails_loud():
    with pytest.raises(typer.BadParameter, match="nope"):
        _filter_units(_units(), ["nope"])


def test_filter_index_out_of_range_fails_loud():
    with pytest.raises(typer.BadParameter, match="out of range"):
        _filter_units(_units(), ["9"])


def test_run_cell_over_one_filtered_unit():
    units = _filter_units(_units(), ["u1"])
    result = run_cell(WordOverlap(), MultiUnitFakeBenchmark(3), k=10, repeats=1,
                      run_id_prefix="r", model="gpt-4o-mini", token_budget=5000, units=units).to_dict()
    assert result["n_units"] == 1
    assert result["per_unit"][0]["unit_id"] == "u1"
