"""Tests for the local run registry (memrank/registry.py)."""
from __future__ import annotations

import json

from memrank.runs import registry


def _write_cell(run_dir, adapter, benchmark, *, composite, config_hash, started_at):
    payload = {"adapter": adapter, "benchmark": benchmark, "composite": composite,
               "receipt": {"adapter_name": adapter, "benchmark_name": benchmark,
                           "config_hash": config_hash, "started_at": started_at}}
    (run_dir / f"{adapter}__{benchmark}.json").write_text(json.dumps(payload))


def test_new_run_dir_is_unique_and_under_root(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMRANK_RUNS_DIR", str(tmp_path))
    a = registry.new_run_dir("demo", "smoke", None)
    b = registry.new_run_dir("demo", "smoke", None)
    assert a != b
    assert a.parent == tmp_path and "demo" in a.name and "smoke" in a.name


def test_list_runs_reads_back_and_orders(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMRANK_RUNS_DIR", str(tmp_path))
    d1 = registry.new_run_dir("demo")
    _write_cell(d1, "word-overlap", "demo", composite=0.5, config_hash="h1", started_at="2026-01-01T00:00:00")
    d2 = registry.new_run_dir("demo")
    _write_cell(d2, "word-overlap", "demo", composite=0.6, config_hash="h2", started_at="2026-01-02T00:00:00")
    infos = registry.list_runs(adapter="word-overlap", benchmark="demo")
    assert [i.composite for i in infos] == [0.5, 0.6]  # oldest first
    assert infos[0].config_hash == "h1"


def test_resolve_path_and_run_id(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMRANK_RUNS_DIR", str(tmp_path))
    d = registry.new_run_dir("demo")
    _write_cell(d, "word-overlap", "demo", composite=0.5, config_hash="h", started_at="t")
    cell = d / "word-overlap__demo.json"
    assert registry.resolve(str(cell)) == cell            # path passes through
    assert registry.resolve(d.name) == cell               # run-id resolves (single cell)


def test_latest_returns_two_most_recent(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMRANK_RUNS_DIR", str(tmp_path))
    for i, h in enumerate(["h1", "h2", "h3"]):
        d = registry.new_run_dir("demo")
        _write_cell(d, "word-overlap", "demo", composite=0.5, config_hash=h,
                    started_at=f"2026-01-0{i + 1}T00:00:00")
    paths = registry.latest("word-overlap", "demo", 2)
    hashes = [json.loads(p.read_text())["receipt"]["config_hash"] for p in paths]
    assert hashes == ["h2", "h3"]  # oldest-of-the-two first, newest last


def test_cell_with_engine_provenance_still_parses(monkeypatch, tmp_path):
    # RunInfo reads only config_hash/timestamp from the receipt; the new facet must not break it.
    monkeypatch.setenv("MEMRANK_RUNS_DIR", str(tmp_path))
    d = registry.new_run_dir("demo")
    payload = {"adapter": "mem0", "benchmark": "demo", "composite": 0.5,
               "receipt": {"adapter_name": "mem0", "benchmark_name": "demo", "config_hash": "h",
                           "started_at": "2026-01-01T00:00:00",
                           "engine_provenance": {"purl": "pkg:oci/mem0@sha256:aaaa"}}}
    (d / "mem0__demo.json").write_text(json.dumps(payload))
    infos = registry.list_runs(adapter="mem0", benchmark="demo")
    assert infos[0].config_hash == "h"


# --- a run folder holds more than cells ------------------------------------------------------- #
# cell_files excluded only "summary__", so it swallowed status.json the moment the run-status
# heartbeat started writing one into every run directory. Every run listed twice -- once as a
# phantom "?xdemo composite=-" -- and `compare-versions <run-id>` raised "holds multiple cells" for
# every single-cell run, making run-id comparison impossible.

def _run_with(tmp_path, **files):
    run_dir = tmp_path / "20260731-000000__demo__abc123"
    run_dir.mkdir(parents=True)
    for name, body in files.items():
        (run_dir / name).write_text(body, encoding="utf-8")
    return run_dir


def test_status_json_is_not_a_cell(tmp_path):
    from memrank.runs.registry import cell_files

    run_dir = _run_with(tmp_path, **{"word-overlap__demo.json": "{}", "status.json": "{}",
                                     "summary__demo.json": "{}"})
    assert [p.name for p in cell_files(run_dir)] == ["word-overlap__demo.json"]


def test_future_metadata_files_are_not_cells(tmp_path):
    """Matched positively on <target>__<benchmark>.json, so the next metadata file added to a run
    folder does not silently become a phantom cell."""
    from memrank.runs.registry import cell_files

    run_dir = _run_with(tmp_path, **{"mem0-bge-tei__demo.json": "{}", "status.json": "{}",
                                     "manifest.json": "{}", "provenance.json": "{}"})
    assert [p.name for p in cell_files(run_dir)] == ["mem0-bge-tei__demo.json"]


def test_a_single_cell_run_resolves_by_run_id(tmp_path, monkeypatch):
    """The break: compare-versions could not resolve ANY run-id."""
    from memrank.runs import registry

    monkeypatch.setenv("MEMRANK_RUNS_DIR", str(tmp_path))
    run_dir = _run_with(tmp_path, **{"word-overlap__demo.json": "{}", "status.json": "{}"})
    assert registry.resolve(run_dir.name).name == "word-overlap__demo.json"


def test_a_genuine_multi_cell_run_still_needs_disambiguation(tmp_path, monkeypatch):
    """The error message is right when there really ARE several cells."""
    import pytest

    from memrank.runs import registry

    monkeypatch.setenv("MEMRANK_RUNS_DIR", str(tmp_path))
    run_dir = _run_with(tmp_path, **{"word-overlap__demo.json": "{}", "fixed-context__demo.json": "{}",
                                     "status.json": "{}"})
    with pytest.raises(ValueError, match="multiple cells"):
        registry.resolve(run_dir.name)
