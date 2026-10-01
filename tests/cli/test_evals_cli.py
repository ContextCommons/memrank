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
"""``memrank evals ls`` / ``show`` -- the goal catalog and its detail view.

``show`` renders DECLARED metadata only: the download-guard test poisons the dataset cache
root and expects every eval to still describe itself, pinning the catalog/download boundary.
"""
from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from memrank.benchmarks import REGISTRY, list_benchmarks
from memrank.benchmarks.beam import _HF_SPLIT_MAP, BEAMBenchmark
from memrank.core import EvalInfo
from memrank.definitions.shipped import shipped_refs
from memrank.runner import app

runner = CliRunner()


def test_evals_ls_lists_every_runnable_variant():
    """The catalog is every EVALUATION `memrank run` takes, not every benchmark family.

    `beam:100k-smoke` is 20 questions over one conversation and `beam:1m` is 700 over
    thirty-five; a line reading "beam" named neither, and their scores were never comparable.
    """
    result = runner.invoke(app, ["evals", "ls"])
    assert result.exit_code == 0
    listed = {line.strip() for line in result.stdout.splitlines() if line.strip()}
    assert listed == set(shipped_refs())
    assert {"locomo", "locomo:smoke", "beam:100k", "beam:1m-mini"}.issubset(listed)
    assert not {"demo", "relation_graph"} & listed, "evaluations an agent cannot be put to"
    assert "beam" not in listed, "a tiered benchmark has no unqualified evaluation"


def test_evals_show_renders_declared_catalog_fields():
    result = runner.invoke(app, ["evals", "show", "locomo"])
    assert result.exit_code == 0
    for expected in ("locomo", "snap-research/locomo10@v1", "smoke", "mini"):
        assert expected in result.stdout


def test_evals_show_beam_declares_tiers_and_judge_requirement():
    result = runner.invoke(app, ["evals", "show", "beam"])
    assert result.exit_code == 0
    for tier in ("100k", "500k", "1m"):
        assert tier in result.stdout
    # Labels are dimmed rather than colon-suffixed now -- the claim is that the field is there.
    assert "judge" in result.stdout and "required" in result.stdout


def test_evals_show_json_is_parseable_and_complete():
    result = runner.invoke(app, ["evals", "show", "beam", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert {"name", "dataset_version", "unit", "slices", "tiers", "judge",
            "is_synthetic", "quality_label"}.issubset(payload)
    assert "\x1b" not in result.stdout


def test_evals_show_unknown_name_fails_loud():
    result = runner.invoke(app, ["evals", "show", "nope"])
    assert result.exit_code == 1
    assert "error:" in result.output
    for name in list_benchmarks():
        assert name in result.output


def test_evals_show_never_touches_dataset_cache(monkeypatch):
    """The detail view is a catalog read, not a download trigger."""
    import memrank.benchmarks as benchmarks

    def poisoned_cache_root():
        raise AssertionError("evals show reached the dataset cache")

    monkeypatch.setattr(benchmarks, "cache_root", poisoned_cache_root)
    for name in list_benchmarks():
        result = runner.invoke(app, ["evals", "show", name])
        assert result.exit_code == 0, f"{name}: {result.output}"


def test_every_registered_eval_declares_info():
    for name, cls in REGISTRY.items():
        assert isinstance(cls.info, EvalInfo), f"{name} lacks an EvalInfo"
    # Tiers exist only where the loader has splits to honor them -- pinned to the map itself.
    assert set(BEAMBenchmark.info.tiers) == set(_HF_SPLIT_MAP)
    tiered = [n for n, c in REGISTRY.items() if c.info.tiers]
    assert tiered == ["beam"]


@pytest.mark.parametrize("name", ["demo", "relation_graph"])
def test_synthetic_evals_say_judging_sends_nothing_real(name):
    """`ack_egress_required` is gone with the flag it named; `is_synthetic` is the fact that
    remains, and it decides what the egress line discloses."""
    result = runner.invoke(app, ["evals", "show", name, "--json"])
    payload = json.loads(result.stdout)
    assert payload["is_synthetic"] is True
    assert "ack_egress_required" not in payload
