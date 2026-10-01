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
"""Every name runner re-exports from `memrank.evaluation` IS the moved object.

The eval loop left `runner.py` but its names did not: tests and in-repo consumers still
import (and monkeypatch) them through `memrank.runner`. A re-export that drifted into a
copy would let a patch on one module miss the code running in the other -- a silently
green test with a real subprocess behind it. Identity, not equality, is the property.
"""
from __future__ import annotations

import importlib

import pytest

from memrank import runner
from memrank.evaluation import aggregate, cell, judge_stage, observer, receipt
from memrank.orchestration import cloud as orchestration_cloud
from memrank.orchestration import observers as orchestration_observers
from memrank.orchestration import placement_gate as orchestration_placement
from memrank.orchestration import resolve as orchestration_resolve
from memrank.orchestration import sweep as orchestration_sweep

MOVED = {
    aggregate: ["_aggregate_cell", "_benchmark_params", "_corpus_size", "_drill_unit",
                "_ingest_throughput", "_mean_composite"],
    cell: ["_INGEST_BACKOFF_BASE_S", "_INGEST_RETRY_DEADLINE_S", "AdapterFactory",
           "EmptyRetrieval", "RateLimitExhausted", "_assert_retrieved_something",
           "_close_adapter", "_effective_budget_mode", "_ingest_one",
           "_ingest_with_progress", "_is_rate_limited", "_not_applicable_cell",
           "_pause_for_rate_limit", "_retrieve_one", "_retrieve_with_progress",
           "_run_one_unit", "_run_unit_repeats", "_run_units_concurrent",
           "_run_units_sequential", "cell_applicable", "run_cell"],
    judge_stage: ["EmptyJudgeCoverage", "_apply_judge", "_context_text", "_grade_all",
                  "_judge_cfg_from_receipt", "_judge_metrics", "_judge_one_query",
                  "_judge_receipt_config", "_truncation_metrics",
                  "assert_judge_coverage", "judge_cost_estimate"],
    observer: ["NULL_OBSERVER", "EvalObserver", "EvalPlan", "_eval_plan",
               "_progress_step"],
    receipt: ["_build_receipt", "_evidence_assessment"],
    orchestration_observers: ["_planned_progress"],
    orchestration_resolve: ["_JUDGE_ONLY_TARGETS", "_cli_experiments",
                            "_ensure_run_credentials", "_experiment_metadata",
                            "_judge_cfg_or_refuse", "_load_org_credentials",
                            "_manifest_factories", "_manifest_factory", "_refuse_abstract",
                            "_run_targets", "_split_overrides", "_validate_positive",
                            "_validate_reader", "_warn_if_meaningless_unjudged",
                            "run_credentials"],
    orchestration_placement: ["PLACEMENTS", "_environment_patch",
                              "_require_placement_ready", "_source_digest_guard",
                              "_validate_source_target", "placement_for"],
    orchestration_cloud: ["_api_client", "_record_submission", "_report_refusals",
                          "_submit_sweep"],
    orchestration_sweep: ["SweepKilled", "_ExecOpts", "_LocalRun", "_child_argv",
                          "_composite_display", "_composite_rankable",
                          "_discard_checkpoint", "_echo_cell_outcome",
                          "_echo_per_query_outcomes", "_execute_local_run",
                          "_filter_units", "_install_kill_handler", "_mark_unstarted",
                          "_mint_local_runs", "_mirror_to_mlflow", "_preflight_engines",
                          "_preload_tokenizer_with_notice", "_record_local_run",
                          "_run_and_persist", "_run_local_sweep", "_safe_label",
                          "_spawn_child", "_submit_local_sweep", "_summary_cell",
                          "_sweep_gates", "_write_summary", "question_gates",
                          "run_identity"],
}


@pytest.mark.parametrize(
    "home,name",
    [(home, name) for home, names in MOVED.items() for name in names],
    ids=lambda v: v if isinstance(v, str) else v.__name__.rsplit(".", 1)[-1])
def test_the_runner_name_is_the_moved_object(home, name):
    assert getattr(runner, name) is getattr(home, name), (
        f"runner.{name} is not {home.__name__}.{name} -- a drifted duplicate exists, and "
        f"whichever module a test patches, the other one keeps running the real thing")


def test_the_public_package_face_matches_cell():
    # `import memrank.evaluation as evaluation` no longer binds the package: `evaluation` is
    # one of the seven's verbs at the top level, and the package deliberately shadows the
    # submodule attribute so `memrank.evaluation("demo")` cannot silently be the module. The
    # package itself is reached by its full name, which is how every caller in the tree
    # already reaches it (`from memrank.evaluation.api import run`).
    evaluation = importlib.import_module("memrank.evaluation")

    assert evaluation.run_cell is cell.run_cell
    assert evaluation.cell_applicable is cell.cell_applicable
