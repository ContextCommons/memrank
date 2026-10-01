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
"""The cell's provenance receipt -- what ran, configured how, on what evidence class.
"""
from __future__ import annotations

import os
from typing import Any

from memrank.evaluation.aggregate import _benchmark_params
from memrank.evaluation.judge_stage import _judge_receipt_config
from memrank.metrics import cost
from memrank.provenance.engine import build_provenance
from memrank.provenance.receipt import build_receipt


def _build_receipt(adapter, benchmark, *, k, repeats, model, token_budget, seed, judge,
                   workers=1):
    """Build the run receipt, including judge config fields when present."""
    base_config: dict[str, Any] = {
        "k": k, "repeats": repeats, "model": model, "token_budget": token_budget,
        "workers": workers,
        "engine_version": adapter.engine_version,
        "pricing_table_version": cost.PRICING_TABLE_VERSION,
        # Sampling is recorded per role in `_judge_receipt_config`: the answer model is pinned
        # (`answer_temperature`), the judge model is provider-default (claude-opus-4-8 rejects
        # the parameter) -- one `temperature` field would misstate one role or the other.
        "endpoint_class": getattr(adapter, "transport", "http"),
        # Effective components (engine + its extraction LLM + embedder). `model`
        # above stays the eval/pricing model the leaderboard keys on; the engine's
        # own extraction LLM lives here so version-vs-version comparison can track
        # it even before it becomes a tunable parameter.
        "components": adapter.effective_config(),
        **_benchmark_params(benchmark),
        **_judge_receipt_config(judge),
    }
    target = getattr(adapter, "_memrank_target", None)
    if target is not None:
        from memrank.targets.manifest import to_dict

        # EVERY target, not only source-bound ones. This used to require `target.binding is not
        # None`, so a container target -- which is all of mem0, hindsight and supermemory -- recorded
        # no manifest at all, and faithful hindsight at budget=high produced a receipt
        # byte-identical to matched-mode hindsight's. A faithful variant that cannot be told apart from the
        # matched one it is meant to contrast with is not evidence of anything (audit F16).
        #
        # `include_local_binding=False` still drops a developer's checkout path; nothing else in a
        # Manifest is a secret, since `secrets` holds NAMES only (manifest.py:156-168) and values
        # are resolved at launch and never enter a manifest.
        semantic_target = to_dict(target, include_local_binding=False)
        semantic_target.pop("name", None)
        base_config["target"] = semantic_target
    provenance = build_provenance(adapter.name)
    return build_receipt(
        adapter_name=adapter.name, adapter_version=adapter.version,
        engine_version=adapter.engine_version, benchmark_name=benchmark.name,
        dataset_version=benchmark.dataset_version, seed=seed,
        config=base_config, llm_model=model,
        engine_provenance=provenance,
        evidence=_evidence_assessment(provenance),
    )


def _evidence_assessment(provenance: dict[str, Any]) -> dict[str, Any]:
    """Derive claim eligibility from execution facts, never from a user assertion."""
    if os.environ.get("MEMRANK_EVIDENCE_CLASS") == "development_observation":
        return {
            "class": "development_observation",
            "publishable": False,
            "reason": "engine executed from a mutable native source workspace",
        }
    if provenance.get("type") != "application" and not provenance.get("declared"):
        return {
            "class": "endpoint_observation",
            "publishable": False,
            "reason": "engine executable identity was not resolved",
        }
    return {"class": "reproducible_evidence", "publishable": True}
