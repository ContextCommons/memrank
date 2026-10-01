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
"""`$/query` is a hypothetical, and now says so.

It is `context_tokens_mean x input price`, nothing more -- `cost.py` states it: "PROMPT cost only --
excludes ingest-time extraction and answer generation". A real mini LoCoMo run priced myengine at
gpt-4o-mini's $0.15/Mtok while the engine made no LLM call at all (its config declares
`"llm": {"provider": "regex"}`).

That is a defensible estimate of "what this retrieved context would cost you". It is misleading
only when the assumed model is invisible, which it was -- the number sat beside real measured
latency with nothing to mark it as a different kind of thing.
"""
from __future__ import annotations

import pytest

from memrank.metrics import cost
from memrank.runs import registry


@pytest.fixture
def artifact(tmp_path):
    """A cell as `_aggregate_cell` now writes one, minus what these tests do not read."""
    import json

    path = tmp_path / "myengine__locomo.json"
    path.write_text(json.dumps({
        "adapter": "myengine", "benchmark": "locomo", "composite": 0.3186,
        "context_tokens_mean": 4987.012987012987,
        "est_dollars_per_query": 0.00074805,
        "est_dollars_per_cell": 0.00074805 * 385,
        "cost_basis": {"model": "gpt-4o-mini", "kind": "prompt_only",
                       "pricing_table_version": cost.PRICING_TABLE_VERSION,
                       "excludes": ["ingest_extraction", "embeddings",
                                    "answer_generation", "infrastructure"]},
        "ingest_throughput": {"documents": 70, "total_seconds": 0.074,
                              "documents_per_second": 945.9},
        "corpus_documents": 70, "corpus_bytes": 319797, "corpus_tokens": 81234,
        "receipt": {"duration_seconds": 3039.6,
                    "extra": {"rate_limit": {"pauses": 12, "paused_seconds": 94.5}}},
    }), encoding="utf-8")
    return path


def test_the_group_is_labelled_a_hypothetical(artifact):
    """A number labelled only `$/query`, beside measured latency, reads as a measurement."""
    groups = registry.cell_metrics(artifact)

    assert "cost (hypothetical prompt cost)" in groups
    assert "cost" not in groups, "the bare label is what made it read as spend"


def test_the_assumed_model_and_what_it_excludes_are_shown(artifact):
    """Pricing a regex-extraction engine at an LLM's input rate is fine, stated. The mini LoCoMo
    run did exactly that and nothing said so."""
    fields = registry.cell_metrics(artifact)["cost (hypothetical prompt cost)"]

    assert fields["priced as"][0] == "gpt-4o-mini"
    assert "ingest_extraction" in fields["excludes"][0]
    assert "infrastructure" in fields["excludes"][0]


def test_the_whole_cell_total_is_recorded_not_left_to_be_multiplied(artifact):
    fields = registry.cell_metrics(artifact)["cost (hypothetical prompt cost)"]

    value, kind = fields["$/cell (est)"]
    assert value == pytest.approx(0.288, abs=1e-3)
    assert kind == "money", "so it renders at full precision, not rounded to cents"


def test_ingest_throughput_appears_beside_retrieval_latency(artifact):
    """Where a reader comparing engines will actually look. The labels dropped their units --
    `ingest total sec` became `ingest total` -- because the KIND now carries the unit and the
    renderer prints `4m 35s` rather than `275.204`.

    `ingest total` then became `ingest summed latency`, because it never was elapsed time: it sums
    every document's latency, so a `--workers 5` run overstated the clock fivefold and an 11m 30s
    run was read off this line as 50m 49s. The number is unchanged; only the claim it makes is.
    """
    latency = registry.cell_metrics(artifact)["latency"]

    assert latency["ingest rate"] == (pytest.approx(945.9), "rate")
    assert latency["ingest summed latency"] == (pytest.approx(0.074), "seconds")
    assert "ingest total" not in latency, "the label that invited reading a sum as a duration"


def test_corpus_size_is_its_own_group(artifact):
    groups = registry.cell_metrics(artifact)

    assert groups["corpus"] == {"documents": (70, "count"), "bytes": (319797, "bytes"),
                                "tokens": (81234, "count")}


def test_an_artifact_predating_all_of_this_still_renders(tmp_path):
    """236 artifacts on the machine this was written on have none of these fields. They must read
    as before -- absent, not zero."""
    import json

    path = tmp_path / "hindsight__demo.json"
    path.write_text(json.dumps({"adapter": "hindsight", "benchmark": "demo", "composite": 1.0,
                                "est_dollars_per_query": 2.1e-05}), encoding="utf-8")

    groups = registry.cell_metrics(path)

    assert "corpus" not in groups
    assert "ingest docs/sec" not in groups.get("latency", {})
    assert "priced as" not in groups["cost (hypothetical prompt cost)"]


def test_the_run_says_how_long_it_took(artifact):
    """The first thing anyone asks, and the one number this never printed. `finalize_receipt` has
    always stamped `duration_seconds`; nothing rendered it, so the only time-shaped figure on
    screen was a SUM of per-document latencies -- which is how an 11m 30s run came to be reported
    as taking 50m 49s.

    Read from the receipt, never derived from latency: the two differ by the worker count, and
    deriving one from the other is the mistake this line exists to make impossible.
    """
    latency = registry.cell_metrics(artifact)["latency"]

    assert latency["wall clock"] == (pytest.approx(3039.6), "seconds")
    assert latency["wall clock"][0] != latency["ingest summed latency"][0]


def test_time_spent_waiting_on_a_rate_limit_is_visible(artifact):
    """Waiting out a 429 inflates wall clock and touches nothing else -- the percentiles, the
    composite and the cost are all identical to a run that never paused. Two runs that took
    wildly different amounts of time would otherwise look like the same measurement.
    """
    latency = registry.cell_metrics(artifact)["latency"]

    assert latency["rate-limit pauses"] == (12, "count")
    assert latency["paused waiting on quota"] == (pytest.approx(94.5), "seconds")


def test_a_run_that_never_paused_claims_nothing(tmp_path):
    """Absent, not zero -- the rule the rest of this module follows. An artifact written before
    pauses were recorded must not appear to report that it had none."""
    import json

    path = tmp_path / "old__locomo.json"
    path.write_text(json.dumps({"adapter": "mem0", "benchmark": "locomo", "composite": 0.4,
                                "receipt": {"duration_seconds": 10.0}}), encoding="utf-8")

    assert "rate-limit pauses" not in registry.cell_metrics(path)["latency"]


def test_a_run_that_never_finished_shows_no_wall_clock(artifact, tmp_path):
    """A killed run has no `finished_at`, so it has no duration. Omitted rather than zero -- `0s`
    would assert the run took no time, which is the opposite of what happened."""
    import json

    raw = json.loads(artifact.read_text(encoding="utf-8"))
    raw["receipt"] = {}
    path = tmp_path / "killed__locomo.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    assert "wall clock" not in registry.cell_metrics(path)["latency"]
