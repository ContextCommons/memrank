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
"""A benchmark's protocol can declare the reader uncapped, and the runner honours it.

The 5000-token cap is memrank's leaderboard fairness control, but BEAM's protocol -- like every
published BEAM harness, including BEAM's own baselines -- hands the reader everything retrieval
returned. A capped run of it is not that benchmark, so `Benchmark.context_policy` exists and
`runner._effective_budget_mode` promotes matched arms accordingly. What must never happen:
the no-memory arm getting promoted (retrieving nothing is its point), or the promotion leaking
into benchmarks that did not ask for it.
"""

from memrank.benchmarks.beam import BEAMBenchmark
from memrank.core import Document
from memrank.runner import _effective_budget_mode, run_cell
from tests.fakes import FakeAdapter, FakeBenchmark


class _UncappedBench(FakeBenchmark):
    context_policy = "uncapped"


def _arm(budget: str) -> FakeAdapter:
    adapter = FakeAdapter(name="eng")
    adapter.context_budget = budget
    return adapter


def test_effective_mode_promotes_matched_and_only_matched():
    uncapped_bench, default_bench = _UncappedBench(), FakeBenchmark()
    assert _effective_budget_mode(_arm("matched"), uncapped_bench) == "uncapped"
    assert _effective_budget_mode(_arm("none"), uncapped_bench) == "none"
    assert _effective_budget_mode(_arm("uncapped"), uncapped_bench) == "uncapped"
    assert _effective_budget_mode(_arm("matched"), default_bench) == "matched"


def test_beam_declares_its_reader_uncapped():
    assert BEAMBenchmark(tier="100k").context_policy == "uncapped"


def test_an_uncapped_protocol_reports_the_tokens_actually_carried(tmp_path):
    """The drill row's `context_tokens` must describe the run that happened: under a tiny cap, a
    matched arm reports the cap and an uncapped-protocol arm reports the full retrieval."""
    long_doc = Document(id="m1", content=" ".join(f"word{i}" for i in range(200)),
                        metadata={"doc_id": "d1"})
    cap = 10

    def run(benchmark):
        adapter = FakeAdapter(name="eng", responses={"q1": [long_doc]})
        cell = run_cell(adapter, benchmark, k=10, repeats=1, run_id_prefix="r",
                        model="gpt-4o-mini", token_budget=cap).to_dict()
        return next(r for r in cell["per_query"] if r["query_id"] == "q1")

    assert run(FakeBenchmark())["context_tokens"] == cap
    assert run(_UncappedBench())["context_tokens"] > cap


def test_longmemeval_declares_its_reader_uncapped():
    """Its protocol truncates only at the MODEL WINDOW (~126k), never at a fairness budget.

    Measured before this landed: at the 5,000-token default the cap bit on 8/8 queries and the
    evidence reached the reader on 0 of 5 scoreable questions, while retrieval itself scored
    recall_all@10 = 1.0 -- the cap was measuring itself.
    """
    from memrank.benchmarks.longmemeval import LongMemEvalBenchmark

    assert LongMemEvalBenchmark().context_policy == "uncapped"


def test_longmemeval_records_its_context_policy_on_the_artifact(tmp_path, monkeypatch):
    """A capped and an uncapped run are different experiments and must not read alike."""
    import json

    from memrank.benchmarks.longmemeval import LongMemEvalBenchmark

    path = tmp_path / "lme.json"
    path.write_text(json.dumps([]), encoding="utf-8")
    monkeypatch.setenv("LONGMEMEVAL_DATA_PATH", str(path))

    assert LongMemEvalBenchmark().config_for_receipt()["context_policy"] == "uncapped"


def test_unranked_arms_report_no_rank_cut_metrics():
    """`icl` and `full-context` return the whole store in ingest order.

    recall_all@k and ndcg_any@k are cut at k, so they are undefined for an arm that does not
    rank -- and scoring anyway produced 0.0 for the two arms that retrieved EVERYTHING. The
    official harness excludes its long-context modes from retrieval scoring by construction.
    """
    from memrank.adapters.controls import FixedContext, FullContext
    from memrank.benchmarks.longmemeval import LongMemEvalBenchmark

    assert FixedContext.ranks_results is False
    assert FullContext.ranks_results is False

    rolled = LongMemEvalBenchmark().rollup(
        [{"retrieval": {"recall_all@5": 0.0, "recall_all@10": 0.0,
                        "ndcg_any@5": 0.0, "ndcg_any@10": 0.0, "n_retrieval_scoreable": 5}}],
        ranked=False)

    assert rolled["recall_all@5"] is None
    assert rolled["retrieval_metrics_apply"] is False


def test_a_ranking_arm_still_reports_them():
    from memrank.adapters.word_overlap import WordOverlap
    from memrank.benchmarks.longmemeval import LongMemEvalBenchmark

    assert WordOverlap.ranks_results is True

    rolled = LongMemEvalBenchmark().rollup(
        [{"retrieval": {"recall_all@5": 1.0, "recall_all@10": 1.0,
                        "ndcg_any@5": 1.0, "ndcg_any@10": 1.0, "n_retrieval_scoreable": 5}}],
        ranked=True)

    assert rolled["recall_all@5"] == 1.0
    assert rolled["retrieval_metrics_apply"] is True
