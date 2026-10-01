"""BEAM scoring-methodology guards.

(1) Substring recall is withheld when it is structurally invalid -- BEAM declares
prose gold answers, and a 0-scoring lexical baseline is the runtime tell.
(2) BEAM is judged by its RUBRIC, and judgeability follows from the rubric rather than
from a gold answer. This used to be pinned against `JUDGE_VALID_CATEGORIES`, which was
the binary grader's category allow-list -- the wrong question to ask about BEAM, since
two of its abilities ship no gold answer for that grader to use.

The five guards that assert how `memrank.analysis.compare` RENDERS these decisions live in
`tests/internal/analysis/test_beam_methodology_render.py`. The methodology is public; the
comparison surface that displays it is not.
"""

from __future__ import annotations

from memrank.benchmarks.beam import BEAMBenchmark
from memrank.benchmarks.locomo import LoCoMoBenchmark
from memrank.judging.shape import BinaryJudgeShape, NuggetJudgeShape


def _row(adapter: str, composite: float, transport: str = "http") -> dict:
    return {"adapter": adapter, "transport": transport, "composite": composite,
            "retrieve_latency": {"p50_ms": 1.0, "iqr_ms": 0.0, "count": 3},
            "ingest_p50_ms": 0.0, "context_tokens_mean": 10.0,
            "est_dollars_per_query": 0.0, "engine_tokens": "n/a"}


def _result(supported: bool, rows: list[dict], benchmark: str = "x") -> dict:
    return {"benchmark": benchmark, "model": "m", "rows": rows,
            "substring_recall_supported": supported}


def _graph_result(rows: list[dict]) -> dict:
    # relation_graph: substring recall N/A, but the composite IS a rankable score.
    return {"benchmark": "relation_graph", "model": "m", "rows": rows,
            "substring_recall_supported": False, "composite_rankable": True,
            "quality_metric": "graph_score"}








def test_no_external_benchmark_claims_the_substring_proxy():
    """All three grade the GENERATED answer in their published protocols, none by string match.

    LoCoMo was the last holdout -- it ranked the proxy as its headline while BEAM withheld the
    same computation as structurally invalid. Enumerated rather than asserted one at a time, so
    a benchmark added later cannot quietly inherit the default and start ranking spans again.
    """
    from memrank.benchmarks.longmemeval import LongMemEvalBenchmark

    for benchmark in (BEAMBenchmark, LoCoMoBenchmark, LongMemEvalBenchmark):
        assert benchmark.substring_recall_supported is False, benchmark.name
        assert benchmark.composite_rankable is False, benchmark.name
        assert benchmark.quality_metric.startswith("judged_"), benchmark.name




def test_relation_graph_renders_with_real_benchmark_metadata():
    # Verify the column label/gate are driven by the benchmark's own attrs.
    from memrank.benchmarks.relation_graph import RelationGraphBenchmark
    assert RelationGraphBenchmark.composite_rankable is True
    assert RelationGraphBenchmark.quality_metric == "graph_score"
    assert RelationGraphBenchmark.substring_recall_supported is False




def test_beam_declares_composite_not_rankable():
    assert BEAMBenchmark.composite_rankable is False


def test_beam_is_judged_by_its_rubric():
    assert isinstance(BEAMBenchmark(tier="100k").judge_shape(), NuggetJudgeShape)


def test_beam_scores_all_ten_abilities():
    """Nine by rubric average, event_ordering by rank correlation. Asserted on BEAM's own shape
    rather than on the generic one, because the generic grader below still excludes ordering."""
    shape = BEAMBenchmark(tier="100k").judge_shape()
    for category in ("abstention", "contradiction_resolution", "event_ordering",
                     "information_extraction", "instruction_following", "knowledge_update",
                     "multi_session_reasoning", "preference_following", "summarization",
                     "temporal_reasoning"):
        assert shape.is_judgeable({"category": category, "rubric": ["x"]}), category


def test_the_generic_rubric_grader_still_declines_ordering():
    """Summarization is IN -- BEAM scores it with the same grader as information extraction, and
    the "partial-credit rubric" it was once said to need IS the rubric. Ordering is genuinely
    different, so the plain nugget grader declines it and `BeamJudgeShape` supplies the path."""
    shape = NuggetJudgeShape()
    for category in ("abstention", "contradiction_resolution", "information_extraction",
                     "instruction_following", "knowledge_update", "multi_session_reasoning",
                     "preference_following", "summarization", "temporal_reasoning"):
        assert shape.is_judgeable({"category": category, "rubric": ["x"]}), category
    assert not shape.is_judgeable({"category": "event_ordering", "rubric": ["x"]})


def test_a_rubric_makes_a_query_judgeable_where_a_gold_answer_would_not():
    """The two compliance abilities ship no gold answer at all, so the binary grader could
    never have judged them; the rubric can."""
    compliance = {"category": "instruction_following", "rubric": ["should use code blocks"],
                  "gold_answers": [""]}
    binary = BinaryJudgeShape(frozenset({"instruction_following"}))
    assert NuggetJudgeShape().is_judgeable(compliance)
    assert not binary.is_judgeable(compliance)
    assert binary.unjudged_reason(compliance) == "no_gold"
    assert NuggetJudgeShape().unjudged_reason({"category": "abstention", "rubric": []}) \
        == "no_rubric"
