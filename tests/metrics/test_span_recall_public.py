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
"""memrank's deterministic scorer has a public name, and it is the demo benchmark's (ATO-2135).

The audit observed `ImportError: cannot import name 'score_query' from 'memrank.metrics'`: the
only scorer memrank ships lived in one benchmark's import namespace, so "my questions, memrank's
scorer" could not be written. These tests pin the three things that fixed it -- the name is
importable, the demo benchmark uses it, and a person who brings only questions gets a scored run.
"""

from __future__ import annotations

import pytest

# `memrank.run` is now the typed run over the seven (memrank/instrument/); the cell run
# that produces an `EvalResult` is imported from its own module, which is where the CLI
# and the cloud reach it too.
import memrank
from memrank import Benchmark, BenchmarkUnit, Document, SpanRecall
from memrank.benchmarks.demo import DemoBenchmark
from memrank.evaluation.api import run as run_cell

# These tests exercise the deprecated in-process route on purpose; its notice is expected here.
pytestmark = pytest.mark.filterwarnings(
    "ignore:.*is deprecated and will be removed:DeprecationWarning")


def test_the_scorer_imports_from_memrank_and_from_memrank_metrics() -> None:
    from memrank.metrics import SpanRecall as from_metrics
    from memrank.metrics import score_query, spec_from_query

    assert memrank.SpanRecall is from_metrics
    # Importable from `memrank`, deliberately not in `__all__`: the advertised surface is the
    # four nouns, and `tests/repo/test_python_vocabulary.py` pins that list.
    assert "SpanRecall" in dir(memrank) and "SpanRecall" not in memrank.__all__
    assert callable(score_query) and callable(spec_from_query)


def test_the_name_does_not_claim_answer_correctness() -> None:
    """The honest-labelling rule: this is a substring proxy and the label says so."""
    assert SpanRecall.quality_metric == "substring_recall"
    assert "not answer correctness" in SpanRecall.METRIC_LABEL


def test_the_demo_benchmark_marks_through_the_public_name() -> None:
    assert isinstance(DemoBenchmark.scorer, SpanRecall)


class QuestionsOnly(Benchmark):
    """Questions brought by a person; the scorer is memrank's, pointed at rather than written."""

    name = "questions-only"
    dataset_version = "test@1"
    scorer = SpanRecall()

    def load(self) -> list[BenchmarkUnit]:
        return [BenchmarkUnit(
            unit_id="u1", isolation_id="u1",
            documents=[Document(id="d1", user_id="u1",
                                content="Acme upgraded to the enterprise plan in March."),
                       Document(id="d2", user_id="u1",
                                content="The outage was traced to an expired webhook secret.")],
            queries=[{"id": "q1", "text": "What plan is Acme on?",
                      "required_spans": ["enterprise"], "category": "single-hop"},
                     {"id": "q2", "text": "What caused the outage?",
                      "required_spans": ["webhook secret"], "category": "single-hop"}])]

    def score(self, unit, responses):  # type: ignore[no-untyped-def]
        return self.scorer.score(unit, responses)

    def report_template(self) -> str:
        return "# questions-only {adapter} {composite}"


def test_questions_only_gets_a_scored_run() -> None:
    result = run_cell("word-overlap", QuestionsOnly(), repeats=1)

    assert result.composite == 1.0
    assert result.per_unit[0]["metric"] == SpanRecall.METRIC_LABEL
    assert result.per_unit[0]["per_category"] == {"single-hop": 1.0}


def test_the_demo_scorer_is_unchanged_by_the_extraction() -> None:
    """The arithmetic moved; the scored dict a demo unit produces did not."""
    benchmark = DemoBenchmark()
    unit = benchmark.load()[0]
    scored = benchmark.score(unit, [])

    # Nothing retrieved: every positive query misses and the one negative query is satisfied,
    # which is the 1/5 the demo scenario has always produced from an empty response list.
    assert scored["composite"] == 0.2
    assert scored["n_queries"] == len(unit.queries)
    assert scored["metric"] == "evidence_recall (retrieval proxy; not answer correctness)"
    assert set(scored) == {"composite", "per_category", "n_queries", "metric"}
    assert scored["per_category"]["abstention"] == 1.0
