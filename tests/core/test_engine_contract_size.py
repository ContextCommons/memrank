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
"""The engine contract asks for nothing that reports a measurement memrank takes (ATO-2136).

`latency_metrics()` and `token_metrics()` were abstract, so every person wrote them, and the run
loop never checked their output -- the audit observed an engine whose both methods returned `{}`
running to completion and producing a composite. They were a contract already satisfied by writing
nothing. These tests pin what replaced them: latency is memrank's own measurement, taken at its own
call boundary, and usage is the engine's optional DECLARATION, recorded as declared.
"""

from __future__ import annotations

from typing import Any

import pytest

# The cell run that produces an `EvalResult` is imported from its own module, which is
# where the CLI and the cloud reach it too; `memrank.run` is the typed run over the seven
# (memrank/instrument/) and is not what these tests exercise.
from memrank import BenchmarkUnit, ComposedEvaluation, Document, MemoryAdapter, Recall
from memrank.core import REQUIRED_LATENCY_KEYS, REQUIRED_TOKEN_KEYS
from memrank.evaluation.api import run as run_cell
from memrank.instrumentation import TokenCollector

# These tests exercise the deprecated in-process route on purpose; its notice is expected here.
pytestmark = pytest.mark.filterwarnings(
    "ignore:.*is deprecated and will be removed:DeprecationWarning")

UNITS = [BenchmarkUnit(
    unit_id="u1", isolation_id="u1",
    documents=[Document(id="d1", user_id="u1",
                        content="Acme upgraded to the enterprise plan in March.")],
    queries=[{"id": "q1", "text": "What plan is Acme on?", "required_spans": ["enterprise"]}])]


class WholeContract(MemoryAdapter):
    """Everything the contract asks for, and nothing else. Four methods."""

    name, version, engine_version = "whole-contract", "0.1", "0.1"

    def __init__(self) -> None:
        self.docs: list[Document] = []

    def prepare(self, isolation_unit: str) -> None:
        self.docs = []

    def ingest(self, documents: list[Document]) -> None:
        self.docs.extend(documents)

    def retrieve(self, query: str, k: int, user_id: str,
                 query_timestamp: Any = None) -> Recall:
        return Recall(documents=self.docs[:k], declared={"engine": self.name})

    def cleanup(self) -> None:
        self.docs = []


class DeclaresItsUsage(WholeContract):
    """The one measurement only an engine can make: what a provider billed it."""

    name = "declares-usage"

    def __init__(self) -> None:
        super().__init__()
        self.tokens = TokenCollector()

    def ingest(self, documents: list[Document]) -> None:
        super().ingest(documents)
        self.tokens.record("ingest", 120)

    def retrieve(self, query: str, k: int, user_id: str,
                 query_timestamp: Any = None) -> Recall:
        self.tokens.record("query", 40)
        return super().retrieve(query, k, user_id, query_timestamp)

    def token_metrics(self) -> dict[str, float | None]:
        return self.tokens.as_metrics()


def test_the_contract_is_four_methods() -> None:
    assert MemoryAdapter.__abstractmethods__ == frozenset(
        {"prepare", "ingest", "retrieve", "cleanup"})
    assert not hasattr(MemoryAdapter, "latency_metrics")


def test_an_engine_written_from_the_contract_alone_runs_and_is_measured() -> None:
    """The stubs are gone AND the result still carries latency -- memrank's own."""
    result = run_cell(WholeContract(), ComposedEvaluation(name="mine", questions=UNITS),
                         repeats=1)

    assert result.composite == 1.0
    assert REQUIRED_LATENCY_KEYS == set(result.latency_metrics)
    assert all(value >= 0.0 for value in result.latency_metrics.values())


def test_an_engine_that_declares_nothing_reports_absence_not_zero() -> None:
    result = run_cell(WholeContract(), ComposedEvaluation(name="mine", questions=UNITS),
                         repeats=1)

    assert set(result.token_metrics) == REQUIRED_TOKEN_KEYS
    # Absent is the absence of a measurement, and must never be spelled 0.0.
    assert all(value is None for value in result.token_metrics.values())


def test_an_engine_that_declares_its_usage_has_it_recorded_as_declared() -> None:
    result = run_cell(DeclaresItsUsage(), ComposedEvaluation(name="mine", questions=UNITS),
                         repeats=1)

    assert result.token_metrics["tokens_per_ingest_mean"] == 120.0
    assert result.token_metrics["tokens_per_query_mean"] == 40.0


def test_the_contract_alone_is_enough_at_more_than_one_worker() -> None:
    """The concurrent path built one adapter per unit and read `.latency`/`.tokens` off it.

    Neither attribute is in the contract, so an engine written from it crashed there. Driven
    through `run_cell` because a factory is what the concurrent path needs, and `memrank.run`
    builds one only for a catalog ref.
    """
    from memrank.evaluation.cell import run_cell

    result = run_cell(WholeContract(), ComposedEvaluation(name="mine", questions=UNITS),
                      k=10, repeats=1, run_id_prefix="w", model="gpt-4o-mini",
                      token_budget=5000, workers=2, make_adapter=WholeContract)

    assert result.composite == 1.0
    assert REQUIRED_LATENCY_KEYS == set(result.latency_metrics)
    assert all(value is None for value in result.token_metrics.values())


def test_a_declaration_that_cannot_be_merged_across_workers_is_refused_not_dropped() -> None:
    """An engine declaring usage with no collector: refused, never reported as unmeasured."""
    import pytest

    from memrank.evaluation.cell import run_cell

    class Unmergeable(WholeContract):
        name = "unmergeable"

        def token_metrics(self) -> dict[str, float | None]:
            return dict.fromkeys(REQUIRED_TOKEN_KEYS, 1.0)

    with pytest.raises(TypeError, match="cannot be pooled across"):
        run_cell(Unmergeable(), ComposedEvaluation(name="mine", questions=UNITS),
                 k=10, repeats=1, run_id_prefix="w", model="gpt-4o-mini",
                 token_budget=5000, workers=2, make_adapter=Unmergeable)


# ---- what an engine CAN still declare about time -----------------------------------------


class DeclaresEngineTime(WholeContract):
    """A translator: memrank's hop is memrank's to measure, the engine's own spend is not."""

    name = "declares-engine-time"

    def retrieve(self, query: str, k: int, user_id: str,
                 query_timestamp: Any = None) -> Recall:
        self.engine_ms = [*getattr(self, "engine_ms", []), 4.0]
        return super().retrieve(query, k, user_id, query_timestamp)

    def declared_latency(self) -> dict[str, list[float]]:
        return {"retrieve_engine": list(getattr(self, "engine_ms", []))}


def test_engine_side_time_is_declared_beside_the_measurement_not_instead_of_it() -> None:
    result = run_cell(DeclaresEngineTime(), ComposedEvaluation(name="mine", questions=UNITS),
                         repeats=1)

    assert result.latency_metrics["retrieve_engine_p50_ms"] == 4.0
    assert result.latency_metrics["retrieve_engine_p95_ms"] == 4.0
    assert REQUIRED_LATENCY_KEYS <= set(result.latency_metrics)


def test_an_engine_declaring_a_key_memrank_measures_is_refused() -> None:
    import pytest

    class Overreaching(WholeContract):
        name = "overreaching"

        def declared_latency(self) -> dict[str, list[float]]:
            return {"retrieve": [0.0]}          # would render retrieve_p50_ms

    with pytest.raises(ValueError, match="measures itself"):
        run_cell(Overreaching(), ComposedEvaluation(name="mine", questions=UNITS), repeats=1)
