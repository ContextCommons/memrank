"""Tests for the checks `memrank targets verify` runs against a translator.

A conformance suite that passes everything is worse than none: it tells a third party their
integration is sound while it silently inflates every score. So each test here breaks the contract
in one specific way and asserts the checker catches THAT, rather than only asserting the happy path.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from memrank.adapters.conformance import contract_checks
from memrank.core import Document, MemoryAdapter, Recall


class FakeTranslator(MemoryAdapter):
    """A conforming translator, with one switch per way to violate the contract."""

    name = "fake"
    version = "0.1.0"
    engine_version = "1.0.0"
    transport = "translator"

    def __init__(self, *, isolates: bool = True, ranks: bool = True,
                 describes: bool = True) -> None:
        self.isolates = isolates
        self.ranks = ranks
        self.describes = describes
        self.store: dict[str, list[Document]] = {}
        self.unit: str | None = None

    def describe_engine(self) -> dict[str, Any] | None:
        return {"llm": None, "embedder": None} if self.describes else None

    def prepare(self, isolation_unit: str) -> None:
        self.unit = "shared" if not self.isolates else isolation_unit
        self.store.setdefault(self.unit, [])

    def ingest(self, documents: list[Document]) -> None:
        self.store[self.unit or "shared"].extend(documents)

    def retrieve(self, query: str, k: int, user_id: str,
                 query_timestamp: datetime | str | None = None) -> Recall:
        found = list(self.store.get(self.unit or "shared", []))
        if not self.ranks:
            found.reverse()
        return Recall(documents=found[:k])

    def cleanup(self) -> None:
        if self.isolates and self.unit is not None:
            self.store.pop(self.unit, None)
        self.unit = None

    def latency_metrics(self) -> dict[str, float]:
        from memrank.core import REQUIRED_LATENCY_KEYS

        return dict.fromkeys(REQUIRED_LATENCY_KEYS, 0.0)

    def token_metrics(self) -> dict[str, float | None]:
        from memrank.core import REQUIRED_TOKEN_KEYS

        return dict.fromkeys(REQUIRED_TOKEN_KEYS, None)


def failures(adapter: MemoryAdapter) -> list[str]:
    return [row.name for row in contract_checks(adapter) if not row.ok]


def test_a_conforming_translator_passes_every_check():
    rows = contract_checks(FakeTranslator())
    assert [row.name for row in rows] == [
        "describe", "ingest", "retrieve", "isolation", "metrics"]
    assert failures(FakeTranslator()) == []


def test_leaked_state_between_units_is_caught():
    """The failure that produces an inflated number rather than an error."""
    assert failures(FakeTranslator(isolates=False)) == ["isolation"]


def test_unranked_results_are_caught():
    """recall@k reads the order, so 'returned it somewhere' is not the same as 'ranked it'."""
    assert failures(FakeTranslator(ranks=False)) == ["retrieve"]


def test_a_silent_describe_short_circuits_the_rest():
    """Without a handshake memrank does not know what it is talking to; later checks would only
    report consequences of this one."""
    rows = contract_checks(FakeTranslator(describes=False))
    assert [row.name for row in rows] == ["describe"]
    assert rows[0].ok is False


def test_unmeasured_tokens_are_reported_as_such_not_as_a_failure():
    """Most engines report no usage at all; that is honest, not a violation."""
    metrics = [row for row in contract_checks(FakeTranslator()) if row.name == "metrics"][0]
    assert metrics.ok
    assert "unmeasured" in metrics.detail
