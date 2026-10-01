"""Deterministic test doubles -- no live backend, no real latency."""

from __future__ import annotations

from datetime import datetime

from memrank.core import (
    Benchmark,
    BenchmarkUnit,
    Document,
    MemoryAdapter,
    Recall,
)


class FakeAdapter(MemoryAdapter):
    """Adapter returning configured docs per query id; counts lifecycle calls."""

    def __init__(self, name: str, responses: dict[str, list[Document]] | None = None,
                 destructive: bool = True) -> None:
        self.name = name
        self.version = "test"
        self.engine_version = "test"
        self._configured_responses = responses or {}
        self.responses: dict[str, list[Document]] = {}
        self.cleanup_is_destructive = destructive
        self.ingest_calls = 0
        self.prepare_calls: list[str] = []
        self._lat = {"ingest": 1.0, "retrieve": 2.0}

    def prepare(self, isolation_unit: str) -> None:
        self.prepare_calls.append(isolation_unit)
        self.responses = dict(self._configured_responses)

    def ingest(self, documents: list[Document]) -> None:
        self.ingest_calls += 1

    def retrieve(self, query: str, k: int, user_id: str,
                 query_timestamp: datetime | str | None = None) -> Recall:
        return Recall(documents=list(self.responses.get(query, [])),
                      declared={"query": query})

    def cleanup(self) -> None:
        self.responses = {}

    def latency_metrics(self) -> dict[str, float]:
        return {
            "ingest_p50_ms": 1.0, "ingest_p95_ms": 1.0, "ingest_p99_ms": 1.0,
            "retrieve_p50_ms": 2.0, "retrieve_p95_ms": 2.0, "retrieve_p99_ms": 2.0,
        }

    def token_metrics(self) -> dict[str, float]:
        return {
            "tokens_per_query_mean": 0.0, "tokens_per_query_p95": 0.0,
            "tokens_per_ingest_mean": 0.0, "tokens_per_ingest_p95": 0.0,
        }


class FakeBenchmark(Benchmark):
    """One unit, two queries (one positive with content match, one that misses)."""

    name = "fake"
    dataset_version = "fake@v1"

    def load(self) -> list[BenchmarkUnit]:
        docs = [Document(id="d1", content="Her favorite animal is the blue whale.",
                         user_id="u1", metadata={"doc_id": "d1"})]
        queries = [
            {"id": "q1", "text": "q1", "user_id": "u1",
             "required_spans": ["blue whale"], "evidence_doc_ids": ["d1"]},
            {"id": "q2", "text": "q2", "user_id": "u1",
             "required_spans": ["red panda"]},
        ]
        return [BenchmarkUnit(unit_id="u1", isolation_id="u1",
                              documents=docs, queries=queries)]

    def score(self, unit, responses):
        from memrank.metrics.scoring import score_query, spec_from_query
        by_id = {r.query_id: r for r in responses}
        hits = []
        for q in unit.queries:
            r = by_id.get(q["id"])
            res = score_query(spec_from_query(q), r.documents if r else [])
            hits.append(1 if res.hit else 0)
        return {"composite": sum(hits) / len(hits), "per_category": {},
                "n_queries": len(hits)}

    def report_template(self) -> str:
        return "# fake {adapter} {composite}"


class MultiUnitFakeBenchmark(Benchmark):
    """N independent units, each with its own doc + a query whose gold span appears
    ONLY in that unit's doc -- so a correct, isolated run scores 1.0, and any cross-unit
    bleed would change the result. Deterministic; works with the in-process baseline."""

    name = "multifake"
    dataset_version = "multifake@v1"

    def __init__(self, n_units: int = 3) -> None:
        self.n_units = n_units

    def load(self) -> list[BenchmarkUnit]:
        units = []
        for i in range(self.n_units):
            token = f"alpha{i}"
            docs = [Document(id=f"d{i}", content=f"the secret token is {token}",
                             user_id=f"u{i}", metadata={"doc_id": f"d{i}"})]
            queries = [{"id": f"q{i}", "text": token, "user_id": f"u{i}",
                        "required_spans": [token], "evidence_doc_ids": [f"d{i}"]}]
            units.append(BenchmarkUnit(unit_id=f"u{i}", isolation_id=f"u{i}",
                                       documents=docs, queries=queries))
        return units

    def score(self, unit, responses):
        from memrank.metrics.scoring import score_query, spec_from_query
        by_id = {r.query_id: r for r in responses}
        hits = [1 if score_query(spec_from_query(q),
                                 (by_id.get(q["id"]).documents if by_id.get(q["id"]) else [])).hit
                else 0 for q in unit.queries]
        return {"composite": sum(hits) / len(hits), "per_category": {}, "n_queries": len(hits)}

    def report_template(self) -> str:
        return "# multifake {adapter} {composite}"


class JudgeFakeBenchmark(Benchmark):
    """One unit: a positive query, a negative query, and a goldless (unjudgeable) one.

    The third query is unjudgeable by MISSING GOLD, not by category: with judgeability declared
    per benchmark, an unknown category raises as a loader defect, and a missing gold answer is
    the one legitimate skip a binary benchmark has."""

    name = "judgefake"
    dataset_version = "judgefake@v1"

    def load(self) -> list[BenchmarkUnit]:
        docs = [Document(id="d1", content="favorite drink is green tea",
                         user_id="u1", metadata={"doc_id": "d1"})]
        queries = [
            {"id": "p1", "text": "drink?", "user_id": "u1", "category": "single-hop",
             "kind": "positive", "required_spans": ["green tea"], "gold_answers": ["green tea"]},
            {"id": "n1", "text": "allergic to peanuts?", "user_id": "u1",
             "category": "abstention", "kind": "negative",
             "forbidden_spans": ["allergic to peanuts"], "gold_answers": ["No"]},
            {"id": "x1", "text": "order events", "user_id": "u1",
             "category": "uncategorized", "kind": "positive",
             "required_spans": ["a"], "gold_answers": []},
        ]
        return [BenchmarkUnit(unit_id="u1", isolation_id="u1", documents=docs, queries=queries)]

    def score(self, unit, responses):
        return {"composite": 0.0, "per_category": {}, "n_queries": len(unit.queries)}

    def report_template(self) -> str:
        return "# judgefake {adapter} {composite}"


def make_fake_completer(*, answer="green tea", nc_answer="I don't know", suff=True, corr=True):
    """Scripted completer. Returns nc_answer when the snippet block is empty
    (no-context control); positive correctness fails for that nc_answer."""
    from memrank.judging import prompts as jp

    def _ctx_empty(user):
        # Memory snippets are the LAST wrapped block in answer_user, so the
        # content sits between the final sentinel pair (parts[-2]).
        parts = user.split(jp.DATA_SENTINEL)
        return len(parts) >= 3 and not parts[-2].strip()

    def complete(model, system, user):
        low = system.lower()
        if "answer the user's question" in low:
            return nc_answer if _ctx_empty(user) else answer
        if "should not answer affirmatively" in low:
            return '{"passed": true, "rationale": "abstained"}'
        if "grade whether the provided memory" in low:
            return '{"passed": %s, "rationale": "s"}' % ("true" if suff else "false")
        passed = (nc_answer not in user) and corr   # the nc "I don't know" answer fails
        return '{"passed": %s, "rationale": "c"}' % ("true" if passed else "false")
    return complete
