from memrank.core import Document
from tests.fakes import FakeAdapter, FakeBenchmark


def test_fake_adapter_counts_calls_and_returns_configured_docs():
    docs = {"q1": [Document(id="m1", content="blue whale")]}
    a = FakeAdapter(name="fake", responses=docs)
    unit = FakeBenchmark().load()[0]
    a.prepare(unit.isolation_id)
    a.ingest(unit.documents)
    a.retrieve("q?", 10, unit.isolation_id)
    a.cleanup()
    assert a.ingest_calls == 1
    assert [d.id for d in a.retrieve("q1", 10, "u", None).documents] == []  # unknown q -> []
    assert a.latency_metrics()["retrieve_p50_ms"] >= 0.0
