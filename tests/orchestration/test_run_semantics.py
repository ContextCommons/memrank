from memrank.runner import run_cell
from tests.fakes import FakeAdapter, FakeBenchmark


def test_repeats_ingest_once_and_retrieve_only():
    a = FakeAdapter(name="fake", responses={"q1": [], "q2": []})
    result = run_cell(a, FakeBenchmark(), k=10, repeats=3,
                      run_id_prefix="run-abc", model="gpt-4o-mini",
                      token_budget=5000).to_dict()
    assert a.ingest_calls == 1
    assert result["repeats"] == 3


def test_unique_run_id_is_used_as_isolation():
    a = FakeAdapter(name="fake", responses={})
    run_cell(a, FakeBenchmark(), k=10, repeats=1,
             run_id_prefix="run-XYZ", model="gpt-4o-mini", token_budget=5000).to_dict()
    assert any("run-XYZ" in p for p in a.prepare_calls)


def test_warmup_excluded_from_retrieve_latency_count():
    # 2 queries x 3 repeats = 6 measured; the one warm-up retrieve is NOT counted.
    a = FakeAdapter(name="fake", responses={"q1": [], "q2": []})
    result = run_cell(a, FakeBenchmark(), k=10, repeats=3,
                      run_id_prefix="r", model="gpt-4o-mini", token_budget=5000).to_dict()
    assert result["retrieve_latency_summary"]["count"] == 6


def test_documents_reingested_under_run_id():
    # docs are re-scoped to the run_id so engines keying on doc.user_id stay consistent
    captured: list[str] = []

    class CapturingAdapter(FakeAdapter):
        def ingest(self, documents):
            captured.extend(d.user_id for d in documents)
            super().ingest(documents)

    run_cell(CapturingAdapter(name="cap", responses={}), FakeBenchmark(),
             k=10, repeats=1, run_id_prefix="run-ZZZ", model="gpt-4o-mini",
             token_budget=5000).to_dict()
    assert captured  # FakeBenchmark has one doc
    assert all("run-ZZZ" in uid for uid in captured)


def test_result_has_per_query_retrieved_docs_and_match_info():
    from memrank.core import Document
    a = FakeAdapter(name="fake", responses={
        "q1": [Document(id="m1", content="blue whale", metadata={"doc_id": "d1"})],
        "q2": [],
    })
    result = run_cell(a, FakeBenchmark(), k=10, repeats=1,
                      run_id_prefix="r", model="gpt-4o-mini", token_budget=5000).to_dict()
    drill = result["per_query"]
    q1 = next(d for d in drill if d["query_id"] == "q1")
    assert q1["hit"] is True
    assert q1["matched_doc_id"] == "m1"
    assert q1["retrieved"][0]["content"] == "blue whale"
