from pathlib import Path

from memrank import benchmarks
from memrank.benchmarks.demo import DemoBenchmark
from memrank.core import AdapterResponse, Document


def _resp(qid, *contents):
    docs = [Document(id=f"m{i}", content=c, metadata={"doc_id": _src(c)})
            for i, c in enumerate(contents)]
    return AdapterResponse(query_id=qid, documents=docs)


def _src(content):
    if "marine biologist" in content or "blue whale" in content:
        return "sess_1"
    if "april" in content.lower():
        return "sess_2"
    return "sess_3"


def test_load_returns_one_unit_with_five_queries():
    units = DemoBenchmark().load()
    assert len(units) == 1
    assert len(units[0].queries) == 5


def test_perfect_retrieval_scores_high():
    unit = DemoBenchmark().load()[0]
    responses = [
        _resp("q_job", "Alex is a marine biologist in Portland."),
        _resp("q_animal", "favorite animal is the blue whale"),
        _resp("q_visit", "Dana is visiting in April"),
        _resp("q_diet", "Alex orders vegetarian sushi"),
        _resp("q_allergy_neg", "Alex enjoys hiking"),
    ]
    score = DemoBenchmark().score(unit, responses)
    assert score["composite"] == 1.0


def test_negative_query_fails_when_wrong_memory_surfaced():
    unit = DemoBenchmark().load()[0]
    responses = [_resp("q_allergy_neg", "Alex is allergic to peanuts")]
    score = DemoBenchmark().score(unit, responses)
    assert score["per_category"]["abstention"] == 0.0


def test_distractor_does_not_false_hit_on_diet():
    unit = DemoBenchmark().load()[0]
    responses = [_resp("q_diet", "Sam loves regular sushi")]
    score = DemoBenchmark().score(unit, responses)
    assert score["per_category"]["preference"] == 0.0


def test_the_bundled_scenario_is_found_inside_the_package():
    """ATO-2059: the scenario must be reachable from the package, not from the repository.

    It used to be resolved by walking two directories up to `examples/data/`, which exists only
    in a checkout -- so a `pip install memrank` had no demo at all. Anchoring the assertion on
    `memrank.benchmarks.__file__` is what makes it fail if the loader starts walking again: a
    path built from the repository root would satisfy `.exists()` here and nowhere else.
    """
    package = Path(benchmarks.__file__).resolve().parent
    scenario = DemoBenchmark()._path
    assert scenario == package / "data" / "demo_scenario.json"
    assert scenario.is_file()


def test_an_override_still_wins_over_the_bundled_scenario(tmp_path, monkeypatch):
    """DEMO_DATA_PATH keeps pointing the loader elsewhere, and keeps marking it non-synthetic."""
    elsewhere = tmp_path / "other.json"
    monkeypatch.setenv("DEMO_DATA_PATH", str(elsewhere))
    benchmark = DemoBenchmark()
    assert benchmark._path == elsewhere
    assert benchmark.is_synthetic is False
    assert benchmark.question_text_public is False
