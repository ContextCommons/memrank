"""The agent's reported retrieval time: kept per answer and summarized apart from latency_ms."""

from __future__ import annotations

import pytest

from tests.service.conftest import create, post, step
from tests.service.test_app import run_to_end


def test_reported_retrieval_is_summarized_over_the_answers_that_carried_it(client):
    run_id = create(client)
    run_to_end(client, run_id, {"c1": {"answer": "blue", "retrieval_ms": 180},
                                "c2": {"answer": "blue"}})
    result = client.get(f"/v1/runs/{run_id}/result").json()
    assert result["retrieval"] == {"p50_ms": 180.0, "p95_ms": 180.0, "reported": 2,
                                   "answers": 3}
    assert sorted(result["latency_ms"]) == ["ask_p50", "ask_p95", "feed_p50", "feed_p95"]
    times = [q["retrieval_ms"] for case in result["cases"] for q in case["questions"]]
    assert times == [180.0, 180.0, None]  # not reported is None, never 0


def test_a_run_whose_agent_reports_nothing_has_no_retrieval(client):
    run_id = create(client)
    run_to_end(client, run_id, {"c1": {"answer": "blue"}, "c2": {"error": "timeout"}})
    assert client.get(f"/v1/runs/{run_id}/result").json()["retrieval"] is None


@pytest.mark.parametrize("fields", [{"ok": True, "retrieval_ms": 5},
                                    {"answer": "blue", "retrieval_ms": -1}])
def test_a_retrieval_time_off_an_answer_or_below_zero_is_refused(client, fields):
    run_id = create(client)
    current = step(client, run_id)
    assert post(client, run_id, current["step_id"], **fields).status_code == 422
