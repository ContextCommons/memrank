"""The service over HTTP: a whole run, the refusals, a resumed runner, and resumable judging."""

from __future__ import annotations

import pytest

from tests.service.conftest import create, post, step


def run_to_end(client, run_id: str, answers: dict[str, dict]) -> None:
    """Answer every step: ok for reset/feed, ``answers[case_id]`` for each ask."""
    while (current := step(client, run_id))["op"] != "done":
        outcome = answers[current["case_id"]] if current["op"] == "ask" else {"ok": True}
        assert post(client, run_id, current["step_id"], **outcome).status_code == 200


def test_a_whole_run_records_every_answer_and_failure_ungraded(client):
    run_id = create(client)
    run_to_end(client, run_id, {"c1": {"answer": "It is blue."}, "c2": {"error": "timeout"}})
    result = client.get(f"/v1/runs/{run_id}/result").json()
    assert result["status"] == "complete" and result["judged"] is False
    assert result["notice"].startswith("NOT JUDGED YET")
    assert result["questions"] == 3 and result["failed"] == 1 and result["score"]["mean"] is None
    first = result["cases"][0]["questions"][0]
    assert (first["question"], first["reference"], first["answer"], first["status"]) == (
        "question c1_q0?", ["blue"], "It is blue.", "not_judged")
    assert first["grading"] == {"gold_answers": ["blue"]}
    assert result["cases"][1]["questions"][0]["error"] == "timeout"


def test_the_result_names_what_was_run(client):
    run_id = create(client)
    run_to_end(client, run_id, {"c1": {"answer": "blue"}, "c2": {"answer": "blue"}})
    identity = client.get(f"/v1/runs/{run_id}/result").json()["identity"]
    assert identity["dataset_version"] == "test@v1" and identity["task_version"] == 2
    assert identity["case_ids"] == ["c1", "c2"] and identity["seed"] == 0
    assert identity["judge_requested"] is True and identity["judge_model"] is None
    assert identity["memrank_version"]


def test_the_agent_sees_history_then_questions_and_never_the_reference(client):
    run_id = create(client)
    reset = step(client, run_id)
    assert reset["op"] == "reset" and reset["sessions"] is None
    post(client, run_id, reset["step_id"], ok=True)
    feed = step(client, run_id)
    assert feed["sessions"][0]["turns"][0]["text"] == "My car is blue."
    assert feed["question"] is None
    post(client, run_id, feed["step_id"], ok=True)
    ask = step(client, run_id)
    assert ask["question"]["text"] == "question c1_q0?"
    assert "blue" not in str(ask["question"]) and "category" not in ask["question"]


def test_no_result_and_so_no_reference_is_handed_out_before_the_run_ends(client):
    run_id = create(client)
    assert client.get(f"/v1/runs/{run_id}/result").status_code == 409


def test_a_repeated_post_is_a_duplicate_and_a_changed_one_is_refused(client):
    run_id = create(client)
    first = step(client, run_id)
    assert post(client, run_id, first["step_id"], ok=True).json()["duplicate"] is False
    assert post(client, run_id, first["step_id"], ok=True).json()["duplicate"] is True
    refused = post(client, run_id, first["step_id"], error="later")
    assert refused.status_code == 409 and "different content" in refused.json()["detail"]


def test_unknown_runs_and_steps_are_not_found(client):
    assert client.post("/v1/runs/nope/next").status_code == 404
    run_id = create(client)
    assert post(client, run_id, "s42", ok=True).status_code == 404


def test_a_runner_that_lost_a_feed_resumes_from_a_fresh_reset(client):
    run_id = create(client)
    post(client, run_id, step(client, run_id)["step_id"], ok=True)
    lost = step(client, run_id)
    resumed = step(client, run_id)
    assert resumed["op"] == "reset" and resumed["session_id"] != lost["session_id"]
    assert post(client, run_id, lost["step_id"], ok=True).status_code == 409


def test_a_run_with_judging_off_says_so(client):
    run_id = client.post("/v1/runs", json={"evaluation": "locomo", "agent": {"name": "a"},
                                           "judge": False}).json()["run_id"]
    run_to_end(client, run_id, {"c1": {"answer": "blue"}, "c2": {"error": "boom"}})
    result = client.get(f"/v1/runs/{run_id}/result").json()
    assert result["judged"] is False and result["notice"].startswith("NOT JUDGED:")
    assert result["identity"]["judge_requested"] is False
    assert result["failure_rate"] == pytest.approx(1 / 3)


def test_lanes_are_query_parameters_and_each_gets_its_own_case(client):
    run_id = create(client)
    first = client.post(f"/v1/runs/{run_id}/next?lane=0&lanes=2").json()
    second = client.post(f"/v1/runs/{run_id}/next?lane=1&lanes=2").json()
    assert (first["case_id"], second["case_id"]) == ("c1", "c2")
    refused = client.post(f"/v1/runs/{run_id}/next?lane=2&lanes=2")
    assert refused.status_code == 409 and "not one of the runner's" in refused.text


def test_status_counts_progress_and_hands_out_no_answer_or_reference(client):
    run_id = create(client)
    for result in ({"ok": True}, {"ok": True}, {"answer": "blue"}):
        post(client, run_id, step(client, run_id)["step_id"], **result)
    status = client.get(f"/v1/runs/{run_id}/status").json()
    assert status["cases"] == 2 and status["questions"] == 3
    assert status["cases_fed"] == 1 and status["fed_ms"] == 5 and status["done"] is False
    assert status["questions_done"] == 1 and status["asked_ms"] == 5
    assert "blue" not in str(status)


def test_a_finished_run_retries_its_failed_case_only_when_asked(client):
    """A feed that failed stays failed on a resume; ``retry-failed`` runs that case again from a
    fresh reset, and the result then holds its answers."""
    run_id = create(client)
    while (current := step(client, run_id))["op"] != "done":
        outcome = ({"error": "engine down"} if (current["case_id"], current["op"]) == ("c2", "feed")
                   else {"answer": "blue"} if current["op"] == "ask" else {"ok": True})
        post(client, run_id, current["step_id"], **outcome)
    before = client.get(f"/v1/runs/{run_id}/result").json()["cases"][1]
    assert before["status"] == "failed" and before["error"] == "feed failed: engine down"
    reopened = client.post(f"/v1/runs/{run_id}/retry-failed")
    assert reopened.status_code == 200 and reopened.json()["cases"] == ["c2"]
    assert client.get(f"/v1/runs/{run_id}/result").status_code == 409  # not finished again
    run_to_end(client, run_id, {"c2": {"answer": "blue"}})
    after = client.get(f"/v1/runs/{run_id}/result").json()
    assert after["cases"][1]["status"] == "complete" and after["cases"][1]["restarts"] == 1
    assert after["failed"] == 0 and after["cases"][0]["restarts"] == 0


def test_retrying_an_unknown_run_is_not_found(client):
    assert client.post("/v1/runs/nope/retry-failed").status_code == 404
