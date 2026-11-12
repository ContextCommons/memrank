"""Each accepted step names the questions it made final, as the result will record them."""

from __future__ import annotations

from memrank.service.protocol import AgentRef, Op, RunCreate, StepResult
from tests.service.conftest import create, post, step

REQUEST = RunCreate(evaluation="locomo", agent=AgentRef(name="a"))


def _to_ask(engine, run_id: str, lane: int = 0):
    """Reset and feed the lane's case; the ask step that follows."""
    for _ in (Op.RESET, Op.FEED):
        current = engine.next(run_id, lane, 2)
        assert engine.post(run_id, current.step_id, StepResult(ok=True, elapsed_ms=1)).outcomes \
            == []
    return engine.next(run_id, lane, 2)


def test_an_answer_makes_its_question_final_at_its_plan_position(engine):
    run_id = engine.create(REQUEST).run_id
    first = _to_ask(engine, run_id)
    engine.post(run_id, first.step_id, StepResult(answer="red", elapsed_ms=3))
    ask = engine.next(run_id, 0, 2)
    ack = engine.post(run_id, ask.step_id, StepResult(answer="blue", elapsed_ms=3))
    [final] = ack.outcomes
    assert final.position == 1 and final.outcome.question_id == "c1_q1"
    assert final.outcome.status == "not_judged" and final.outcome.answer == "blue"
    assert final.outcome.reference == ["blue"]  # released: its answer is final
    second = _to_ask(engine, run_id, lane=1)
    [last] = engine.post(run_id, second.step_id, StepResult(error="502", elapsed_ms=3)).outcomes
    assert (last.position, last.outcome.status, last.outcome.error) == (2, "failed", "502")


def test_a_failed_feed_makes_every_unasked_question_of_its_case_final(engine):
    run_id = engine.create(REQUEST).run_id
    reset = engine.next(run_id)
    engine.post(run_id, reset.step_id, StepResult(ok=True, elapsed_ms=1))
    feed = engine.next(run_id)
    ack = engine.post(run_id, feed.step_id, StepResult(error="feed refused", elapsed_ms=1))
    assert [(f.position, f.outcome.question_id, f.outcome.status) for f in ack.outcomes] == [
        (0, "c1_q0", "failed"), (1, "c1_q1", "failed")]
    assert ack.outcomes[0].outcome.error == "feed failed: feed refused"


def test_a_repeated_post_names_nothing_again(engine):
    run_id = engine.create(REQUEST).run_id
    ask = _to_ask(engine, run_id)
    result = StepResult(answer="blue", elapsed_ms=3)
    assert len(engine.post(run_id, ask.step_id, result).outcomes) == 1
    again = engine.post(run_id, ask.step_id, result)
    assert again.duplicate and again.outcomes == []


def test_the_live_outcomes_equal_the_sealed_result(engine):
    run_id = engine.create(REQUEST).run_id
    logged = {}
    while (current := engine.next(run_id)).op is not Op.DONE:
        result = (StepResult(answer="blue", elapsed_ms=2) if current.op is Op.ASK
                  else StepResult(ok=True, elapsed_ms=1))
        for final in engine.post(run_id, current.step_id, result).outcomes:
            logged[final.position] = final.outcome
    questions = [q for case in engine.result(run_id).cases for q in case.questions]
    assert [logged[i] for i in range(len(questions))] == questions


def test_status_says_how_the_plan_is_graded(engine):
    run_id = engine.create(REQUEST).run_id
    status = engine.status(run_id)
    assert status.uses_model and status.evaluation_source == "locomo"


def test_the_outcomes_travel_over_http(client):
    run_id = create(client)
    for _ in range(2):
        current = step(client, run_id)
        assert post(client, run_id, current["step_id"], ok=True).json()["outcomes"] == []
    ask = step(client, run_id)
    outcomes = post(client, run_id, ask["step_id"], answer="blue").json()["outcomes"]
    assert [(o["position"], o["outcome"]["question_id"]) for o in outcomes] == [(0, "c1_q0")]
