"""The run's rules, one row per transition: every legal move, and each illegal neighbour.

The first half drives one lane, which is the strictly sequential run. The second half scripts
several lanes interleaving, each row a sequence of moves and what each lane was handed.
"""

from __future__ import annotations

import pytest

from memrank.service import machine
from memrank.service.machine import ProtocolViolation, RunState, UnknownStep
from memrank.service.protocol import Op, StepResult

OK = StepResult(ok=True, elapsed_ms=1)
ANSWER = StepResult(answer="blue", elapsed_ms=1)
ERROR = StepResult(error="boom", elapsed_ms=1)


def fresh() -> RunState:
    return machine.new_state({"c1": 2, "c2": 1})


def drive(state: RunState, results: list[StepResult]) -> RunState:
    """Take lane 0's next step and answer it with each result in turn."""
    for result in results:
        state, out = machine.next_step(state)
        assert out is not None
        state, _ = machine.submit(state, out.step_id, result)
    return state


def where(state: RunState) -> tuple[Op | None, str | None, int]:
    """The op and case lane 0's next step would be, and that case's question index."""
    state, out = machine.next_step(state)
    if out is None:
        return None, None, 0
    return out.op, state.case_ids[out.case], state.cases[out.case].question_index


@pytest.mark.parametrize(("history", "expected"), [
    ([], (Op.RESET, "c1", 0)),
    ([OK], (Op.FEED, "c1", 0)),
    ([OK, OK], (Op.ASK, "c1", 0)),
    ([OK, OK, ANSWER], (Op.ASK, "c1", 1)),
    ([OK, OK, ERROR], (Op.ASK, "c1", 1)),
    ([OK, OK, ANSWER, ANSWER], (Op.RESET, "c2", 0)),
    ([ERROR], (Op.RESET, "c2", 0)),
    ([OK, ERROR], (Op.RESET, "c2", 0)),
    ([OK, OK, ANSWER, ANSWER, OK, OK, ANSWER], (None, None, 0)),
], ids=["starts-with-reset", "reset-then-feed", "feed-then-ask", "next-question",
        "ask-error-moves-on", "next-case-starts-with-reset", "reset-error-fails-case",
        "feed-error-fails-case", "done"])
def test_legal_transitions(history, expected):
    assert where(drive(fresh(), history)) == expected


def test_a_failed_reset_or_feed_fails_the_case():
    assert drive(fresh(), [OK, ERROR]).failed_cases == {"c1": "feed failed: boom"}


@pytest.mark.parametrize(("history", "result"), [
    ([], ANSWER), ([OK], ANSWER), ([OK, OK], OK),
], ids=["answer-to-reset", "answer-to-feed", "ok-to-ask"])
def test_a_result_of_the_wrong_kind_is_refused(history, result):
    state, out = machine.next_step(drive(fresh(), history))
    with pytest.raises(ProtocolViolation):
        machine.submit(state, out.step_id, result)


def test_a_retried_identical_result_is_a_duplicate():
    state, out = machine.next_step(fresh())
    state, _ = machine.submit(state, out.step_id, OK)
    again, duplicate = machine.submit(state, out.step_id, OK)
    assert duplicate and again == state


def test_a_step_cannot_be_answered_twice_with_different_content():
    state, out = machine.next_step(drive(fresh(), [OK, OK]))
    state, _ = machine.submit(state, out.step_id, ANSWER)
    with pytest.raises(ProtocolViolation, match="different content"):
        machine.submit(state, out.step_id, StepResult(answer="red", elapsed_ms=1))


@pytest.mark.parametrize("step_id", ["s99", "x", "s"])
def test_a_step_never_issued_is_unknown(step_id):
    state, _ = machine.next_step(fresh())
    with pytest.raises(UnknownStep):
        machine.submit(state, step_id, OK)


def test_a_lost_reset_is_reissued_unchanged():
    state, first = machine.next_step(fresh())
    state, second = machine.next_step(state)
    assert second == first and state.restarts == {}


@pytest.mark.parametrize("history", [[OK], [OK, OK], [OK, OK, ANSWER]],
                         ids=["lost-feed", "lost-ask", "lost-second-ask"])
def test_a_lost_feed_or_ask_restarts_the_case_from_reset(history):
    state, lost = machine.next_step(drive(fresh(), history))
    asked_before = state.cases[0].question_index
    state, again = machine.next_step(state)
    assert again.op is Op.RESET and again.step_id != lost.step_id
    assert state.restarts == {"c1": 1}
    state = drive(state, [OK, OK])
    assert where(state) == (Op.ASK, "c1", asked_before)


def test_the_lost_step_is_refused_after_the_restart():
    state, lost = machine.next_step(drive(fresh(), [OK]))
    state, _ = machine.next_step(state)
    with pytest.raises(ProtocolViolation, match="no longer outstanding"):
        machine.submit(state, lost.step_id, OK)


def test_the_session_id_changes_with_each_attempt():
    state, _ = machine.next_step(drive(fresh(), [OK]))
    before = machine.session_id("run0123456789ab", state, 0)
    state, _ = machine.next_step(state)
    assert machine.session_id("run0123456789ab", state, 0) != before


#: (history, what reopening returns, where lane 0 goes next, restarts after)
REOPENED = [
    ("feed-failed-case-restarts", [OK, ERROR], ["c1"], (Op.RESET, "c1", 0), {"c1": 1}),
    ("reset-failed-case-restarts", [ERROR], ["c1"], (Op.RESET, "c1", 0), {"c1": 1}),
    ("finished-run-reopens-only-the-failed-case", [OK, OK, ANSWER, ANSWER, ERROR], ["c2"],
     (Op.RESET, "c2", 0), {"c2": 1}),
    ("nothing-failed-changes-nothing", [OK, OK, ANSWER], [], (Op.ASK, "c1", 1), {}),
    ("complete-case-with-failed-answer-restarts-at-it", [OK, OK, ANSWER, ERROR, OK, OK, ANSWER],
     ["c1"], (Op.RESET, "c1", 1), {"c1": 1}),
    ("failed-answer-and-failed-case-both-reopen", [OK, OK, ERROR, ANSWER, OK, ERROR],
     ["c1", "c2"], (Op.RESET, "c1", 0), {"c1": 1, "c2": 1}),
]


@pytest.mark.parametrize(("history", "reopened", "expected", "restarts"),
                         [row[1:] for row in REOPENED], ids=[row[0] for row in REOPENED])
def test_reopening_failed_cases(history, reopened, expected, restarts):
    before = drive(fresh(), history)
    state, ids = machine.reopen_failed(before)
    assert ids == reopened and where(state) == expected
    assert state.restarts == restarts and state.failed_cases == {}
    assert (state == before) == (not reopened)


def test_a_reopened_case_runs_under_a_new_session_and_leaves_finished_cases_alone():
    failed = drive(fresh(), [OK, OK, ANSWER, ANSWER, OK, ERROR])
    assert failed.done and failed.failed_cases == {"c2": "feed failed: boom"}
    state, _ = machine.reopen_failed(failed)
    assert not state.done and state.cases[0] == failed.cases[0]
    assert machine.session_id("run0123456789ab", state, 1) != \
        machine.session_id("run0123456789ab", failed, 1)
    state = drive(state, [OK, OK, ANSWER])
    assert state.done and state.failed_cases == {}


def asked(state: RunState, case_id: str) -> list[tuple[int, str | None]]:
    """Every (question index, answer) ``case_id`` recorded, in step order."""
    return [(r.question_index, r.result.answer) for r in state.recorded.values()
            if r.op is Op.ASK and r.case_id == case_id]


def test_a_reopened_complete_case_is_fed_again_and_asks_only_its_failed_answers():
    three = machine.new_state({"c1": 3})
    finished = drive(three, [OK, OK, ERROR, ANSWER, ERROR])
    assert finished.done and finished.failed_cases == {}
    state, ids = machine.reopen_failed(finished)
    assert ids == ["c1"] and state.cases[0].retry == [0, 2]
    retried = drive(state, [OK, OK, StepResult(answer="red", elapsed_ms=1), ANSWER])
    assert retried.done and retried.restarts == {"c1": 1}
    assert asked(retried, "c1")[3:] == [(0, "red"), (2, "blue")]
    assert machine.reopen_failed(retried) == (retried, [])


def test_an_answer_that_fails_again_is_reopened_again():
    state, _ = machine.reopen_failed(drive(fresh(), [OK, OK, ANSWER, ERROR, OK, OK, ANSWER]))
    state = drive(state, [OK, OK, ERROR])
    assert state.done
    again, ids = machine.reopen_failed(state)
    assert ids == ["c1"] and again.cases[0].retry == [1] and again.restarts == {"c1": 2}


def test_a_lost_ask_during_a_retry_restarts_at_the_same_question():
    state, _ = machine.reopen_failed(drive(machine.new_state({"c1": 3}),
                                           [OK, OK, ERROR, ANSWER, ERROR]))
    state = drive(state, [OK, OK, ANSWER])
    state, lost = machine.next_step(state)  # the ask of question 2, then the runner crashes
    assert lost.op is Op.ASK and state.cases[0].question_index == 2
    assert where(state) == (Op.RESET, "c1", 2)


def test_without_reopening_a_failed_case_stays_finished():
    state = drive(fresh(), [OK, OK, ANSWER, ANSWER, ERROR])
    assert state.done and where(state) == (None, None, 0)


@pytest.mark.parametrize("fields", [{}, {"ok": True, "answer": "a"}, {"ok": False}],
                         ids=["none", "two", "ok-false"])
def test_a_result_carries_exactly_one_outcome(fields):
    with pytest.raises(ValueError):
        StepResult(elapsed_ms=1, **fields)


# -- lanes: cases side by side ------------------------------------------------------------------

def play(script: list[tuple], lanes: int, cases: dict[str, int] | None = None):
    """Run ``script`` -- ("next", lane) or ("post", lane, result) -- over ``lanes`` lanes.

    Returns every step handed out as ``(lane, op, case id, question index)``, ``None`` for a lane
    told nothing is left for it, and the final state.
    """
    state = machine.new_state(cases or {"c1": 2, "c2": 1, "c3": 1})
    held: dict[int, str] = {}
    handed: list = []
    for move in script:
        if move[0] == "next":
            state, out = machine.next_step(state, move[1], lanes)
            handed.append(None if out is None else (
                move[1], out.op, state.case_ids[out.case], state.cases[out.case].question_index))
            if out is not None:
                held[move[1]] = out.step_id
        else:
            state, _ = machine.submit(state, held[move[1]], move[2])
    return handed, state


def steps(lane: int, *results: StepResult) -> list[tuple]:
    return [move for result in results for move in (("next", lane), ("post", lane, result))]


R, F, A = Op.RESET, Op.FEED, Op.ASK

#: (case, lanes, script, what was handed out, in order)
INTERLEAVED = [
    ("lanes-take-cases-in-order", 3, [("next", 0), ("next", 1), ("next", 2)],
     [(0, R, "c1", 0), (1, R, "c2", 0), (2, R, "c3", 0)]),
    ("each-case-keeps-its-own-order", 2,
     [("next", 0), ("next", 1), ("post", 1, OK), ("post", 0, OK), ("next", 1), ("next", 0),
      ("post", 0, OK), ("post", 1, OK), ("next", 0), ("next", 1)],
     [(0, R, "c1", 0), (1, R, "c2", 0), (1, F, "c2", 0), (0, F, "c1", 0), (0, A, "c1", 0),
      (1, A, "c2", 0)]),
    ("a-finished-lane-takes-the-next-free-case", 2,
     [("next", 0), *steps(1, OK, OK, ANSWER), ("next", 1)],
     [(0, R, "c1", 0), (1, R, "c2", 0), (1, F, "c2", 0), (1, A, "c2", 0), (1, R, "c3", 0)]),
    ("a-lane-with-nothing-left-is-told-so", 4,
     [("next", 0), ("next", 1), ("next", 2), ("next", 3)],
     [(0, R, "c1", 0), (1, R, "c2", 0), (2, R, "c3", 0), None]),
    ("a-lost-step-restarts-only-that-lanes-case", 2,
     [*steps(0, OK), ("next", 0), ("next", 1), ("next", 0)],
     [(0, R, "c1", 0), (0, F, "c1", 0), (1, R, "c2", 0), (0, R, "c1", 0)]),
]


@pytest.mark.parametrize(("lanes", "script", "expected"), [row[1:] for row in INTERLEAVED],
                         ids=[row[0] for row in INTERLEAVED])
def test_interleaved_lanes(lanes, script, expected):
    handed, _ = play(script, lanes)
    assert handed == expected


def test_a_lost_step_in_one_lane_leaves_the_other_lanes_case_alone():
    _, state = play([*steps(0, OK), ("next", 0), *steps(1, OK), ("next", 1), ("next", 0)], 2)
    assert state.restarts == {"c1": 1}
    assert state.cases[1].outstanding.op is Op.FEED and "c2" not in state.restarts


def test_a_resume_with_fewer_lanes_adopts_the_orphaned_case_as_lost():
    _, state = play([*steps(0, OK), ("next", 0), *steps(1, OK, OK), ("next", 1)], 2)
    state, mine = machine.next_step(state, 0, 1)  # lane 0's own lost feed restarts c1
    state = drive(state, [OK, OK, ANSWER, ANSWER])
    state, adopted = machine.next_step(state, 0, 1)
    assert (mine.op, mine.case) == (Op.RESET, 0)
    assert (adopted.op, adopted.case) == (Op.RESET, 1) and state.restarts == {"c1": 1, "c2": 1}


def test_every_case_finishes_whatever_the_interleaving():
    script = [*steps(1, OK, OK, ANSWER), *steps(0, OK, OK, ANSWER, ERROR), *steps(1, ERROR)]
    _, state = play(script, 2)
    assert state.done and state.failed_cases == {"c3": "reset failed: boom"}


@pytest.mark.parametrize(("lane", "lanes"), [(2, 2), (-1, 2), (0, 0)])
def test_a_lane_the_runner_does_not_have_is_refused(lane, lanes):
    with pytest.raises(ProtocolViolation, match="not one of the runner's"):
        machine.next_step(fresh(), lane, lanes)


def test_a_retry_is_recognised_while_other_lanes_move():
    state = fresh()
    state, first = machine.next_step(state, 0, 2)
    state, _ = machine.submit(state, first.step_id, OK)
    state, other = machine.next_step(state, 1, 2)
    state, _ = machine.submit(state, other.step_id, OK)
    again, duplicate = machine.submit(state, first.step_id, OK)
    assert duplicate and again == state
