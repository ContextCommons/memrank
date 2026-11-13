"""The run's rules, one row per transition: every legal move, and each illegal neighbour.

Here a runner carries one step at a time, taking a question when one is offered and ingest
work otherwise -- the strictly sequential run. ``test_machine_pools.py`` has the pools side by
side.
"""

from __future__ import annotations

import pytest

from memrank.service import machine
from memrank.service.machine import ProtocolViolation, RunState, UnknownStep
from memrank.service.protocol import Op, Pool, StepResult

OK = StepResult(ok=True, elapsed_ms=1)
ANSWER = StepResult(answer="blue", elapsed_ms=1)
ERROR = StepResult(error="boom", elapsed_ms=1)


def fresh() -> RunState:
    return machine.new_state({"c1": 2, "c2": 1})


def take(state: RunState) -> tuple[RunState, machine.Outstanding | None]:
    """What a one-at-a-time runner is handed next: a question if any, else a reset or feed."""
    state, out = machine.next_step(state, Pool.ANSWER)
    if out is None:
        state, out = machine.next_step(state, Pool.INGEST)
    return state, out


def drive(state: RunState, results: list[StepResult]) -> RunState:
    """Take the next step and answer it with each result in turn."""
    for result in results:
        state, out = take(state)
        assert out is not None
        state, _ = machine.submit(state, out.step_id, result)
    return state


def question(state: RunState, out: machine.Outstanding) -> int:
    """The question ``out`` asks; for a reset or feed, the question its case asks next."""
    return out.question_index if out.question_index is not None \
        else state.cases[out.case].question_index


def where(state: RunState) -> tuple[Op | None, str | None, int]:
    """The op and case of the next step, and the question it asks or leads to."""
    state, out = take(state)
    if out is None:
        return None, None, 0
    return out.op, state.case_ids[out.case], question(state, out)


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
    state, out = take(drive(fresh(), history))
    with pytest.raises(ProtocolViolation):
        machine.submit(state, out.step_id, result)


def test_a_retried_identical_result_is_a_duplicate():
    state, out = take(fresh())
    state, _ = machine.submit(state, out.step_id, OK)
    again, duplicate = machine.submit(state, out.step_id, OK)
    assert duplicate and again == state


def test_a_step_cannot_be_answered_twice_with_different_content():
    state, out = take(drive(fresh(), [OK, OK]))
    state, _ = machine.submit(state, out.step_id, ANSWER)
    with pytest.raises(ProtocolViolation, match="different content"):
        machine.submit(state, out.step_id, StepResult(answer="red", elapsed_ms=1))


@pytest.mark.parametrize("step_id", ["s99", "x", "s"])
def test_a_step_never_issued_is_unknown(step_id):
    state, _ = take(fresh())
    with pytest.raises(UnknownStep):
        machine.submit(state, step_id, OK)


# -- a runner that starts: what the one before it held is lost (rule 4) -----------------------

def test_a_lost_reset_is_handed_out_again_under_the_same_session():
    state, first = take(fresh())
    before = machine.session_id("run0123456789ab", state, 0)
    state, restarted = machine.begin(state)
    state, second = take(state)
    assert restarted == [] and state.restarts == {}
    assert second.op is Op.RESET and second.case == 0 and second.step_id != first.step_id
    assert machine.session_id("run0123456789ab", state, 0) == before


@pytest.mark.parametrize("history", [[OK], [OK, OK], [OK, OK, ANSWER]],
                         ids=["lost-feed", "lost-ask", "lost-second-ask"])
def test_a_lost_feed_or_ask_restarts_the_case_from_reset(history):
    state, lost = take(drive(fresh(), history))
    asked_before = question(state, lost)
    state, restarted = machine.begin(state)
    state, again = take(state)
    assert restarted == ["c1"] and again.op is Op.RESET and again.case == 0
    assert state.restarts == {"c1": 1}
    state, _ = machine.submit(state, again.step_id, OK)
    state = drive(state, [OK])
    assert where(state) == (Op.ASK, "c1", asked_before)


def test_the_lost_step_is_refused_after_a_runner_begins():
    state, lost = take(drive(fresh(), [OK]))
    state, _ = machine.begin(state)
    with pytest.raises(ProtocolViolation, match="no longer outstanding"):
        machine.submit(state, lost.step_id, OK)


def test_the_session_id_changes_with_each_attempt():
    state, _ = take(drive(fresh(), [OK]))
    before = machine.session_id("run0123456789ab", state, 0)
    state, _ = machine.begin(state)
    assert machine.session_id("run0123456789ab", state, 0) != before


@pytest.mark.parametrize("history", [[], [OK], [OK, OK, ANSWER, ANSWER, OK, OK, ANSWER]],
                         ids=["nothing-started", "a-reset-acknowledged", "every-case-finished"])
def test_a_runner_beginning_with_no_case_fed_and_unfinished_changes_nothing(history):
    state = drive(fresh(), history)
    assert machine.begin(state) == (state, [])


def test_a_fed_unfinished_case_restarts_with_nothing_outstanding():
    state, restarted = machine.begin(drive(fresh(), [OK, OK, ANSWER]))
    assert restarted == ["c1"] and state.restarts == {"c1": 1}
    assert where(state) == (Op.RESET, "c1", 1)  # fed again, then the question still owed


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
    state, lost = take(state)  # the ask of question 2, then the runner crashes
    assert lost.op is Op.ASK and lost.question_index == 2
    state, _ = machine.begin(state)
    assert where(state) == (Op.RESET, "c1", 2)


def test_without_reopening_a_failed_case_stays_finished():
    state = drive(fresh(), [OK, OK, ANSWER, ANSWER, ERROR])
    assert state.done and where(state) == (None, None, 0)


@pytest.mark.parametrize("fields", [{}, {"ok": True, "answer": "a"}, {"ok": False}],
                         ids=["none", "two", "ok-false"])
def test_a_result_carries_exactly_one_outcome(fields):
    with pytest.raises(ValueError):
        StepResult(elapsed_ms=1, **fields)
