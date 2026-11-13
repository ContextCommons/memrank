"""The ingest and answer pools side by side, one row per transition.

Each row scripts moves -- ask a pool for a step, post the n-th step handed out, a runner
beginning -- over three cases of three, two and one questions, and lists what each ask was
handed: the legal moves, and beside them the neighbours a pool must not hand out (None).
"""

from __future__ import annotations

import pytest

from memrank.service import machine
from memrank.service.machine import ProtocolViolation, RunState
from memrank.service.protocol import Op, Pool, StepResult

OK = StepResult(ok=True, elapsed_ms=1)
ERROR = StepResult(error="boom", elapsed_ms=1)
R, F, A = Op.RESET, Op.FEED, Op.ASK
ING, Q, QO = ("next", Pool.INGEST, False), ("next", Pool.ANSWER, False), ("next", Pool.ANSWER, True)


def answer(text: str) -> StepResult:
    return StepResult(answer=text, elapsed_ms=1)


def post(n: int, result: StepResult = OK) -> tuple:
    return ("post", n, result)


BEGIN = ("begin",)


def fresh() -> RunState:
    return machine.new_state({"c1": 3, "c2": 2, "c3": 1})


def play(script: list[tuple], state: RunState | None = None) -> tuple[list, RunState]:
    """Run ``script``; return what each ``next`` was handed, ``(op, case id, question)`` or
    None, and the final state. ``post n`` posts the n-th step handed out (from 0)."""
    state = state or fresh()
    issued: list[str] = []
    handed: list = []
    for move in script:
        if move[0] == "next":
            state, out = machine.next_step(state, move[1], move[2])
            handed.append(None if out is None else
                          (out.op, state.case_ids[out.case], out.question_index))
            issued.append("" if out is None else out.step_id)
        elif move[0] == "post":
            state, _ = machine.submit(state, issued[move[1]], move[2])
        else:
            state, _ = machine.begin(state)
    return handed, state


#: Cases c1 and c2 reset and fed: the first four steps handed out.
FED_TWO = [ING, ING, post(0), post(1), ING, ING, post(2), post(3)]

#: (case, script, what each next was handed)
LEGAL = [
    ("ingest-hands-out-resets-in-plan-order", [ING, ING, ING, ING],
     [(R, "c1", None), (R, "c2", None), (R, "c3", None), None]),
    ("a-reset-acknowledged-makes-its-feed-next", [ING, ING, post(1), post(0), ING, ING],
     [(R, "c1", None), (R, "c2", None), (F, "c1", None), (F, "c2", None)]),
    ("no-question-before-its-feed-is-acknowledged", [ING, post(0), ING, Q],
     [(R, "c1", None), (F, "c1", None), None]),
    ("a-reset-out-is-the-cases-only-step", [ING, Q, ING],
     [(R, "c1", None), None, (R, "c2", None)]),
    ("questions-spread-across-fed-cases", [*FED_TWO, Q, Q, Q, Q, Q, Q],
     [(R, "c1", None), (R, "c2", None), (F, "c1", None), (F, "c2", None),
      (A, "c1", 0), (A, "c1", 1), (A, "c1", 2), (A, "c2", 0), (A, "c2", 1), None]),
    ("ingest-goes-on-while-questions-are-out", [*FED_TWO, Q, ING],
     [(R, "c1", None), (R, "c2", None), (F, "c1", None), (F, "c2", None),
      (A, "c1", 0), (R, "c3", None)]),
    ("in-order-one-question-per-case-at-a-time", [*FED_TWO, QO, QO, QO],
     [(R, "c1", None), (R, "c2", None), (F, "c1", None), (F, "c2", None),
      (A, "c1", 0), (A, "c2", 0), None]),
    ("in-order-the-next-question-once-the-last-is-answered",
     [*FED_TWO, QO, QO, post(4, answer("a")), QO],
     [(R, "c1", None), (R, "c2", None), (F, "c1", None), (F, "c2", None),
      (A, "c1", 0), (A, "c2", 0), (A, "c1", 1)]),
    ("a-failed-feed-offers-none-of-its-questions", [ING, post(0), ING, post(1, ERROR), Q, ING],
     [(R, "c1", None), (F, "c1", None), None, (R, "c2", None)]),
]


@pytest.mark.parametrize(("script", "expected"), [row[1:] for row in LEGAL],
                         ids=[row[0] for row in LEGAL])
def test_legal_moves(script, expected):
    handed, _ = play(script)
    assert handed == expected


def test_a_case_finishes_at_its_last_answer_whatever_order_they_come_back_in():
    _, state = play([*FED_TWO, Q, Q, Q, post(6, answer("c")), post(4, answer("a"))])
    assert not state.cases[0].finished and len(state.cases[0].outstanding) == 1
    state, _ = machine.submit(state, state.cases[0].outstanding[0].step_id, answer("b"))
    recorded = {r.question_index: r.result.answer for r in state.recorded.values()
                if r.op is A and r.case_id == "c1"}
    assert state.cases[0].finished and recorded == {0: "a", 1: "b", 2: "c"}


def test_answers_are_recorded_against_their_questions_whatever_order_they_land_in():
    results = []
    for order in ([4, 5, 6], [6, 4, 5]):
        _, state = play([*FED_TWO, Q, Q, Q, *[post(n, answer(f"q{n - 4}")) for n in order]])
        results.append({r.question_index: r.result.answer for r in state.recorded.values()
                        if r.op is A})
    assert results[0] == results[1] == {0: "q0", 1: "q1", 2: "q2"}


def test_the_run_is_done_when_neither_pool_offers_a_step_and_none_is_out():
    script = [ING, ING, ING, post(0), post(1), post(2), ING, ING, ING, post(3), post(4), post(5),
              Q, Q, Q, Q, Q, Q, *[post(n, answer("x")) for n in range(6, 12)], ING, Q]
    handed, state = play(script)
    assert handed[-2:] == [None, None] and state.done


# -- a runner begins: whatever was outstanding is lost (rule 4) --------------------------------

#: (case, script before the runner begins, cases restarted, then per case: retry, next question)
LOST = [
    ("only-resets-out-restart-nothing", [ING, ING], [],
     {"c1": ([], 0), "c2": ([], 0)}),
    ("asks-out-in-two-cases-restart-both", [*FED_TWO, Q, Q, Q, Q], ["c1", "c2"],
     {"c1": ([], 0), "c2": ([], 0)}),
    ("an-answered-middle-question-is-kept", [*FED_TWO, Q, Q, Q, post(5, answer("b"))],
     ["c1", "c2"], {"c1": ([0, 2], 0), "c2": ([], 0)}),
    ("an-answered-first-question-runs-on-from-the-second",
     [*FED_TWO, Q, Q, post(4, answer("a"))], ["c1", "c2"], {"c1": ([], 1), "c2": ([], 0)}),
    ("a-fed-case-with-nothing-out-is-fed-again", [*FED_TWO], ["c1", "c2"],
     {"c1": ([], 0), "c2": ([], 0)}),
    ("a-feed-out-restarts-its-case-and-a-reset-acknowledged-waits-for-its-feed",
     [ING, ING, post(0), ING], ["c1"], {"c1": ([], 0), "c2": ([], 0)}),
]


@pytest.mark.parametrize(("script", "restarted", "owed"), [row[1:] for row in LOST],
                         ids=[row[0] for row in LOST])
def test_a_runner_beginning_loses_what_was_outstanding(script, restarted, owed):
    _, before = play(script)
    state, ids = machine.begin(before)
    assert ids == restarted and all(case.outstanding == [] for case in state.cases)
    assert state.restarts == dict.fromkeys(restarted, 1)
    for case_id, (retry, question) in owed.items():
        case = state.cases[state.case_ids.index(case_id)]
        assert (case.retry, case.question_index) == (retry, question)


def test_a_restarted_case_asks_its_lost_questions_after_its_new_feed():
    _, state = play([*FED_TWO, Q, Q, Q, post(5, answer("b")), BEGIN])
    handed, state = play([ING, post(0), ING, post(1), Q, Q], state)
    assert handed == [(R, "c1", None), (F, "c1", None), (A, "c1", 0), (A, "c1", 2)]
    recorded = {r.question_index: r.result.answer for r in state.recorded.values()
                if r.op is A and r.case_id == "c1"}
    assert recorded == {1: "b"}


def test_a_lost_step_posted_after_a_runner_begins_is_refused():
    _, state = play([*FED_TWO, Q, Q])
    lost = state.cases[0].outstanding[1].step_id
    state, _ = machine.begin(state)
    with pytest.raises(ProtocolViolation, match="no longer outstanding"):
        machine.submit(state, lost, answer("late"))


def test_in_order_a_lost_question_is_asked_again_first():
    _, state = play([*FED_TWO, QO, post(4, answer("a")), QO, BEGIN])
    handed, _ = play([ING, post(0), ING, post(1), QO], state)
    assert handed[-1] == (A, "c1", 1)


# -- a run saved by an earlier memrank ---------------------------------------------------------

#: (case, the saved case, what it reads as: outstanding question, next question, retry)
SAVED = [
    ("no-step-out", {"outstanding": None, "question_index": 1, "lane": 0}, (None, 1, [])),
    ("a-feed-out", {"stage": "feed", "outstanding": {"step_id": "s2", "op": "feed", "case": 0}},
     (None, 0, [])),
    ("an-ask-out", {"stage": "ask", "question_index": 2,
                    "outstanding": {"step_id": "s5", "op": "ask", "case": 0}}, (2, 3, [])),
    ("an-ask-out-in-a-retry", {"stage": "ask", "question_index": 1, "retry": [1, 3],
                               "outstanding": {"step_id": "s9", "op": "ask", "case": 0}},
     (1, 3, [3])),
    ("the-last-retry-ask-out", {"stage": "ask", "question_index": 3, "retry": [3],
                                "outstanding": {"step_id": "s9", "op": "ask", "case": 0}},
     (3, 4, [])),
]


@pytest.mark.parametrize(("saved", "expected"), [row[1:] for row in SAVED],
                         ids=[row[0] for row in SAVED])
def test_a_case_saved_with_one_step_outstanding_reads_as_the_list(saved, expected):
    state = RunState.model_validate({"case_ids": ["c1"], "question_counts": [4],
                                     "cases": [saved], "issued": 9})
    case = state.cases[0]
    asked = [out.question_index for out in case.outstanding]
    assert (asked[0] if asked else None, case.question_index, case.retry) == expected
    assert len(case.outstanding) == (saved["outstanding"] is not None)


def test_a_saved_lost_ask_restarts_at_its_question_when_a_runner_begins():
    state = RunState.model_validate({"case_ids": ["c1"], "question_counts": [4], "issued": 5,
                                     "cases": [{"stage": "ask", "question_index": 2, "lane": 0,
                                                "outstanding": {"step_id": "s5", "op": "ask",
                                                                "case": 0}}]})
    state, restarted = machine.begin(state)
    assert restarted == ["c1"]
    assert (state.cases[0].stage, state.cases[0].question_index, state.cases[0].retry) == \
        (Op.RESET, 2, [])


# -- every transition leaves the state it was given as it was ----------------------------------

#: (case, the script that reaches the state, the transition applied to it)
TRANSITIONS = [
    ("next-ingest", [], lambda s: machine.next_step(s, Pool.INGEST)),
    ("next-answer", FED_TWO, lambda s: machine.next_step(s, Pool.ANSWER)),
    ("submit", [*FED_TWO, Q], lambda s: machine.submit(s, s.cases[0].outstanding[0].step_id,
                                                       answer("a"))),
    ("begin", [*FED_TWO, Q, Q], machine.begin),
    ("reopen-failed", [ING, post(0), ING, post(1, ERROR)], machine.reopen_failed),
]


@pytest.mark.parametrize(("script", "transition"), [row[1:] for row in TRANSITIONS],
                         ids=[row[0] for row in TRANSITIONS])
def test_a_transition_never_changes_the_state_it_was_given(script, transition):
    _, state = play(script)
    before = state.model_dump()
    after, _ = transition(state)
    assert state.model_dump() == before and after.model_dump() != before
