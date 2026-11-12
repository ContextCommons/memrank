"""The run folder's append-only event log: numbering, stale verdicts, torn lines, listeners."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from memrank.loop.event_log import EVENTS_FILE, EventLog
from memrank.service.live_events import AnswerEvent, VerdictEvent
from memrank.service.protocol import QuestionOutcome

PASS = {"status": "judged", "score": 1.0, "passed": True, "rationale": "matches"}


def outcome(qid: str, answer: str = "blue") -> QuestionOutcome:
    return QuestionOutcome(case_id="c1", question_id=qid, question=f"{qid}?",
                           reference=["blue"], category=None, status="not_judged",
                           answer=answer, elapsed_ms=10.0)


def test_events_are_numbered_from_one_and_read_back_in_order(tmp_path: Path) -> None:
    log = EventLog(tmp_path)
    log.judging(None, judge_samples=0, judge_prompt_version=None)
    log.answer(position=0, answer_ref="s1", outcome=outcome("q0"))
    log.verdict(question_id="q0", answer_ref="s1", fields=PASS)
    assert log.through_seq == 3
    assert [(e.seq, e.kind) for e in log.read_after(0)] == [
        (1, "judging"), (2, "answer"), (3, "verdict")]
    assert [e.seq for e in log.read_after(2)] == [3]


def test_a_verdict_on_a_replaced_answer_is_not_logged(tmp_path: Path) -> None:
    log = EventLog(tmp_path)
    log.answer(position=0, answer_ref="s1", outcome=outcome("q0"))
    log.answer(position=0, answer_ref="s5", outcome=outcome("q0", "navy"))
    assert log.verdict(question_id="q0", answer_ref="s1", fields=PASS) is None
    assert log.verdict(question_id="q1", answer_ref="s2", fields=PASS) is None
    current = log.verdict(question_id="q0", answer_ref="s5", fields=PASS)
    assert isinstance(current, VerdictEvent) and current.seq == 3
    assert [e.kind for e in log.read_after(0)] == ["answer", "answer", "verdict"]


def test_reopening_resumes_numbering_and_current_answers(tmp_path: Path) -> None:
    first = EventLog(tmp_path)
    first.answer(position=0, answer_ref="s1", outcome=outcome("q0"))
    first.answer(position=0, answer_ref="s4", outcome=outcome("q0"))
    again = EventLog(tmp_path)
    assert again.through_seq == 2
    assert again.verdict(question_id="q0", answer_ref="s1", fields=PASS) is None
    assert again.verdict(question_id="q0", answer_ref="s4", fields=PASS) is not None
    assert again.through_seq == 3


def test_a_torn_last_line_is_ignored_and_cut_on_reopen(tmp_path: Path) -> None:
    log = EventLog(tmp_path)
    log.answer(position=0, answer_ref="s1", outcome=outcome("q0"))
    with (tmp_path / EVENTS_FILE).open("ab") as handle:
        handle.write(b'{"seq": 2, "kind": "ans')  # the process died mid-write
    assert [e.seq for e in log.read_after(0)] == [1]
    reopened = EventLog(tmp_path)
    assert reopened.through_seq == 1
    reopened.answer(position=1, answer_ref="s2", outcome=outcome("q1"))
    assert [e.seq for e in reopened.read_after(0)] == [1, 2]


def test_a_torn_line_before_the_end_is_refused(tmp_path: Path) -> None:
    log = EventLog(tmp_path)
    log.answer(position=0, answer_ref="s1", outcome=outcome("q0"))
    path = tmp_path / EVENTS_FILE
    path.write_bytes(b"not an event\n" + path.read_bytes())
    with pytest.raises(ValueError, match="line 1 is not an event"):
        log.read_after(0)
    with pytest.raises(ValueError, match="line 1 is not an event"):
        EventLog(tmp_path)


def test_listeners_hear_each_new_seq(tmp_path: Path) -> None:
    log = EventLog(tmp_path)
    heard: list[int] = []
    log.on_append(heard.append)
    log.answer(position=0, answer_ref="s1", outcome=outcome("q0"))
    log.verdict(question_id="q0", answer_ref="stale", fields=PASS)
    log.verdict(question_id="q0", answer_ref="s1", fields=PASS)
    assert heard == [1, 2]


def test_sixteen_threads_appending_leave_a_gap_free_readable_log(tmp_path: Path) -> None:
    log = EventLog(tmp_path)
    threads_n, per_thread = 16, 25
    start = threading.Barrier(threads_n)

    def lane(index: int) -> None:
        start.wait()
        for n in range(per_thread):
            log.answer(position=index * per_thread + n, answer_ref=f"s{index}-{n}",
                       outcome=outcome(f"q{index}-{n}"))

    threads = [threading.Thread(target=lane, args=(i,)) for i in range(threads_n)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    events = EventLog(tmp_path).read_after(0)
    assert [e.seq for e in events] == list(range(1, threads_n * per_thread + 1))
    assert all(isinstance(e, AnswerEvent) for e in events)
    assert len({e.question_id for e in events if isinstance(e, AnswerEvent)}) == len(events)
