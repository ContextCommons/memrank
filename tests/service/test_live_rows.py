"""The live-results contract: events fold into rows, and rows map one-to-one to answers.json."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import TypeAdapter, ValidationError

from memrank.service.live_events import AnswerEvent, LiveEvent, Upload
from memrank.service.live_rows import (
    LiveAnswers,
    LiveRows,
    apply,
    cases_of,
    judging_row,
    rows_of,
)
from memrank.service.protocol import CaseOutcome, JudgeIdentity, RunResult
from memrank.service.scoring import apply_verdicts, provisional, summarize

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "live"
RUNS = ("judged", "model-free")
_EVENTS = TypeAdapter(list[LiveEvent])
_CASES = TypeAdapter(list[CaseOutcome])


def load(run: str, name: str) -> Any:
    return json.loads((FIXTURES / run / name).read_text())


def sealed(run: str) -> RunResult:
    """The run's record with its answers put back: what the runner uploaded."""
    summary = RunResult.model_validate(load(run, "record.json")["result"])
    return summary.model_copy(update={"cases": _CASES.validate_python(
        load(run, "answers.json")["cases"])})


@pytest.mark.parametrize("run", RUNS)
def test_folding_the_events_yields_the_committed_rows(run: str) -> None:
    events = _EVENTS.validate_python(load(run, "events.json"))
    assert apply(events) == LiveRows.model_validate(load(run, "rows.json"))


@pytest.mark.parametrize("run", RUNS)
def test_answers_json_maps_to_the_same_rows_the_events_fold_to(run: str) -> None:
    result = sealed(run)
    rows = rows_of(result.cases, result.identity, judged=result.judged)
    assert rows == LiveRows.model_validate(load(run, "rows.json"))


@pytest.mark.parametrize("run", RUNS)
def test_rows_round_trip_to_answers_json(run: str) -> None:
    result = sealed(run)
    summary = RunResult.model_validate(load(run, "record.json")["result"])
    rows = rows_of(result.cases, result.identity, judged=result.judged)
    assert cases_of(rows, summary.cases) == result.cases


def test_an_older_record_with_questions_inline_round_trips() -> None:
    result = RunResult.model_validate(load("inline", "record.json")["result"])
    rows = rows_of(result.cases, result.identity, judged=result.judged)
    assert rows.judging is not None and rows.judging.judge == "unknown:claude-haiku-4-5"
    assert cases_of(rows, result.cases) == result.cases


def test_an_unjudged_run_has_no_judging_and_scores_nothing() -> None:
    result = sealed("judged")
    bare = [case.model_copy(update={"questions": [
        q.model_copy(update={"status": "failed" if q.status == "failed" else "not_judged",
                             "score": None, "passed": None, "rationale": None,
                             "error": q.error if q.status == "failed" else None})
        for q in case.questions]}) for case in result.cases]
    rows = rows_of(bare, result.identity, judged=False)
    assert rows.judging is None and rows.verdicts == []
    assert cases_of(rows, result.cases) == bare
    assert provisional(rows, total=6).mean is None


def test_a_reanswered_question_is_ungraded_until_its_new_verdict() -> None:
    events = _EVENTS.validate_python(load("judged", "events.json"))
    before = apply(events[:3])  # c1_q0 answered and judged
    assert [v.question_id for v in before.verdicts] == ["c1_q0"]
    after = apply(events[3:4], before)  # c1_q0 answered again
    assert after.verdicts == [] and after.answers[0].answer == "a blue car"
    assert provisional(after, total=6).ungraded == 1


def test_a_live_run_without_its_record_reads_its_cases_as_pending() -> None:
    events = _EVENTS.validate_python(load("judged", "events.json"))
    cases = cases_of(apply(events[:5]))
    assert [(c.case_id, c.status, len(c.questions)) for c in cases] == [("c1", "pending", 2)]


@pytest.mark.parametrize("run", RUNS)
def test_provisional_equals_the_final_score_once_everything_is_graded(run: str) -> None:
    result = sealed(run)
    rows = rows_of(result.cases, result.identity, judged=True)
    live = provisional(rows, total=result.questions)
    assert live.mean == summarize(result).score.mean
    assert live.failed == result.failed and live.ungraded == 0


def test_provisional_matches_apply_verdicts_over_the_same_verdicts() -> None:
    result = sealed("model-free")
    recorded = result.model_copy(update={"judged": False, "cases": [
        case.model_copy(update={"questions": [q.model_copy(update={
            "status": "not_judged", "score": None, "passed": None, "rationale": None})
            for q in case.questions]}) for case in result.cases]})
    verdicts = {q.question_id: q.model_dump() for c in result.cases for q in c.questions}
    final = apply_verdicts(recorded, verdicts, judge=None, judge_samples=0)
    rows = apply(_EVENTS.validate_python(load("model-free", "events.json")))
    live = provisional(rows, total=final.questions)
    assert (live.graded, live.total) == (2, 2)
    assert live.mean == final.score.mean == 0.5


def test_provisional_counts_failures_as_graded_zeros() -> None:
    rows = apply(_EVENTS.validate_python(load("judged", "events.json")))
    live = provisional(rows, total=6)
    # c1_q0 1, c1_q1 0, c2 two failures 0, c3_q1 1; c3_q0 unjudged is left out
    assert (live.graded, live.failed, live.ungraded) == (5, 2, 0)
    assert live.mean == pytest.approx(2 / 5)


def test_a_verdict_needs_a_judging_and_an_answer() -> None:
    events = _EVENTS.validate_python(load("judged", "events.json"))
    with pytest.raises(ValueError, match="precedes the run's judging"):
        apply(events[1:3])
    with pytest.raises(ValueError, match="no answer to grade"):
        apply([events[0], events[2]])


def test_a_second_different_judging_is_refused() -> None:
    judged = _EVENTS.validate_python(load("judged", "events.json"))[0]
    free = _EVENTS.validate_python(load("model-free", "events.json"))[0]
    assert apply([judged, judged]).judging is not None  # a resume logs it again
    with pytest.raises(ValueError, match="different one"):
        apply([judged, free])


def test_an_upload_carries_one_contiguous_range() -> None:
    events = load("judged", "events.json")
    assert len(Upload.model_validate(
        {"from_seq": 2, "to_seq": 4, "events": events[1:4]}).events) == 3
    with pytest.raises(ValidationError, match="each once"):
        Upload.model_validate({"from_seq": 2, "to_seq": 4, "events": [events[1], events[3]]})


def test_an_answer_event_carries_only_a_final_ungraded_answer() -> None:
    event = load("judged", "events.json")[1]
    AnswerEvent.model_validate(event)
    for status in ("judged", "pending"):
        with pytest.raises(ValidationError, match="answer event's outcome"):
            AnswerEvent.model_validate({**event, "outcome": {**event["outcome"],
                                                             "status": status}})


def test_the_mid_run_answers_body_is_the_rows_read_with_a_provisional_score() -> None:
    events = _EVENTS.validate_python(load("judged", "events.json"))
    rows = apply(events[:9])
    body = LiveAnswers(cases=cases_of(rows), judged=True,
                       provisional=provisional(rows, total=6))
    assert body == LiveAnswers.model_validate(load("judged", "live-answers.json"))


def test_a_judging_row_keeps_where_its_judge_identity_came_from() -> None:
    attributed = JudgeIdentity.unverified("claude-haiku-4-5")
    row = judging_row(attributed, judge_samples=1, judge_prompt_version=None)
    assert row.judge_source == "backfill:unverified"
    assert judging_row(None, judge_samples=0, judge_prompt_version=None).judge_source == "run"
