"""``memrank run`` sends its event log to the org as it grows, and refuses an API without it.

Through :func:`run_agent` against the fake org (:class:`tests.loop.conftest.OrgApi`), which
gates uploads on the next contiguous range as the API does.
"""

from __future__ import annotations

import json
import threading

import pytest

from memrank.loop.event_log import EventLog
from memrank.loop.run import RunOptions, UploadIncomplete, run_agent
from memrank.loop.uploader import LiveUnsupported
from memrank.service.protocol import Ack, StepResult
from tests.loop.conftest import OrgApi, run_id_of, stub_grading
from tests.loop.test_run import full_context, quiet

LOCOMO = RunOptions(evaluation="locomo")
#: A guard against a hang, never a timing assumption: each wait it bounds is asserted.
HANG_GUARD_S = 30.0


class Arriving(OrgApi):
    """The org, saying when the first answer event has arrived."""

    def __init__(self) -> None:
        super().__init__()
        self.answer_arrived = threading.Event()

    def _events(self, run_id, body):
        answered = any(event["kind"] == "answer" for event in body["events"])
        response = super()._events(run_id, body)
        if answered and response.status_code == 200:
            self.answer_arrived.set()
        return response


class LastStepWaits:
    """The engine, except that its last step is posted only once an answer is in the org: a
    runner that sent its events only at the end would never get past it."""

    def __init__(self, inner, arrived: threading.Event) -> None:
        self.inner, self.arrived = inner, arrived

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def post(self, run_id: str, step_id: str, result: StepResult) -> Ack:
        status = self.inner.status(run_id)
        if status.questions_done == status.questions - 1:
            assert self.arrived.wait(HANG_GUARD_S)
        return self.inner.post(run_id, step_id, result)


def logged(folder) -> list[dict]:
    return [json.loads(event.model_dump_json()) for event in EventLog(folder).read_after(0)]


def test_answers_reach_the_org_while_the_run_is_going(engine, tmp_path, monkeypatch):
    stub_grading(monkeypatch)
    org = Arriving()
    done = run_agent(LastStepWaits(engine, org.answer_arrived), full_context(), LOCOMO,
                     tmp_path, org.hosted(), quiet([]))
    assert org.events[done.result.run_id] == logged(done.folder)
    assert org.states == ["running", "done"]


def test_an_api_without_live_results_is_refused_before_the_first_step(engine, tmp_path,
                                                                     monkeypatch):
    stub_grading(monkeypatch)
    org, echoed = OrgApi(), []
    org.live_offered = False
    with pytest.raises(LiveUnsupported, match="does not accept live run results"):
        run_agent(engine, full_context(), LOCOMO, tmp_path, org.hosted(), quiet(echoed))
    assert org.states == ["running", "stopped"]
    assert "does not accept live run results" in org.syncs[-1]["error"]
    assert engine.status(run_id_of(echoed)).questions_done == 0


def test_an_org_that_is_briefly_down_misses_nothing(engine, tmp_path, monkeypatch):
    stub_grading(monkeypatch)
    org, echoed = OrgApi(), []
    org.events_down = 2
    done = run_agent(engine, full_context(), LOCOMO, tmp_path, org.hosted(), quiet(echoed))
    assert org.events[done.result.run_id] == logged(done.folder)
    assert sum("live results could not be sent" in line for line in echoed) == 1


def test_unsent_events_stop_the_upload_and_a_resume_sends_only_the_rest(engine, tmp_path,
                                                                       monkeypatch):
    stub_grading(monkeypatch)
    org, echoed = OrgApi(), []
    hosted = org.hosted()
    _first_through_then_down(org)  # one upload gets through, then the drain cannot
    with pytest.raises(UploadIncomplete, match="--resume"):
        run_agent(engine, full_context(), LOCOMO, tmp_path, hosted, quiet(echoed))
    run_id = run_id_of(echoed)
    held = len(org.events[run_id])
    assert 0 < held < len(logged(tmp_path / run_id))
    assert org.states == ["running", "stopped"]

    org.events_down, sent_before = 0, len(org.uploads_live)
    done = run_agent(engine, full_context(), RunOptions(resume=run_id), tmp_path, hosted,
                     quiet([]))
    assert org.uploads_live[sent_before][0] == held + 1
    assert org.events[run_id] == logged(done.folder)
    assert org.states[-1] == "done"


def _first_through_then_down(org: OrgApi) -> None:
    """Let exactly one upload through, then answer every later one 503."""
    real = org._events
    taken: list[bool] = []

    def events(run_id, body):
        if not taken:
            taken.append(True)
            org.events_down = 0
            response = real(run_id, body)
            org.events_down = 10**6
            return response
        return real(run_id, body)

    org._events = events
