"""The live uploader: each append sends, appends during a send coalesce, the cap splits.

Driven with a fake ``send`` and real threads sequenced by events, never by time: a send that
must be in flight waits on a gate the test opens.
"""

from __future__ import annotations

import json
import threading

import httpx
import pytest

from memrank.loop.event_log import EventLog
from memrank.loop.uploader import (
    EventsUnsent,
    LiveAhead,
    LiveUnsupported,
    Uploader,
    encode,
    endpoint_of,
)
from memrank.placement.run_api_client import RunApiError


class Server:
    """A send that records each upload's range and answers as the API does when it takes it."""

    def __init__(self) -> None:
        self.ranges: list[tuple[int, int]] = []
        self.sizes: list[int] = []
        #: Refusals or failures to raise, in order, before taking uploads again.
        self.failures: list[Exception] = []
        self.gate: threading.Event | None = None
        self.sending = threading.Event()

    def __call__(self, body: bytes) -> dict:
        upload = json.loads(body)
        self.ranges.append((upload["from_seq"], upload["to_seq"]))
        self.sizes.append(len(body))
        self.sending.set()
        if self.gate is not None:
            self.gate.wait()
            self.gate = None
        if self.failures:
            raise self.failures.pop(0)
        return {"through_seq": upload["to_seq"], "state": "running"}


def judgings(log: EventLog, count: int) -> None:
    """Append ``count`` events; a model-free judging is the smallest one there is."""
    for _ in range(count):
        log.judging(None, judge_samples=0, judge_prompt_version=None)


def uploader(log: EventLog, server: Server, *, through_seq: int = 0, cap: int = 1_048_576,
             echoed: list[str] | None = None, slept: list[float] | None = None) -> Uploader:
    return Uploader(log=log, through_seq=through_seq, send=server, cap=cap,
                    echo=(echoed if echoed is not None else []).append,
                    sleep=(slept if slept is not None else []).append)


def test_events_appended_during_a_send_ride_the_next_one(tmp_path):
    log, server = EventLog(tmp_path), Server()
    server.gate = threading.Event()
    sender = uploader(log, server)
    sender.start()
    judgings(log, 1)
    server.sending.wait()  # the first event's send is in flight
    judgings(log, 3)
    server.gate.set()
    sender.settle()
    assert server.ranges == [(1, 1), (2, 4)]
    assert sender.through_seq == 4


def test_a_failed_send_is_retried_from_the_same_event_on_the_next_append(tmp_path):
    log, server, echoed = EventLog(tmp_path), Server(), []
    server.failures = [httpx.ConnectError("refused")]
    sender = uploader(log, server, echoed=echoed)
    sender.start()
    judgings(log, 1)
    sender.settle()
    judgings(log, 1)
    sender.settle()
    assert server.ranges == [(1, 1), (1, 2)]
    assert sender.through_seq == 2
    assert sum("could not be sent" in line for line in echoed) == 1
    assert echoed[-1] == "live results are being sent again"


def test_the_byte_cap_splits_a_backlog(tmp_path):
    log, server = EventLog(tmp_path), Server()
    judgings(log, 5)
    events = log.read_after(0)
    two, _ = encode(events[:2])
    uploader(log, server, cap=len(two)).drain()
    assert server.ranges == [(1, 2), (3, 4), (5, 5)]
    assert max(server.sizes) <= len(two)


def test_an_event_alone_over_the_cap_is_still_sent_for_the_api_to_refuse(tmp_path):
    log = EventLog(tmp_path)
    judgings(log, 2)
    body, count = encode(log.read_after(0), cap=1)
    assert count == 1 and json.loads(body)["to_seq"] == 1


def test_a_429_is_waited_out_once_then_sent_again(tmp_path):
    log, server, slept = EventLog(tmp_path), Server(), []
    judgings(log, 1)
    server.failures = [_refusal(429, "too_many_uploads", retry_after_s=7.0)]
    uploader(log, server, slept=slept).drain()
    assert slept == [7.0]
    assert server.ranges == [(1, 1), (1, 1)]


def test_a_second_429_is_a_failure_and_the_drain_says_what_is_unsent(tmp_path):
    log, server, slept = EventLog(tmp_path), Server(), []
    judgings(log, 1)
    server.failures = [_refusal(429, "too_many_uploads", retry_after_s=7.0)] * 10
    with pytest.raises(EventsUnsent, match="could not all be sent"):
        uploader(log, server, slept=slept).drain()
    assert slept.count(7.0) == 1


def test_a_resume_sends_only_what_the_server_lacks(tmp_path):
    log, server = EventLog(tmp_path), Server()
    judgings(log, 5)
    sender = uploader(log, server, through_seq=3)
    sender.start()
    sender.settle()
    assert server.ranges == [(4, 5)]


def test_a_server_ahead_of_the_folder_is_refused(tmp_path):
    log = EventLog(tmp_path)
    judgings(log, 2)
    with pytest.raises(LiveAhead, match="holds 3 of this run's events but its folder"):
        uploader(log, Server(), through_seq=3).start()


def test_a_gap_moves_the_cursor_to_the_servers(tmp_path):
    log, server = EventLog(tmp_path), Server()
    judgings(log, 4)
    server.failures = [_refusal(409, "seq_gap", through_seq=2)]
    uploader(log, server, through_seq=0).drain()
    assert server.ranges == [(1, 4), (3, 4)]


def test_a_refusal_sending_again_cannot_cure_is_said_at_once_and_raised_by_the_drain(tmp_path):
    log, server, echoed = EventLog(tmp_path), Server(), []
    server.failures = [_refusal(422, "bad_upload")]
    sender = uploader(log, server, echoed=echoed)
    sender.start()
    judgings(log, 2)
    sender.settle()
    with pytest.raises(EventsUnsent, match="refused"):
        sender.drain()
    assert any("were refused" in line for line in echoed)


def test_a_sealed_run_takes_no_more_events_and_needs_none(tmp_path):
    log, server = EventLog(tmp_path), Server()
    judgings(log, 2)
    server.failures = [_refusal(409, "run_sealed")]
    uploader(log, server).drain()
    assert server.ranges == [(1, 2)]


def test_a_registration_without_live_names_the_api_it_needs():
    with pytest.raises(LiveUnsupported, match=r"POST /orgs/\{org\}/runs/\{id\}/events"):
        endpoint_of(None)


def _refusal(status: int, code: str, **detail) -> RunApiError:
    return RunApiError(code, code=code, status=status,
                       detail={"code": code, "message": code, **detail})
