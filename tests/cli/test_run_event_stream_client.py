# Copyright 2026 AtomicStrata
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.
"""The terminal reading a run's event stream, and falling back when there is none.

Two properties are worth pinning. The parser must read SSE as SSE -- a keepalive comment is not an
event, and a frame this build cannot parse is not fatal. And a `watch` against a deployment with
no event transport must be indistinguishable from a `watch` before this existed: the stream is an
optimisation over the poll, never a replacement for it, so every way it can fail has to end in
the poll rather than in an error.
"""
from __future__ import annotations

import json

import pytest

from memrank.cli import watch as watch_cli
from memrank.placement import run_api_client


class _Stream:
    """An httpx streaming response, as `stream_run_events` uses one."""

    def __init__(self, lines: list[str], status_code: int = 200) -> None:
        self._lines = lines
        self.status_code = status_code
        # A refusal is JSON; a stream is not. `_refusal` reads the content type to decide whether
        # to look for a `detail`, so the fake has to carry one or it tests a path production
        # never takes.
        self.headers = {"content-type": "application/json"} if status_code >= 400 \
            else {"content-type": "text/event-stream"}
        self.text = "no transport" if status_code >= 400 else ""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def iter_lines(self):
        yield from self._lines

    def read(self):
        return b""

    def json(self):
        return {"detail": {"code": "events_not_configured", "message": "no transport"}}


class _Http:
    def __init__(self, stream: _Stream) -> None:
        self._stream = stream
        self.requested: list[tuple] = []

    def stream(self, method, url, **kwargs):
        self.requested.append((method, url, kwargs))
        return self._stream


def _sse(*events: tuple[str, dict]) -> list[str]:
    lines: list[str] = []
    for kind, payload in events:
        lines += [f"event: {kind}", f"data: {json.dumps(payload)}", ""]
    return lines


RECORD = {"stage": "ingest", "unit": 1, "units": 4, "pct": 0.25}


# -- the parser --------------------------------------------------------------------------

def test_each_frame_is_delivered_with_its_type_and_payload():
    http = _Http(_Stream(_sse(("run.progressed", {"progress": RECORD, "sequence": 1}))))
    seen: list[tuple] = []

    run_api_client.stream_run_events(http, "acme", "r-1",
                                     lambda kind, payload: seen.append((kind, payload)) or True)

    assert seen == [("run.progressed", {"progress": RECORD, "sequence": 1})]


def test_a_keepalive_comment_is_not_an_event():
    """The server sends these so a proxy does not close an idle connection. A reader that
    mistook one for an event would hand the caller a frame with no type at all."""
    http = _Http(_Stream([": keepalive", ""] +
                         _sse(("run.progressed", {"progress": RECORD, "sequence": 1}))))
    seen: list[tuple] = []

    run_api_client.stream_run_events(http, "acme", "r-1",
                                     lambda kind, payload: seen.append((kind, payload)) or True)

    assert [kind for kind, _ in seen] == ["run.progressed"]


def test_a_frame_that_is_not_json_is_skipped_rather_than_fatal():
    http = _Http(_Stream(["event: run.progressed", "data: {not json", ""] +
                         _sse(("run.finished", {"succeeded": True}))))
    seen: list[str] = []

    run_api_client.stream_run_events(http, "acme", "r-1",
                                     lambda kind, _payload: seen.append(kind) or True)

    assert seen == ["run.finished"]


def test_the_caller_decides_when_to_stop_reading():
    """A watch that has seen its run finish should be able to say so, rather than having to
    close a socket from underneath its own iterator."""
    http = _Http(_Stream(_sse(("run.progressed", {"sequence": 1}),
                              ("run.progressed", {"sequence": 2}),
                              ("run.progressed", {"sequence": 3}))))
    seen: list[dict] = []

    run_api_client.stream_run_events(http, "acme", "r-1",
                                     lambda _kind, payload: bool(seen.append(payload)) or
                                     len(seen) < 2)

    assert [p["sequence"] for p in seen] == [1, 2]


def test_a_deployment_with_no_transport_is_a_refusal_the_caller_can_read():
    http = _Http(_Stream([], status_code=503))

    with pytest.raises(run_api_client.RunApiError) as raised:
        run_api_client.stream_run_events(http, "acme", "r-1", lambda *_: True)

    assert raised.value.code == "events_not_configured"


# -- the watch's use of it ---------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clean_live():
    """The live cache is module state shared by the streams; a leak across tests would make one
    test's sample another's assertion."""
    watch_cli._LIVE.clear()
    yield
    watch_cli._LIVE.clear()


def test_a_progress_event_becomes_the_runs_live_record():
    http = _Http(_Stream(_sse(("run.progressed", {"progress": RECORD, "sequence": 1}))))
    stream = watch_cli._EventStream(http, "acme", "r-1")

    stream._run()

    assert watch_cli._live_progress("r-1") == RECORD


def test_other_events_do_not_disturb_the_bar():
    """`run.submitted` and `run.finished` come down the same stream; only progress is a record."""
    http = _Http(_Stream(_sse(("run.finished", {"succeeded": True}))))
    stream = watch_cli._EventStream(http, "acme", "r-1")

    stream._run()

    assert watch_cli._live_progress("r-1") is None


def test_a_stream_that_cannot_be_opened_leaves_the_watch_alone():
    """The whole fallback story in one assertion: no transport, no exception, no bar -- and the
    poll path below it still draws one."""
    http = _Http(_Stream([], status_code=503))
    stream = watch_cli._EventStream(http, "acme", "r-1")

    stream._run()  # must not raise

    assert watch_cli._live_progress("r-1") is None


def test_a_stream_that_dies_mid_read_leaves_what_it_had():
    """A dropped connection is the ordinary end of a long watch. What already arrived is still
    the newest thing anybody knows."""
    class _Dies(_Stream):
        def iter_lines(self):
            yield from _sse(("run.progressed", {"progress": RECORD, "sequence": 1}))
            raise ConnectionError("reset by peer")

    stream = watch_cli._EventStream(_Http(_Dies([])), "acme", "r-1")

    stream._run()  # must not raise

    assert watch_cli._live_progress("r-1") == RECORD


def test_a_stopped_stream_stops_reading():
    http = _Http(_Stream(_sse(("run.progressed", {"progress": RECORD, "sequence": 1}),
                              ("run.progressed", {"progress": {"pct": 0.9}, "sequence": 2}))))
    stream = watch_cli._EventStream(http, "acme", "r-1")
    stream.stop()

    stream._run()

    # The first frame is taken -- the reader only learns it should stop by being asked after one --
    # and the second is not.
    assert watch_cli._live_progress("r-1") == RECORD
