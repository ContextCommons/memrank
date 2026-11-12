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
"""Shipping a live run's event log to the organisation as it grows.

The run's ``events.jsonl`` (:class:`~memrank.loop.event_log.EventLog`) is the outbox; this is
the relay that empties it into ``POST /orgs/{org}/runs/{id}/events``
(:mod:`memrank.service.live_events` has the contract). Event-driven, with no timer and no batch
size (decision D8): each append wakes the uploader; when no send is in flight it sends every
event the server lacks, and events appended while a send is in flight ride the next one, split
only at :data:`~memrank.service.live_events.MAX_UPLOAD_BYTES`. One send at a time, on a worker
thread of its own, so neither a lane nor a judge worker ever waits on the network.

The server's cursor (``through_seq``) is where sending starts: from the registration's ``live``
answer, and from each acknowledgement after. A resumed run therefore sends only what the server
lacks. A server that holds more than the folder has is refused: resume is same-machine.

**An unreachable or failing server never stops the run.** The failure is said once, the next
append tries the same range again, and :meth:`Uploader.drain` -- before the final upload --
sends whatever is left, retrying a connection that was never made, and fails loudly if it
cannot. A refusal that sending again cannot cure (the range contradicts the rows, another
process is uploading the run) is said at once and raised by the drain. A server-directed wait
(``retry_after_s``) is honoured once before the next send. A sealed run takes no more events
and needs none: its sealed answers are its results.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx

from memrank import config
from memrank.errors import MemrankError
from memrank.loop.client import retry_unreached
from memrank.loop.event_log import EventLog
from memrank.placement import run_api_client
from memrank.placement.run_api_client import RunApiError
from memrank.service.live_events import MAX_UPLOAD_BYTES, LiveEndpoint, LiveEvent, UploadAck

#: The bytes of an upload around its events: ``{"from_seq":N,"to_seq":N,"events":[...]}``.
_ENVELOPE = '{{"from_seq":{first},"to_seq":{last},"events":[{events}]}}'

#: Refusals that sending the same range again cannot cure.
_FATAL = frozenset({"bad_upload", "upload_too_large", "seq_conflict"})

#: How a send is made: the encoded upload in, the API's acknowledgement out.
Send = Callable[[bytes], dict[str, Any]]


class LiveUnsupported(MemrankError):
    """The API registered the run but offers no live results, so this CLI cannot record it."""


class LiveAhead(MemrankError):
    """The server holds more of the run's events than its folder does."""


class EventsUnsent(MemrankError):
    """Some of the run's events could not be sent before its final upload."""


class _Unreached(MemrankError):
    """A send that may be repeated: the server was not reached, or failed on its own side."""


def endpoint_of(answer: dict[str, Any] | None) -> LiveEndpoint:
    """The registration answer's ``live``; :class:`LiveUnsupported` when it has none (D6)."""
    if not answer:
        raise LiveUnsupported(
            f"The memrank API at {config.memrank_api_url()} does not accept live run results, "
            "so this memrank cannot record runs there: it sends each answer and verdict as it "
            "happens. That API needs the live-results route "
            "(POST /orgs/{org}/runs/{id}/events). Upgrade the API, or use the memrank release "
            "that matches it.")
    return LiveEndpoint.model_validate(answer)


def encode(events: list[LiveEvent], cap: int = MAX_UPLOAD_BYTES) -> tuple[bytes, int]:
    """The first events of ``events`` that fit in ``cap`` bytes as one upload, and how many.

    Always at least one: an event alone over the cap is sent, and the API's ``413`` says so.
    """
    parts: list[str] = []
    size = len(_ENVELOPE.format(first=events[0].seq, last=events[-1].seq, events=""))
    for event in events:
        part = event.model_dump_json()
        size += len(part) + (1 if parts else 0)
        if parts and size > cap:
            break
        parts.append(part)
    body = _ENVELOPE.format(first=events[0].seq, last=events[len(parts) - 1].seq,
                            events=",".join(parts))
    return body.encode("utf-8"), len(parts)


@dataclass
class Uploader:
    """One run's relay from its event log to the API.

    Args:
        log: The run's event log.
        through_seq: How many events the server held when the run was registered.
        send: Posts one encoded upload (:func:`run_api_client.post_events` on the live url).
        echo: The runner's narration.
        sleep: The pause before a retry; a test's stand-in for time.
        cap: The byte ceiling of one upload.
    """

    log: EventLog
    through_seq: int
    send: Send
    echo: Callable[[str], None]
    sleep: Callable[[float], None] = time.sleep
    cap: int = MAX_UPLOAD_BYTES
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _worker: threading.Thread | None = field(default=None, repr=False)
    _failing: bool = False
    _fatal: MemrankError | None = None
    _sealed: bool = False
    #: A wait the server asked for, taken before the next send.
    _wait_s: float = 0.0
    #: A ``429`` was answered by waiting; another before a send gets through is a failure.
    _throttled: bool = False
    #: The drain has begun: it sends the rest itself, so no worker is started.
    _draining: bool = False

    def start(self) -> None:
        """Check the server's cursor against the log, then listen; send what the server lacks."""
        held = self.log.through_seq
        if self.through_seq > held:
            raise LiveAhead(
                f"The organisation holds {self.through_seq} of this run's events but its folder "
                f"{self.log.path.parent}/ has {held}. A run is resumed from the folder it was "
                "recorded in, on the machine that recorded it.")
        self.log.on_append(lambda seq: self.wake())
        self.wake()

    def wake(self) -> None:
        """Send on a worker thread unless a send is in flight; that one's next send takes
        whatever was appended meanwhile, since it looks again before it stops."""
        with self._lock:
            if self._worker is not None or self._draining or not self._sendable():
                return
            self._worker = threading.Thread(target=self._pump, name="memrank-upload",
                                            daemon=True)
            self._worker.start()

    def _sendable(self) -> bool:
        """Under the lock: something to send, and nothing that says not to send it."""
        return (self._fatal is None and not self._sealed
                and self.log.through_seq > self.through_seq)

    def _pump(self) -> None:
        """Send until the server holds the whole log or a send fails; then stop listening as
        the one in flight, deciding that under the lock so an append is never missed."""
        while True:
            with self._lock:
                if not self._sendable():
                    self._worker = None
                    return
            try:
                self._send_next()
            except _Unreached as exc:
                with self._lock:
                    self._worker = None
                    self._report(str(exc))
                return

    def settle(self) -> None:
        """Wait until no send is in flight."""
        with self._lock:
            worker = self._worker
        if worker is not None:
            worker.join()

    def drain(self) -> None:
        """Send every event the server lacks, before the final upload; raise if any is left.

        Waits for the send in flight, then sends the rest here, repeating one that did not get
        through as a call that never connected is repeated (:func:`retry_unreached`).
        """
        with self._lock:
            self._draining = True
        self.settle()
        try:
            while self._remaining():
                retry_unreached(self._send_next, unreached=_Unreached, sleep=self.sleep)
        except _Unreached as exc:
            raise EventsUnsent(f"the run's live results could not all be sent: {exc}") from exc
        if self._fatal is not None:
            raise EventsUnsent(f"the run's live results were refused: {self._fatal}")

    def _remaining(self) -> bool:
        with self._lock:
            return self._sendable()

    def _send_next(self) -> None:
        """One upload of the next events after the server's cursor, and its answer handled."""
        with self._lock:
            wait_s, self._wait_s = self._wait_s, 0.0
            first = self.through_seq
        if wait_s:
            self.sleep(wait_s)
        events = self.log.read_after(first)
        body, _ = encode(events, self.cap)
        try:
            answer = self.send(body)
        except RunApiError as exc:
            self._refused(exc, first)
            return
        except httpx.HTTPError as exc:
            raise _Unreached(str(exc) or type(exc).__name__) from exc
        ack = UploadAck.model_validate(answer)
        with self._lock:
            self.through_seq = max(self.through_seq, ack.through_seq)
            self._wait_s = ack.retry_after_s or 0.0
            self._throttled = False
            if self._failing:
                self.echo("live results are being sent again")
            self._failing = False

    def _refused(self, exc: RunApiError, first: int) -> None:
        """The API's refusal: wait when asked, resync on a gap, stop on what cannot be cured;
        anything else may pass and is sent again."""
        with self._lock:
            if exc.code == "too_many_uploads" and not self._throttled:
                self._throttled = True
                self._wait_s = float(exc.detail.get("retry_after_s") or 0.0)
                return  # sent again once, after the wait
            if exc.code == "run_sealed":
                self._sealed = True
                self.echo("live results: the organisation already holds this run's finished "
                          "results; no more events are sent")
                return
            held = exc.detail.get("through_seq")
            if exc.code == "seq_gap" and isinstance(held, int) and held != first:
                if held > self.log.through_seq:
                    self._fatal = LiveAhead(f"the organisation holds {held} of this run's "
                                            f"events, more than its folder's "
                                            f"{self.log.through_seq}")
                    self._report(str(self._fatal))
                else:
                    self.through_seq = held
                return
            if exc.code in _FATAL:
                self._fatal = exc
                self._report(str(exc))
                return
        raise _Unreached(str(exc)) from exc

    def _report(self, why: str) -> None:
        """Under the lock: say a failure once, until a send gets through again."""
        if self._fatal is not None and why == str(self._fatal):
            self.echo(f"live results were refused ({why}). The run goes on, and its final "
                      "upload will fail; the run is saved and can be resumed.")
            return
        if not self._failing:
            self.echo(f"live results could not be sent ({why}). The run goes on, and the "
                      "next answer or verdict tries again.")
        self._failing = True


def start_uploading(live: dict[str, Any] | None, log: EventLog, http: Any,
                    echo: Callable[[str], None],
                    sleep: Callable[[float], None] = time.sleep) -> Uploader:
    """The run's uploader, sending from the registration's cursor; refused without ``live``."""
    endpoint = endpoint_of(live)
    uploader = Uploader(log=log, through_seq=endpoint.through_seq, echo=echo, sleep=sleep,
                        send=lambda body: run_api_client.post_events(http, endpoint.url, body))
    uploader.start()
    return uploader
