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
"""A run online from its first step: registered, reporting progress, and marked if it stops.

``memrank run`` registers the run in the organisation's history as ``running`` before any step,
so its link is printed at the start. As steps complete it reports progress the way a cloud task
does -- ``POST /runs/{id}/progress`` with the run token the registration handed back, on the
interval the server answers with -- and the run page shows the same bars a cloud run's does. The
record is :class:`memrank.runs.progress.RunProgress`'s, with feeding as its ingest stage and
asking as its retrieve stage; the counts come from the service's own ``status``, so a resumed
run starts where it stopped rather than at zero. A run that stops short is recorded as
``stopped`` with its last progress, and ``--resume`` registers it as running again under the
same id and link. The finished run is uploaded by :func:`memrank.loop.upload.upload`.

**Failures are reported, never swallowed, and never fail the run.** A progress update that does
not get through is said once -- the bar online stands still, the run goes on, and the next update
tries again -- and said again only after one has got through. A progress update is a sample: the
next one supersedes it, and the final upload records everything. The registration and the final
upload are different: without the first there is no link, and without the last there is no
record, so both fail loudly (the latter naming ``--resume``).
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx

from memrank.errors import MemrankError
from memrank.loop.upload import Hosted, register
from memrank.runs.progress import RunProgress
from memrank.runs.status import REMOTE_PUBLISH_INTERVAL_S
from memrank.service.protocol import RunStatus

#: The server's answers that mean "keep reporting".
_LIVE = frozenset({"submitted", "running"})


@dataclass
class Channel:
    """Where progress goes, and how often: the run-token route and the server's interval."""

    url: str
    token: str
    interval_s: float = REMOTE_PUBLISH_INTERVAL_S


@dataclass
class LiveRun:
    """One run's online presence: registration, progress samples, and a stop if it stops.

    Every part that touches time or the network is injected -- ``clock`` and ``echo`` are the
    runner's :class:`~memrank.loop.run.Instruments`, ``status`` is the evaluator's -- so tests
    drive it with no wall clock and no server.
    """

    hosted: Hosted
    run_id: str
    agent: str
    evaluation: str
    status: Callable[[], RunStatus]
    clock: Callable[[], float]
    echo: Callable[[str], None]
    judged_run: bool
    url: str | None = None
    _channel: Channel | None = None
    _progress: RunProgress = field(init=False)
    _seen: RunStatus | None = None
    _last_sent: float = 0.0
    _sequence: int = 0
    _failing: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        self._progress = RunProgress(_now=self.clock)

    def open(self) -> str | None:
        """Register the run as running and return its page. Refused registration is raised."""
        with self._lock:
            self._tally(self.status())
            answer = register(self.hosted, self.run_id, self.agent, self.evaluation,
                              state="running", progress=self._progress.as_dict())
            self.url = answer.get("url")
            self._channel = self._channel_from(answer.get("progress"))
            self._last_sent = self.clock()
        return self.url

    def _channel_from(self, offered: dict[str, str] | None) -> Channel | None:
        if offered:
            return Channel(url=offered["url"], token=offered["token"])
        self.echo("live progress: this deployment offers no progress channel, so the run's "
                  "page shows it when it is uploaded")
        return None

    def step_done(self) -> None:
        """A step was recorded: send a sample when the server's interval has passed."""
        self._maybe_send(force=False)

    def judging(self, answers: int) -> None:
        """Judging begins over ``answers``; send a sample now, since the stage changes."""
        with self._lock:
            status = self.status()  # every step is done: the last sample may predate the last
            self._tally(status)
            self._plan(status, judgements=answers)
            self._progress.enter_stage("judge")
        self._maybe_send(force=True)

    def one_judged(self) -> None:
        """One verdict landed (called from the judge's workers)."""
        with self._lock:
            self._progress.record("judge")
        self._maybe_send(force=False)

    def stopped(self, reason: str) -> None:
        """Record the run as stopped short, with its last progress. Failure is reported, since
        the run is already ending on the error that brought it here; the run's ending shows its
        link and how to continue it."""
        try:
            with self._lock:
                self._tally(self.status())
                record = self._progress.as_dict()
            register(self.hosted, self.run_id, self.agent, self.evaluation, state="stopped",
                     progress=record, reason=reason)
        except MemrankError as exc:
            self.echo(f"the run could not be marked stopped online: {exc}")

    def _maybe_send(self, *, force: bool) -> None:
        with self._lock:
            channel = self._channel
            if channel is None or (not force and
                                   self.clock() - self._last_sent < channel.interval_s):
                return
            self._last_sent = self.clock()
            if self._progress.stage != "judge":
                self._tally(self.status())
            self._sequence += 1
            body = {"progress": self._progress.as_dict(), "sequence": self._sequence}
        self._send(channel, body)

    def _send(self, channel: Channel, body: dict[str, Any]) -> None:
        """POST one sample, outside the lock so no lane waits on the network; its outcome is
        handled under the lock. Any failure is reported (:meth:`_failed`), never raised."""
        answer: dict[str, Any] | None = None
        why = ""
        try:
            response = self.hosted.http.post(
                channel.url, json=body, headers={"Authorization": f"Bearer {channel.token}"})
            if response.is_error:
                why = f"the API answered {response.status_code}: {response.text[:300]}"
            else:
                answer = response.json()
        except (MemrankError, httpx.HTTPError, ValueError) as exc:
            why = str(exc) or type(exc).__name__
        with self._lock:
            if answer is None:
                self._failed(why)
            else:
                self._answered(channel, answer)

    def _failed(self, why: str) -> None:
        if not self._failing:
            self.echo(f"live progress could not be sent ({why}). The run goes on, the next "
                      "update tries again, and the final upload records everything.")
        self._failing = True

    def _answered(self, channel: Channel, answer: dict[str, Any]) -> None:
        if self._failing:
            self.echo("live progress is being sent again")
        self._failing = False
        if answer.get("state") not in _LIVE or float(answer.get("interval_s", 0)) <= 0:
            self.echo(f"live progress: the API says this run is {answer.get('state')}; "
                      "no more progress is sent")
            self._channel = None
            return
        # The server may slow a run down, never speed it past the harness's own floor.
        channel.interval_s = max(float(answer["interval_s"]), REMOTE_PUBLISH_INTERVAL_S)

    def _plan(self, status: RunStatus, judgements: int) -> None:
        self._progress.plan(units=status.cases, documents=status.cases,
                            retrievals=status.questions, judgements=judgements)

    def _tally(self, status: RunStatus) -> None:
        """Bring the record up to ``status``: feeding is ingest, asking is retrieve."""
        previous = self._seen
        if previous is None:
            self._plan(status, judgements=status.questions if self.judged_run else 0)
            previous = status.model_copy(update={"cases_fed": 0, "fed_ms": 0.0,
                                                 "questions_done": 0, "asked_ms": 0.0})
        fed = status.cases_fed - previous.cases_fed
        asked = status.questions_done - previous.questions_done
        if fed > 0:
            self._progress.record("ingest", items=fed,
                                  seconds=(status.fed_ms - previous.fed_ms) / 1000)
        if asked > 0:
            self._progress.record("retrieve", items=asked,
                                  seconds=(status.asked_ms - previous.asked_ms) / 1000)
        self._progress.enter_unit(min(status.cases, status.cases_finished + 1))
        self._seen = status
