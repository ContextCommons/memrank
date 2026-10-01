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
"""Getting a cloud run's progress out of the container.

The task IS memrank -- the ECS command is `memrank <args> --on none ... && python upload.py` -- so it
has always known its own progress and had nowhere to put it. Five minutes into a live run the
terminal showed `progress: {}` and `message: "submitted via API"`.

It goes to S3 because that is the credential the task already holds: `s3:PutObject` on
`cloud-runs/*`. That path is the FLOOR and is unconditional: it works with no API reachable, no
event transport wired, and against a server that has never heard of the ingest route.

Beside it there is now a live path -- a POST to `memrank/api/run_progress.py` carrying the same
record, authenticated by a per-run token the launch injected. The task still holds no bus
credential and no user token; what it holds names one run, expires, and reaches one route.
"""
from __future__ import annotations

import json

import pytest

from memrank.evaluation.observer import EvalPlan
from memrank.orchestration.observers import RunStatusObserver
from memrank.runs import artifacts
from memrank.runs import status as run_status


class _S3:
    """Records what would have been written."""

    def __init__(self, fail: bool = False) -> None:
        self.puts: list[dict] = []
        self.fail = fail

    def put_object(self, **kwargs):
        if self.fail:
            raise RuntimeError("AccessDenied")
        self.puts.append(kwargs)


@pytest.fixture
def run(tmp_path, monkeypatch):
    """A registered run with a work model, observed as the runner observes one."""
    status = run_status.RunStatus.create(tmp_path, target="hindsight", benchmark="locomo",
                                         slice_=None)
    observer = RunStatusObserver(status)
    observer.planned(EvalPlan(units=10, documents=272, retrievals=4620, judgements=0))
    monkeypatch.setattr(run_status, "_last_published", 0.0)
    monkeypatch.setattr(run_status, "_interval_s", run_status.REMOTE_PUBLISH_INTERVAL_S)
    monkeypatch.setattr(run_status, "_sequence", 0)
    monkeypatch.setattr(run_status, "_reporting", True)
    return status, observer


class _Posted:
    """Stands in for `httpx.post`, recording calls and answering what the server would."""

    def __init__(self, status_code: int = 200, body: dict | None = None) -> None:
        self.calls: list[dict] = []
        self._status = status_code
        self._body = body if body is not None else {"state": "running", "interval_s": 10.0}

    def __call__(self, url, *, timeout, headers, json):
        self.calls.append({"url": url, "headers": headers, "json": json})
        return _Answer(self._status, self._body)


class _Answer:
    def __init__(self, status_code: int, body: dict) -> None:
        self.status_code = status_code
        self._body = body
        self.text = ""

    def json(self) -> dict:
        return self._body


@pytest.fixture
def posts(monkeypatch):
    """Intercept the live POST. `httpx` is imported inside `_post_progress`, so the patch lands
    on the module attribute the import resolves to."""
    import httpx

    posted = _Posted()
    monkeypatch.setattr(httpx, "post", posted)
    return posted


@pytest.fixture
def reporting(monkeypatch):
    """The environment a launched cloud task actually has."""
    monkeypatch.setenv("MEMRANK_PROGRESS_BUCKET", "bench-artifacts")
    monkeypatch.setenv("MEMRANK_PROGRESS_URL", "https://api.example.com/runs/r1/progress")
    monkeypatch.setenv("MEMRANK_RUN_TOKEN", "mrr_token.sig")


def test_progress_lands_under_the_prefix_the_task_may_write(run):
    """`cloud-runs/*` -- the one grant the ECS task role holds. Anywhere else is an AccessDenied
    the run would discover only in production."""
    s3 = _S3()

    artifacts.put_progress(bucket="b", run_id="r1", record={"pct": 24}, client=s3)

    assert s3.puts[0]["Key"] == "cloud-runs/r1/progress.json"
    assert json.loads(s3.puts[0]["Body"])["pct"] == 24


def test_nothing_is_published_when_the_run_is_not_a_cloud_task(run, monkeypatch):
    """A local run's heartbeat is already on the machine that will read it. Publishing would be a
    billed write for a file nobody fetches."""
    _status, observer = run
    monkeypatch.delenv("MEMRANK_PROGRESS_BUCKET", raising=False)
    calls = []
    monkeypatch.setattr(artifacts, "put_progress", lambda **kw: calls.append(kw))

    observer.stage_started("ingest")

    assert not calls


def test_a_cloud_task_publishes_what_it_recorded(run, monkeypatch):
    _status, observer = run
    monkeypatch.setenv("MEMRANK_PROGRESS_BUCKET", "bench-artifacts")
    calls = []
    monkeypatch.setattr(artifacts, "put_progress", lambda **kw: calls.append(kw))

    observer.item_done("ingest", seconds=5.16, done=1, total=1)

    assert calls[0]["bucket"] == "bench-artifacts"
    assert calls[0]["record"]["ingest"]["done"] == 1


def test_publishing_is_throttled(run, monkeypatch):
    """Every write is billed and the watcher polls at 15s, so anything finer buys nothing a
    viewer could see."""
    _status, observer = run
    monkeypatch.setenv("MEMRANK_PROGRESS_BUCKET", "b")
    calls = []
    monkeypatch.setattr(artifacts, "put_progress", lambda **kw: calls.append(kw))

    for _ in range(50):
        observer.stage_started("ingest")

    assert len(calls) == 1


def test_a_failed_publish_never_fails_the_run(run, monkeypatch):
    """The one deliberate swallow in this path. Losing an update costs a stale bar for ten
    seconds; raising would discard a run that has done real work -- which for a full LoCoMo cell
    is roughly forty-five minutes and a live engine's worth of it."""
    _status, observer = run
    monkeypatch.setenv("MEMRANK_PROGRESS_BUCKET", "b")

    def deny(**kwargs):
        raise RuntimeError("AccessDenied")

    monkeypatch.setattr(artifacts, "put_progress", deny)

    observer.stage_started("ingest")  # must not raise


def test_the_task_is_told_where_to_publish(monkeypatch):
    """As an environment prefix, not a flag -- deliberately not a REMOTE_CLI_CONTRACT bump, since
    an unknown environment variable cannot make an older image fail to parse its command."""
    from memrank.placement.cloud_submit import remote_command

    command = remote_command(["submit", "hindsight", "locomo"], bucket="bench", run_id="r1")

    assert command.startswith("MEMRANK_PROGRESS_BUCKET=bench memrank ")


def test_an_older_image_is_unobserved_not_degraded():
    """It ignores the variable and evaluates identically. The distinction matters: a run that
    produces the same numbers with no progress bar is fine; one that produces different numbers
    would not be, and is what "no fallback modes" is about.

    The pin guards the PRINCIPLE, not the number: the env prefix did not warrant a bump (2), and
    the bumps since came from real argv-format changes -- 3 retired tier/slice from the rendered
    command, 4 retired `--max-judge-calls` with the judge cap, 5 retired `--ack-egress` with the
    egress acknowledgement, 6 made judging derived so the command now states `--judge` or
    `--no-judge` rather than relying on a switch's absence. Whoever bumps it next must have
    changed what `remote_command` emits."""
    from memrank.placement.cloud_submit import REMOTE_CLI_CONTRACT

    assert REMOTE_CLI_CONTRACT == 6, (
        "bumping this invalidates every published image; only a change to the rendered "
        "command's format warrants it -- an env prefix does not")


# -- the live path, beside the floor -----------------------------------------------------

def test_a_reporting_task_posts_the_same_record_it_wrote_to_s3(run, monkeypatch, posts,
                                                               reporting):
    """One record, two destinations. Two renderings of a progress record is how a bar in the
    browser and a bar in a terminal come to disagree about the same run."""
    _status, observer = run
    puts = []
    monkeypatch.setattr(artifacts, "put_progress", lambda **kw: puts.append(kw))

    observer.item_done("ingest", seconds=5.16, done=1, total=1)

    assert posts.calls[0]["json"]["progress"] == puts[0]["record"]


def test_the_post_carries_the_run_token_and_names_no_run(run, posts, reporting, monkeypatch):
    """Identity comes from the token; there is nothing in the body for a server to believe."""
    _status, observer = run
    monkeypatch.setattr(artifacts, "put_progress", lambda **kw: None)

    observer.stage_started("ingest")

    assert posts.calls[0]["headers"]["Authorization"] == "Bearer mrr_token.sig"
    assert set(posts.calls[0]["json"]) == {"progress", "sequence"}


def test_samples_are_numbered_so_a_late_one_can_be_dropped(run, posts, reporting, monkeypatch):
    _status, observer = run
    monkeypatch.setattr(artifacts, "put_progress", lambda **kw: None)

    for _ in range(3):
        monkeypatch.setattr(run_status, "_last_published", 0.0)
        observer.stage_started("ingest")

    assert [c["json"]["sequence"] for c in posts.calls] == [1, 2, 3]


def test_a_task_with_no_endpoint_configured_posts_nothing(run, posts, monkeypatch):
    """An older server, or a deployment with no public URL. The S3 write still happens."""
    _status, observer = run
    monkeypatch.setenv("MEMRANK_PROGRESS_BUCKET", "b")
    monkeypatch.delenv("MEMRANK_PROGRESS_URL", raising=False)
    monkeypatch.delenv("MEMRANK_RUN_TOKEN", raising=False)
    puts = []
    monkeypatch.setattr(artifacts, "put_progress", lambda **kw: puts.append(kw))

    observer.stage_started("ingest")

    assert posts.calls == []
    assert len(puts) == 1


def test_a_refused_post_never_fails_the_run(run, monkeypatch, reporting):
    """Same swallow, same reason as the S3 write above."""
    _status, observer = run
    import httpx

    monkeypatch.setattr(artifacts, "put_progress", lambda **kw: None)
    monkeypatch.setattr(httpx, "post", _Posted(status_code=401, body={}))

    observer.stage_started("ingest")  # must not raise


def test_a_post_that_raises_never_fails_the_run(run, monkeypatch, reporting):
    """The API being unreachable is the ordinary case during a deploy, not an exceptional one."""
    _status, observer = run
    import httpx

    monkeypatch.setattr(artifacts, "put_progress", lambda **kw: None)

    def refuse(*args, **kwargs):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(httpx, "post", refuse)

    observer.stage_started("ingest")  # must not raise


def test_the_task_stops_posting_once_the_server_says_the_run_is_over(run, monkeypatch,
                                                                     reporting):
    """The control channel's whole point: a killed run learns it over the connection it was
    already making, rather than by being told nothing until ECS stops it."""
    import httpx

    _status, observer = run
    monkeypatch.setattr(artifacts, "put_progress", lambda **kw: None)
    posted = _Posted(body={"state": "stopped-failed", "interval_s": 0.0})
    monkeypatch.setattr(httpx, "post", posted)

    for _ in range(3):
        monkeypatch.setattr(run_status, "_last_published", 0.0)
        observer.stage_started("ingest")

    assert len(posted.calls) == 1


def test_the_server_can_slow_a_task_down(run, monkeypatch, reporting):
    """A throttle that changes without shipping a new harness."""
    _status, observer = run
    import httpx

    monkeypatch.setattr(artifacts, "put_progress", lambda **kw: None)
    monkeypatch.setattr(httpx, "post", _Posted(body={"state": "running", "interval_s": 60.0}))

    observer.stage_started("ingest")

    assert run_status._interval_s == 60.0


def test_the_server_cannot_speed_a_task_below_its_own_floor(run, monkeypatch, reporting):
    """A bad answer must not turn a progress channel into a request loop."""
    import httpx

    _status, observer = run
    monkeypatch.setattr(artifacts, "put_progress", lambda **kw: None)
    monkeypatch.setattr(httpx, "post", _Posted(body={"state": "running", "interval_s": 0.001}))

    observer.stage_started("ingest")

    assert run_status._interval_s == run_status.REMOTE_PUBLISH_INTERVAL_S
