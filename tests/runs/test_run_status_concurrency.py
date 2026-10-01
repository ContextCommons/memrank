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
"""The progress bar must never be able to kill a run.

`memrank submit --workers 10` evaluates units in a thread pool, and every worker publishes the
heartbeat per document and per query. With a fixed staging filename, two workers wrote the same
`status.json.tmp`, the first rename consumed it, and the second raised

    FileNotFoundError: '.../status.json.tmp' -> '.../status.json'

killing a LongMemEval run at 975/1232 documents -- from a *successful* progress update.

Contention here is structural, not timed: a `threading.Barrier` releases every worker at the same
instant and each does many writes, so the interleaving is forced rather than waited for.
"""
from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from memrank.runs import status as run_status

WORKERS = 8
WRITES_PER_WORKER = 40


@pytest.fixture
def status(tmp_path):
    return run_status.RunStatus.create(
        tmp_path, target="mem0:memory-arena-sdk", benchmark="longmemeval", slice_="mini")


def _run_workers(work):
    """Release WORKERS threads together; return every exception they raised."""
    barrier = threading.Barrier(WORKERS)
    failures: list[BaseException] = []

    def worker(index):
        barrier.wait()
        try:
            work(index)
        except BaseException as exc:  # noqa: BLE001 - the test's whole subject is what escapes
            failures.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(WORKERS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return failures


def test_concurrent_updates_never_raise(status, tmp_path):
    failures = _run_workers(lambda index: [
        status.update(state="retrieving", message=f"worker {index} write {n}")
        for n in range(WRITES_PER_WORKER)])

    assert failures == [], f"a heartbeat write raised: {failures[0]!r}"
    assert json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))["state"] == "retrieving"


def test_concurrent_progress_notes_never_raise(status, tmp_path):
    """The exact call path that failed: a heartbeat publish from every eval worker."""
    from memrank.evaluation.observer import EvalPlan
    from memrank.orchestration.observers import RunStatusObserver

    observer = RunStatusObserver(status, verbose=True)  # verbose: publish EVERY item
    observer.planned(EvalPlan(units=WORKERS, documents=WORKERS * WRITES_PER_WORKER,
                              retrievals=WORKERS, judgements=0))

    def work(index):
        for n in range(WRITES_PER_WORKER):
            observer.item_done("ingest", seconds=0.01, done=n + 1,
                               total=WRITES_PER_WORKER)

    assert _run_workers(work) == []
    record = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))["progress"]
    assert record["ingest"]["done"] == WORKERS * WRITES_PER_WORKER


def test_no_staging_files_survive_a_concurrent_burst(status, tmp_path):
    """Leftover staging files would be mistaken for run artifacts by later readers."""
    _run_workers(lambda index: [status.update(message=str(n)) for n in range(WRITES_PER_WORKER)])
    assert list(tmp_path.glob("*.tmp")) == []


class _ObservedLock:
    """Let a paused publisher proceed once its successor contends for the lock."""

    def __init__(self, contended):
        self._lock = threading.Lock()
        self._contended = contended

    def __enter__(self):
        if not self._lock.acquire(blocking=False):
            self._contended.set()
            self._lock.acquire()

    def __exit__(self, *exc):
        self._lock.release()


def _pause_first_publish(publish, stage, paused, resume):
    def wrapped(record, **kwargs):
        if record[stage]["done"] == 1:
            paused.set()
            resume.wait()
        publish(record, **kwargs)

    return wrapped


def _publish_overlapping_items(observer, stage, paused, resume):
    def newer_item():
        try:
            observer.item_done(stage, seconds=0.01, done=2, total=2)
        finally:
            # Without serialization the newer publication finishes first. With it,
            # lock contention releases the older publisher before this one can snapshot.
            resume.set()

    with ThreadPoolExecutor(max_workers=2) as pool:
        older = pool.submit(observer.item_done, stage, seconds=0.01, done=1, total=2)
        paused.wait()
        newer = pool.submit(newer_item)
        older.result()
        newer.result()


@pytest.mark.parametrize("stage", ["ingest", "retrieve", "judge"])
@pytest.mark.parametrize("pause_at", ["local", "remote"])
def test_older_progress_cannot_overwrite_completed_progress(
        status, tmp_path, monkeypatch, stage, pause_at):
    """Force an older snapshot to stall at each publication boundary, without sleeps."""
    from memrank.evaluation.observer import EvalPlan
    from memrank.orchestration.observers import RunStatusObserver

    observer = RunStatusObserver(status, verbose=True)
    observer.planned(EvalPlan(units=1, documents=2, retrievals=2, judgements=2))
    paused, resume = threading.Event(), threading.Event()
    monkeypatch.setattr(observer, "_publish_lock", _ObservedLock(resume), raising=False)
    remote_records = []
    monkeypatch.setattr(run_status, "publish_remote",
                        lambda record, **kwargs: remote_records.append(record))
    target, name = ((status, "set_progress_record") if pause_at == "local"
                    else (run_status, "publish_remote"))
    monkeypatch.setattr(target, name, _pause_first_publish(
        getattr(target, name), stage, paused, resume))

    _publish_overlapping_items(observer, stage, paused, resume)

    record = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))["progress"]
    assert record[stage]["done"] == 2
    assert [record[stage]["done"] for record in remote_records] == [1, 2]
    assert remote_records[-1] == record
