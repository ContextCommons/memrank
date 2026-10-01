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
"""Fetching a run's artifacts concurrently must land exactly what fetching them serially landed.

A cloud run is no longer a handful of files: `deploy/upload_results.py` writes one detail JSON per
question per cell, so a locomo run uploads thousands of small artifacts whose cost is one round
trip each. Taken one at a time that IS the wait.

What makes the fan-out safe rather than merely fast is the shape: list (single-threaded) ->
download (concurrent, one artifact each, disjoint destinations) -> the listing's own order out.
`ex.map` and never `as_completed`, so two things do not depend on scheduling -- the paths returned,
and WHICH failure is the one the user is told about. A fetch that blames a different file on every
attempt is not one anybody can act on.

Nothing here is timed. Overlap is proven with a barrier, and every equality holds at every width.
"""
from __future__ import annotations

import json
import threading
from collections import Counter

import pytest

from memrank.placement import run_api_client
from memrank.runs import reconcile
from memrank.runs import status as run_status

WORKER_COUNTS = (1, 2, 4, 8)

#: A shape a real run produces: a big cell, the small top-level metadata beside it, its corpus,
#: and one detail file per question. The proportions are the measured ones -- a locomo smoke run
#: listed 158 artifacts of which 153 were `detail/` and `corpus/`.
def _entries(details: int = 12) -> list[dict]:
    return ([{"name": "myengine__locomo.json", "size": 83_491_477},
             {"name": "questions.json", "size": 160_000},
             {"name": "record.json", "size": 4_096},
             {"name": "summary__locomo.json", "size": 2_048},
             {"name": "corpus/myengine__locomo.json", "size": 512_000}]
            + [{"name": f"detail/myengine__locomo/q{i:04d}.json", "size": 20_000}
               for i in range(details)])


def _renderable(entries: list[dict]) -> list[dict]:
    """The artifacts a fetch is expected to bring down, spelled out independently of the
    production constant -- a test that imported `SERVICE_ONLY_DIRS` would agree with a wrong one."""
    return [e for e in entries if not e["name"].startswith(("detail/", "corpus/"))]


class _Api:
    """The org's artifact surface: a listing, and bodies derived from each name.

    Records every download under a lock so a test can assert on what the pool actually did.
    """

    def __init__(self, entries, refuse: set[str] | None = None):
        self.entries = entries
        self.refuse = refuse or set()
        self.lock = threading.Lock()
        self.downloaded: Counter[str] = Counter()
        self.threads: set[int] = set()
        self.barrier: threading.Barrier | None = None

    def list_artifacts(self, http, org, run_id):
        return list(self.entries)

    def download_artifact(self, http, org, run_id, name, dest, *args, **kwargs):
        with self.lock:
            self.downloaded[name] += 1
            self.threads.add(threading.get_ident())
        if self.barrier is not None:
            # Bounds the FAILURE, never decides a pass: a build where these do not overlap raises
            # BrokenBarrier rather than depending on which thread the machine scheduled first.
            try:
                self.barrier.wait()
            except threading.BrokenBarrierError:
                pass
        if name in self.refuse:
            raise run_api_client.RunApiError("forbidden", code="denied")
        dest.write_text(json.dumps({"artifact": name}), encoding="utf-8")
        return dest


@pytest.fixture
def api(monkeypatch):
    """Install an `_Api` over the client module, returning a factory the test parameterises."""
    def install(entries=None, refuse=None):
        fake = _Api(entries if entries is not None else _entries(), refuse)
        monkeypatch.setattr(run_api_client, "list_artifacts", fake.list_artifacts)
        monkeypatch.setattr(run_api_client, "download_artifact", fake.download_artifact)
        return fake
    return install


def _fetch(tmp_path, workers, **kwargs):
    run_dir = tmp_path / f"w{workers}"
    return reconcile.fetch_artifacts(None, "acme", run_dir, show_progress=False,
                                     workers=workers, **kwargs)


def _on_disk(root):
    return {str(p.relative_to(root)): p.read_bytes()
            for p in sorted(root.rglob("*")) if p.is_file()}


# ------------------------------------------------------------------ #
# The property: same artifacts, same order, any width
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("workers", WORKER_COUNTS)
def test_the_same_files_land_in_the_same_order_at_every_width(tmp_path, api, workers):
    api()
    sequential = _fetch(tmp_path, 1)

    written = _fetch(tmp_path, workers)

    assert [p.name for p in written] == [p.name for p in sequential]
    assert _on_disk(tmp_path / f"w{workers}") == _on_disk(tmp_path / "w1")


@pytest.mark.parametrize("workers", WORKER_COUNTS)
def test_the_order_returned_is_the_listing_order(tmp_path, api, workers):
    """Not completion order. `runs show` reads this list positionally, and a fetch that reordered
    itself by which thread won would be a different answer at every width."""
    fake = api()

    written = _fetch(tmp_path, workers, everything=True)

    assert [p.name for p in written] == [e["name"].rsplit("/", 1)[-1] for e in fake.entries]


@pytest.mark.parametrize("workers", WORKER_COUNTS)
def test_the_selection_keeps_its_order_too(tmp_path, api, workers):
    fake = api()

    written = _fetch(tmp_path, workers)

    assert [p.name for p in written] == [e["name"] for e in _renderable(fake.entries)]


def test_every_artifact_is_downloaded_exactly_once(tmp_path, api):
    fake = api()

    _fetch(tmp_path, 8, everything=True)

    assert set(fake.downloaded) == {e["name"] for e in fake.entries}
    assert set(fake.downloaded.values()) == {1}


def test_the_downloads_really_do_overlap(tmp_path, api):
    """The point of the change. Without this, every assertion above would also pass a fetch that
    quietly stayed sequential."""
    fake = api(entries=_entries(details=1))
    fake.barrier = threading.Barrier(2, timeout=10)

    _fetch(tmp_path, 4)

    assert len(fake.threads) > 1


# ------------------------------------------------------------------ #
# Widths that are not the interesting ones
# ------------------------------------------------------------------ #

def test_a_pool_wider_than_the_work_is_harmless(tmp_path, api):
    api(entries=_entries(details=0)[:1])

    assert len(_fetch(tmp_path, 64)) == 1


def test_a_run_that_uploaded_nothing_builds_no_pool(tmp_path, api):
    """`max_workers=0` is an error rather than a no-op, and an empty listing is a real state --
    a run whose task died before it uploaded."""
    api(entries=[])

    assert _fetch(tmp_path, 8) == []


# ------------------------------------------------------------------ #
# Which failure is reported, and what it leaves behind
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("workers", WORKER_COUNTS)
def test_the_first_refusal_in_listing_order_is_the_one_raised(tmp_path, api, workers):
    """Two artifacts refuse; the earlier one is always the reason given. `ex.map` consumes futures
    by index, so this does not depend on which thread lost first -- and a fetch that named a
    different file on every attempt would send the user chasing the wrong one."""
    entries = _entries()
    api(entries=entries, refuse={entries[1]["name"], entries[3]["name"]})

    with pytest.raises(run_api_client.RunApiError) as raised:
        _fetch(tmp_path, workers)

    assert entries[1]["name"] in str(raised.value)
    assert entries[3]["name"] not in str(raised.value)


def test_a_refusal_names_the_artifact_and_keeps_its_code(tmp_path, api):
    """"forbidden" alone says nothing actionable when the fetch was two thousand files, and
    `code` is machine-dispatched rather than decorative."""
    api(refuse={"detail/myengine__locomo/q0003.json"})

    with pytest.raises(run_api_client.RunApiError) as raised:
        _fetch(tmp_path, 8, everything=True)

    assert "detail/myengine__locomo/q0003.json" in str(raised.value)
    assert raised.value.code == "denied"


def test_a_failed_fetch_leaves_the_run_unreconciled(tmp_path, api, monkeypatch):
    """No partial success. A run whose results did not all arrive must not read as recorded -- it
    stays retryable, which is the whole reason the refusal is not caught here."""
    monkeypatch.setenv("MEMRANK_RUNS_DIR", str(tmp_path))
    monkeypatch.setattr("memrank.orchestration.sweep._mirror_to_mlflow", lambda run_dir: None)
    api(refuse={"record.json"})
    run_dir = tmp_path / "20260804-040634__locomo__e850cb"

    with pytest.raises(run_api_client.RunApiError):
        reconcile.reconcile(None, "acme", run_dir, {"state": "stopped-success"})

    assert not (run_dir / "record.json").exists()
    assert (run_status.read(run_dir) or {}).get("state") != "done"


# ------------------------------------------------------------------ #
# What the fetch says, and on whose thread it says it
# ------------------------------------------------------------------ #

def test_only_the_artifacts_worth_waiting_for_are_named(tmp_path, api):
    """1540 detail files used to be 1540 lines of narration, which hid the 83 MB one among them."""
    api(entries=_entries(details=1540))
    said: list[str] = []

    _fetch(tmp_path, 8, report=said.append, everything=True)

    assert said == ["1545 artifact(s), 115.0 MB", "myengine__locomo.json (83.5 MB)"]


@pytest.mark.parametrize("workers", WORKER_COUNTS)
def test_report_never_runs_on_a_worker_thread(tmp_path, api, workers):
    """Two of the four callers write into a live `LiveRegion`, which only one writer may hold. So
    narration happens on the calling thread, before a byte moves -- not from inside the pool."""
    api()
    threads: set[int] = set()

    _fetch(tmp_path, workers, report=lambda _: threads.add(threading.get_ident()))

    assert threads == {threading.get_ident()}


# ------------------------------------------------------------------ #
# Fetching what this machine reads, and nothing else
# ------------------------------------------------------------------ #

def test_the_drilldown_artifacts_are_not_fetched(tmp_path, api):
    """The whole point. `detail/` and `corpus/` are read by the API service from S3 to serve the
    web drilldown (`api/results_repository.py`); no local path opens them, because `runs show` renders
    from `registry.cell_files`, a non-recursive glob of the run directory's top level.

    Stated as a COUNT because the cost of a fetch is a round trip per file rather than its bytes:
    a locomo smoke run listed 158 artifacts, 153 of them these, and the API serves them roughly
    one at a time. 158 requests is two minutes; 5 is seconds."""
    fake = api(entries=_entries(details=1540))

    written = _fetch(tmp_path, 8)

    assert len(written) == 4
    assert not [name for name in fake.downloaded if "/" in name]


def test_everything_still_brings_down_everything(tmp_path, api):
    """The escape hatch is a real mirror, not a slightly larger selection -- a caller archiving a
    run wants every byte it uploaded."""
    fake = api(entries=_entries(details=20))

    _fetch(tmp_path, 8, everything=True)

    assert set(fake.downloaded) == {e["name"] for e in fake.entries}


def test_what_stays_in_the_org_is_said_out_loud(tmp_path, api):
    """A fetch that lists 1545 and downloads 4 must not read as one that silently lost 1541. The
    artifacts are still there and still served; the line says so."""
    api(entries=_entries(details=1540))
    said: list[str] = []

    _fetch(tmp_path, 8, report=said.append)

    assert said[0] == "4 artifact(s), 83.7 MB (1541 drilldown artifact(s) stay in the org)"


def test_a_full_mirror_claims_nothing_stayed_behind(tmp_path, api):
    """No parenthetical when nothing was left -- the clause is information, not decoration."""
    api(entries=_entries(details=3))
    said: list[str] = []

    _fetch(tmp_path, 8, report=said.append, everything=True)

    assert "stay in the org" not in said[0]


def test_a_selectively_fetched_run_is_a_reconciled_run(tmp_path, api, monkeypatch):
    """The property that makes skipping safe: nothing a caller can see changes. The run reads as
    reconciled and renders its metrics from the cell file, exactly as a full mirror would."""
    from memrank.cli import sync as sync_cli
    from memrank.runs import registry

    monkeypatch.setenv("MEMRANK_RUNS_DIR", str(tmp_path))
    monkeypatch.setattr("memrank.orchestration.sweep._mirror_to_mlflow", lambda run_dir: None)
    api(entries=_entries(details=1540))
    run_dir = tmp_path / "20260804-040634__locomo__e850cb"

    reconcile.reconcile(None, "acme", run_dir, {"state": "stopped-success"})

    cells = registry.cell_files(run_dir)
    assert [p.name for p in cells] == ["myengine__locomo.json"], "the renderable cell is here"
    assert not sync_cli.is_unreconciled({"placement": "cloud", "state": "done",
                                         "_has_cells": bool(cells)})
