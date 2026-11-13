"""Total evaluation runtime (ATO-2698): only the time a run was going, counted across resumes,
and no figure at all when the run's sessions cannot account for the whole of it."""

from __future__ import annotations

import json
from pathlib import Path

import click
import pytest

from memrank.connect.spec import AgentSpec
from memrank.loop import runtime
from memrank.loop.run import Instruments, RunAborted, RunOptions, run_agent
from memrank.service.protocol import AgentRef
from tests.loop.test_run import (  # noqa: F401 - hosted and org are pytest fixtures
    GoesDown,
    full_context,
    hosted,
    org,
)

#: How long the run sits stopped between its two sessions, in the shared clock's seconds.
STOPPED_S = 10_000.0


class Clock:
    """One clock across sessions: a second per reading, and a jump to stand for a stop."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        self.now += 1.0
        return self.now


def _tools(clock: Clock) -> Instruments:
    return Instruments(clock=clock, sleep=lambda s: None, echo=lambda line: click.unstyle(line))


def _ledger(folder: Path) -> list[dict]:
    lines = (folder / runtime.SESSIONS_FILE).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def test_a_stopped_then_resumed_run_counts_only_the_time_it_was_going(engine, tmp_path,
                                                                        hosted):  # noqa: F811
    clock = Clock()
    down = AgentSpec(ref=AgentRef(name="a"), connector=GoesDown(), start=None, description=None)
    with pytest.raises(RunAborted):
        run_agent(engine, down, RunOptions(evaluation="locomo"), tmp_path, hosted, _tools(clock))
    folder = next(tmp_path.glob(f"*/{runtime.SESSIONS_FILE}")).parent
    first = _ledger(folder)
    assert [line["event"] for line in first] == ["start", "end"] and first[0]["created"]
    clock.now += STOPPED_S
    resumed = RunOptions(resume=folder.name)
    finished = run_agent(engine, full_context(), resumed, tmp_path, hosted, _tools(clock))
    total = finished.result.runtime_ms
    assert total is not None
    assert first[1]["active_ms"] < total < STOPPED_S * 1000.0
    saved = json.loads((folder / "result.json").read_text(encoding="utf-8"))
    assert saved["runtime_ms"] == total


def test_a_run_that_never_stopped_records_its_runtime(engine, tmp_path, hosted):  # noqa: F811
    clock = Clock()
    finished = run_agent(engine, full_context(), RunOptions(evaluation="locomo"), tmp_path,
                         hosted, _tools(clock))
    assert finished.result.runtime_ms is not None and finished.result.runtime_ms > 0


def _session(folder: Path, created: bool, clock: Clock) -> runtime.Session:
    return runtime.begin(folder, created=created, clock=clock, started=clock())


def test_sessions_add_up_and_the_gap_between_them_is_not_counted(tmp_path):
    clock = Clock()
    first = _session(tmp_path, True, clock)
    clock.now += 4.0
    first.end()  # began at 1, ended at 6: 5 s
    clock.now += STOPPED_S
    second = _session(tmp_path, False, clock)  # begins at 10007
    clock.now += 2.0
    assert second.total_ms() == 5000.0 + 3000.0  # read at 10010


def test_a_run_begun_before_runtime_was_recorded_has_no_total(tmp_path):
    clock = Clock()
    assert _session(tmp_path, False, clock).total_ms() is None


def test_a_session_that_never_recorded_its_end_leaves_no_total(tmp_path):
    clock = Clock()
    _session(tmp_path, True, clock)  # killed: no end line
    assert _session(tmp_path, False, clock).total_ms() is None
