"""The run is online from its first step: link first, progress while it runs, stopped or done."""

from __future__ import annotations

import pytest

from memrank.loop import run as loop_run
from memrank.loop.run import Instruments, RunOptions, run_agent
from tests.loop.conftest import PAGE, OrgApi, run_id_of
from tests.loop.test_lanes import Memory, spec
from tests.loop.test_run import quiet

OPTIONS = RunOptions(evaluation="locomo", judge=False)


@pytest.fixture(autouse=True)
def org(monkeypatch) -> OrgApi:
    fake = OrgApi()
    monkeypatch.setattr(loop_run, "judge", lambda result, key, cfg, judged=None: result)
    return fake


class Interrupting(Memory):
    """Ctrl-C arriving while the agent answers its ``at``-th question, the leak probe's first."""

    def __init__(self, at: int) -> None:
        super().__init__()
        self.at, self.asked = at, 0

    def ask(self, step):
        self.asked += 1
        if self.asked == self.at:
            raise KeyboardInterrupt
        return super().ask(step)


def test_the_link_is_printed_before_any_step_and_the_run_ends_done(engine, tmp_path, org):
    echoed: list[str] = []
    done = run_agent(engine, spec(Memory()), OPTIONS, tmp_path, org.hosted(None), quiet(echoed))
    run_id = run_id_of(echoed)
    live = f"memrank: View run live at {PAGE.format(run_id=run_id)}"
    assert live in echoed
    assert echoed.index(live) < min(i for i, line in enumerate(echoed) if "reset" in line)
    assert org.states == ["running", "done"] and done.url == PAGE.format(run_id=run_id)
    first = org.syncs[0]["record"]["progress"]
    assert (first["units"], first["ingest"]["total"], first["retrieve"]["total"]) == (2, 2, 3)


def test_progress_is_sampled_on_the_servers_interval_in_sequence(engine, tmp_path, org):
    run_agent(engine, spec(Memory()), OPTIONS, tmp_path, org.hosted(None), quiet([]))
    sequences = [sample["sequence"] for sample in org.samples]
    assert sequences and sequences == sorted(set(sequences))
    last = org.samples[-1]["progress"]
    assert last["retrieve"]["done"] <= 3 and last["stage"] in ("ingest", "retrieve")


def test_a_progress_channel_that_fails_is_reported_once_and_the_run_finishes(engine, tmp_path,
                                                                            org):
    org.progress_down = True
    echoed: list[str] = []
    run_agent(engine, spec(Memory()), OPTIONS, tmp_path, org.hosted(None), quiet(echoed))
    told = [line for line in echoed if line.startswith("live progress could not be sent")]
    assert len(told) == 1 and "503" in told[0] and org.states == ["running", "done"]


def test_ctrl_c_leaves_the_run_stopped_online_and_a_resume_finishes_it_under_the_same_link(
        engine, tmp_path, org):
    echoed: list[str] = []
    one_lane = RunOptions(evaluation="locomo", judge=False, concurrency=1)  # "3rd ask" is an order
    with pytest.raises(KeyboardInterrupt):
        run_agent(engine, spec(Interrupting(at=3)), one_lane, tmp_path, org.hosted(None),
                  quiet(echoed))
    run_id = run_id_of(echoed)
    stopped = org.syncs[-1]
    assert org.states == ["running", "stopped"]
    assert stopped["error"].startswith("interrupted (Ctrl-C) -- resume with `memrank run "
                                       f"--resume {run_id}")
    assert stopped["record"]["progress"]["retrieve"]["done"] == 1
    resumed = run_agent(engine, spec(Memory()), RunOptions(resume=run_id, concurrency=1), tmp_path,
                        org.hosted(None), quiet([]))
    assert org.states == ["running", "stopped", "running", "done"]
    assert resumed.url == PAGE.format(run_id=run_id) and resumed.result.run_id == run_id
    assert [q.answer for c in resumed.result.cases for q in c.questions] == \
        ["blue", "blue", "I don't know"]


def test_the_sample_that_starts_judging_counts_every_step_done(engine, tmp_path, org,
                                                              monkeypatch):
    monkeypatch.setattr(loop_run, "judge", lambda result, key, cfg, judged=None: result)
    frozen = Instruments(clock=lambda: 0.0, sleep=lambda s: None, echo=lambda line: None)
    run_agent(engine, spec(Memory()), RunOptions(evaluation="locomo"), tmp_path,
              org.hosted(), frozen)  # a clock that never moves sends no sample during steps
    assert [s["progress"]["stage"] for s in org.samples] == ["judge"]
    judging = org.samples[0]["progress"]
    assert (judging["ingest"]["done"], judging["retrieve"]["done"]) == (2, 3)
    assert judging["judge"]["total"] == 3
