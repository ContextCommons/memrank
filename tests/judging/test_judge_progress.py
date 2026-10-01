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
"""The judge phase reports itself.

Judging is five sequential LLM calls per query -- ~15 minutes on a LoCoMo smoke -- and it used to
publish nothing at all. `status.json` went untouched for the whole phase, so `runs show` froze at
`retrieving` and `watch` called a working run `stale (the process died without recording an
outcome)`. A live run was twice declared dead on 2026-08-04 because of it.

The judge is injectable (`JudgeConfig.completer`), so all of this runs against a scripted
completer with no network and no timing.
"""
from __future__ import annotations

import json

import pytest

from memrank import runner
from memrank.evaluation.judge_stage import AllVerdictsUnparseable
from memrank.judging.judge import JudgeConfig
from memrank.orchestration.observers import RunStatusObserver
from memrank.runs import status as run_status
from tests.fakes import JudgeFakeBenchmark, make_fake_completer

#: `JudgeFakeBenchmark` has three queries and exactly two the judge can grade: `event_ordering`
#: needs a rank-correlation scorer, so the binary prompt is not valid for it. Two, not three, is
#: the assertion that the denominator counts JUDGEABLE queries.
JUDGEABLE = 2


@pytest.fixture
def units():
    return JudgeFakeBenchmark().load()


@pytest.fixture
def judging(tmp_path, units):
    """A registered run mid-judge, observed exactly as `_execute_local_run` observes one."""
    status = run_status.RunStatus.create(tmp_path, target="word-overlap", benchmark="judgefake",
                                         slice_=None)
    cfg = JudgeConfig(no_context_control=False)
    observer = RunStatusObserver(status)
    observer.planned(runner._eval_plan(units, 1, cfg))
    return status, cfg, observer


def _rows(units):
    """Drill rows as `_run_units_*` produce them -- one per query, with what was retrieved."""
    return [{"query_id": q["id"], "retrieved": [{"content": "favorite drink is green tea"}]}
            for unit in units for q in unit.queries]


def _judge(units, cfg, observer, **kwargs):
    return runner._apply_judge(units, _rows(units), cfg, make_fake_completer(**kwargs),
                               observer=observer)


def test_the_denominator_is_the_judgeable_queries(judging):
    """Sized before the first call, by the same predicate that decides what gets graded."""
    _status, _cfg, observer = judging
    assert observer._progress.as_dict()["judge"]["total"] == JUDGEABLE


def test_the_phase_announces_itself_before_its_first_call(judging, units):
    """Each query is five LLM calls, so waiting for the first verdict would leave the run
    reporting the PREVIOUS stage for minutes -- which is exactly what read as frozen."""
    status, cfg, observer = judging
    state_when_called = []

    def watching_completer(model, system, user):
        state_when_called.append(status.data["state"])
        return make_fake_completer()(model, system, user)

    runner._apply_judge(units, _rows(units), cfg, watching_completer, observer=observer)

    assert state_when_called[0] == "judging", "published before the first call went out"


def test_the_stage_advances_once_per_judged_query(judging, units):
    """The heartbeat that was missing. Two of three queries are judgeable, and both land."""
    _status, cfg, observer = judging

    metrics = _judge(units, cfg, observer)
    judge = observer._progress.as_dict()["judge"]

    assert metrics["n_judged"] == JUDGEABLE
    assert judge["done"] == JUDGEABLE == judge["total"]
    assert judge["rate_per_second"] is not None, "so the phase carries an ETA of its own"


def test_the_heartbeat_on_disk_says_judging(judging, tmp_path, units):
    """What `runs show` and `ps` read. It froze at `retrieving` for the whole phase."""
    _status, cfg, observer = judging

    _judge(units, cfg, observer)
    data = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))

    assert data["state"] == "judging"
    assert data["progress"]["stage"] == "judge"
    assert data["progress"]["judge"]["done"] == JUDGEABLE


def test_an_unjudgeable_verdict_still_costs_the_run_its_time(judging, units, monkeypatch):
    """A query whose verdict will not parse burned five calls before saying so. Counting it out
    of `done` would leave the bar short of a total it could then never reach -- a stuck bar, on a
    run that is fine. Progress does not lie about it.

    Every query fails here, which is now the loud case: `_apply_judge` raises rather than handing
    back a receipt at `judged_coverage` 0.0 (ATO-1885). The progress property under test is
    unchanged and is asserted where it now lands -- on the observer, after the raise, because the
    items were attempted and timed before the stage refused its own result."""
    _status, cfg, observer = judging
    # Patched on the SHAPE, which is where grading now happens; this used to reach for
    # `runner.judge_query`. Only the injection point moved -- every assertion below is unchanged.
    monkeypatch.setattr(runner.BinaryJudgeShape, "grade",
                        lambda *a, **k: (_ for _ in ()).throw(runner.UnparseableVerdict("nope")))

    with pytest.raises(AllVerdictsUnparseable):
        _judge(units, cfg, observer)
    judge = observer._progress.as_dict()["judge"]

    assert judge["done"] == JUDGEABLE, "attempted, timed, and counted"


def test_an_unjudged_run_plans_no_judge_work(units):
    """`--judge` off is the common case and must be untouched by any of this."""
    assert runner._planned_progress(units, 1, None).as_dict()["judge"]["total"] == 0
