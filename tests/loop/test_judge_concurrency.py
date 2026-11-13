"""``memrank run --judge-concurrency``: how many judge calls run at once.

The grading pool is as wide as the judge ceiling, the opening line and the session ledger state
it, and no score depends on it.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import click
import pytest

from memrank.loop import grading as loop_grading
from memrank.loop import runtime
from memrank.loop.event_log import EventLog
from memrank.loop.grading_pool import GradingPool
from memrank.loop.run import Instruments, RunOptions, run_agent
from memrank.loop.stages import Ceilings
from memrank.service.protocol import QuestionOutcome
from tests.loop.conftest import StubGrading, passed
from tests.loop.test_run import full_context, hosted, org  # noqa: F401 - pytest fixtures

#: A guard against a hang, never a timing assumption: each wait it bounds is asserted.
HANG_GUARD_S = 30.0


def _run(engine, folder: Path, hosted, options: RunOptions):  # noqa: F811
    echoed: list[str] = []
    tools = Instruments(clock=lambda: 0.0, sleep=lambda s: None,
                        echo=lambda line: echoed.append(click.unstyle(line)))
    return run_agent(engine, full_context(), options, folder, hosted, tools), echoed


def _started(folder: Path) -> dict:
    ledger = (folder / runtime.SESSIONS_FILE).read_text(encoding="utf-8").splitlines()
    return json.loads(ledger[0])


@pytest.mark.parametrize("width", [8, 5, 1])
def test_the_grading_pool_is_as_wide_as_the_judge_ceiling(engine, tmp_path, hosted,  # noqa: F811
                                                          monkeypatch, width):
    widths: list[int] = []
    real = loop_grading.GradingPool

    def recording(*args, **kwargs):
        widths.append(kwargs["workers"])
        return real(*args, **kwargs)

    monkeypatch.setattr(loop_grading, "GradingPool", recording)
    options = RunOptions(evaluation="locomo", ceilings=Ceilings(ingest=2, answer=2, judge=width))
    done, echoed = _run(engine, tmp_path, hosted, options)
    assert widths == [width]
    calls = "call" if width == 1 else "calls"
    assert f"memrank: At most 2 feeds, 2 answers, {width} judge {calls} at once" in echoed
    assert _started(done.folder)["judge_concurrency"] == width


def test_no_score_depends_on_the_judge_concurrency(engine, tmp_path, hosted):  # noqa: F811
    results = [_run(engine, tmp_path / str(width), hosted,
                    RunOptions(evaluation="locomo",
                               ceilings=Ceilings(judge=width)))[0].result
               for width in (1, 7)]
    one, many = (r.model_dump(exclude={"run_id", "runtime_ms", "latency_ms", "cases"})
                 for r in results)
    assert one == many
    verdicts = [[(q.question_id, q.status, q.score, q.rationale) for c in r.cases
                 for q in c.questions] for r in results]
    assert verdicts[0] == verdicts[1]


def test_the_pool_grades_as_many_answers_at_once_as_it_is_given(tmp_path):
    width = 3
    together = threading.Barrier(width, timeout=HANG_GUARD_S)

    def meets_the_others(question: QuestionOutcome):
        together.wait()  # breaks, failing every grading, unless `width` run at once
        return passed(question)

    graders = GradingPool(EventLog(tmp_path), StubGrading(meets_the_others),
                          resume="memrank run --resume r1", workers=width)
    graders.start()
    for index in range(width):
        outcome = QuestionOutcome(case_id="c1", question_id=f"q{index}", question="?",
                                  reference=["blue"], category=None, status="not_judged",
                                  answer="blue", elapsed_ms=1.0)
        graders.submit(graders.log.answer(position=index, answer_ref=f"s{index}",
                                          outcome=outcome))
    assert sorted(graders.finish()) == ["q0", "q1", "q2"]
    graders.close()
