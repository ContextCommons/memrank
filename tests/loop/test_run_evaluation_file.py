"""An evaluation file run end to end in process, exactly as a shipped benchmark is.

The keyword agent answers from what it was fed; the judged question goes to a fake judge; the
command grader and the choice, exact and numeric graders run for real.
"""

from __future__ import annotations

import itertools
import shutil
from dataclasses import replace

import pytest

from memrank.connect.spec import AgentSpec
from memrank.connect.template import transcript
from memrank.loop import judge as loop_judge
from memrank.loop import run as loop_run
from memrank.loop.judge_key import JudgeKeyMissing
from memrank.loop.run import Instruments, RunOptions, run_agent
from memrank.service.cases import EvaluationChanged
from memrank.service.engine import EvaluationService
from memrank.service.protocol import AgentRef, RunCreate, Step
from memrank.service.store import FolderRunStore
from tests.definitions.conftest import FIXTURES
from tests.fixtures.agents.keyword_agent import answer
from tests.loop.conftest import OrgApi

PASS = '{"passed": true, "rationale": "same"}'


class Keyword:
    """The keyword agent in process: one memory per session id."""

    def __init__(self) -> None:
        self.memory: dict[str, list[str]] = {}

    def reset(self, step: Step) -> None:
        self.memory.pop(str(step.session_id), None)

    def feed(self, step: Step) -> None:
        lines = transcript(step.sessions or []).splitlines()
        self.memory.setdefault(str(step.session_id), []).extend(lines)

    def ask(self, step: Step) -> str:
        assert step.question is not None
        return answer(self.memory.get(str(step.session_id), []), step.question.text)

    def close(self) -> None:
        """Nothing is held."""


def keyword() -> AgentSpec:
    return AgentSpec(ref=AgentRef(name="keyword"), connector=Keyword(), start=None,
                     description=None)


def quiet() -> Instruments:
    ticks = itertools.count()
    return Instruments(clock=lambda: float(next(ticks)), sleep=lambda s: None,
                       echo=lambda line: None)


@pytest.fixture
def judged(monkeypatch) -> list[str | None]:
    """The key each judging was given; the judge model is a fake that passes everything."""
    keys: list[str | None] = []

    def fake(result, api_key, cfg, judged=lambda: None):
        keys.append(api_key)
        cfg = replace(cfg, completer=lambda model, system, user: PASS, cache=False)
        return loop_judge.judge(result, api_key, cfg, judged=judged)

    monkeypatch.setattr(loop_run, "judge", fake)
    return keys


def run(tmp_path, evaluation: str, key: str | None = "org-key", org: OrgApi | None = None):
    org = org or OrgApi()
    engine = EvaluationService(store=FolderRunStore(tmp_path / "results"))
    return run_agent(engine, keyword(), RunOptions(evaluation=evaluation), tmp_path / "results",
                     org.hosted(judge_key=key), quiet()), org


def test_a_file_runs_and_uploads_under_its_name_and_version(tmp_path, judged):
    done, org = run(tmp_path, str(FIXTURES / "travel.yaml"))
    result = done.result
    verdicts = {q.question_id: (q.status, q.score) for c in result.cases for q in c.questions}
    assert verdicts == {"trip-city": ("judged", 0.0), "trip-nights": ("judged", 0.0),
                        "trip-when": ("judged", 1.0), "trip-mentions": ("judged", 1.0),
                        "capital": ("judged", 0.0)}
    assert result.evaluation == "travel@2" and judged == ["org-key"]
    identity = result.identity
    assert identity.evaluation_version == "2" and identity.judge_model == "claude-haiku-4-5"  # the default model grader
    assert identity.evaluation_fingerprint and identity.evaluation_source.endswith("travel.yaml")
    assert identity.dataset_version.startswith("cases sha256:")
    assert org.syncs[0]["benchmark"] == "travel@2" and org.uploads[0].evaluation == "travel@2"
    assert "__travel-2__" in result.run_id


def test_an_evaluation_graded_without_the_model_needs_no_key(tmp_path, judged):
    for name in ("counting.yaml", "make_cases.py", "grade_contains.py"):
        shutil.copy(FIXTURES / name, tmp_path / name)
    done, _ = run(tmp_path, str(tmp_path / "counting.yaml"), key=None)
    assert done.result.judged and done.result.score.mean == 1.0
    assert done.result.identity.judge_model is None and judged == [None]
    assert done.result.identity.judge_prompt_version is None


def test_a_judged_question_without_a_key_is_refused(tmp_path, judged):
    with pytest.raises(JudgeKeyMissing):
        run(tmp_path, str(FIXTURES / "travel.yaml"), key=None)


def test_a_run_does_not_continue_on_changed_cases(tmp_path):
    path = tmp_path / "eval.yaml"
    path.write_text("name: e\ncases: [{question: q, answer: a, grade: exact}]\n")
    engine = EvaluationService(store=FolderRunStore(tmp_path / "results"))
    run_id = engine.create(RunCreate(evaluation=str(path), agent=AgentRef(name="a"))).run_id
    path.write_text("name: e\ncases: [{question: other, answer: a, grade: exact}]\n")
    with pytest.raises(EvaluationChanged, match="started on other cases"):
        engine.next(run_id)
