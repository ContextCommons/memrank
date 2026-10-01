"""The whole loop in process: engine, runner, openai-chat preset, full-context reference agent."""

from __future__ import annotations

import itertools
import json
from dataclasses import replace
from html.parser import HTMLParser

import click
import pytest

from memrank.benchmarks.answer_prompts import LOCOMO_READER, LONGMEMEVAL_READER
from memrank.connect.base import AgentError, AgentUnreachable
from memrank.connect.http import HttpConnector, HttpSpec
from memrank.connect.presets import preset_mapping
from memrank.connect.probe import leak_probe
from memrank.connect.spec import AgentSpec
from memrank.judging.judge import JudgeConfig
from memrank.loop import judge as loop_judge
from memrank.loop import run as loop_run
from memrank.loop.client import ServiceClient
from memrank.loop.judge_key import JudgeKeyMissing
from memrank.loop.run import (
    Instruments,
    ProbeIncomplete,
    RunAborted,
    RunOptions,
    UploadIncomplete,
    execute,
    run_agent,
)
from memrank.loop.upload import Hosted
from memrank.placement.run_api_client import RunApiError
from memrank.reference.full_context import create_app as agent_app
from memrank.service.protocol import AgentRef, Op, Question, RunResult, Step
from tests.asgi_transport import asgi_transport
from tests.loop.conftest import OrgApi
from tests.service.conftest import FakeJudge

LOCOMO = RunOptions(evaluation="locomo")


class Org(OrgApi):
    """The organisation's side: its runs API, and the key each judging used."""

    def __init__(self) -> None:
        super().__init__()
        self.keys: list[str] = []

    def judge(self, result: RunResult, api_key: str, cfg: JudgeConfig,
              judged=lambda: None) -> RunResult:
        self.keys.append(api_key)
        return loop_judge.judge(result, api_key,
                                replace(cfg, completer=FakeJudge(), cache=False), judged=judged)


@pytest.fixture(autouse=True)
def org(monkeypatch) -> Org:
    """Judging with a fake judge that passes answers saying "blue", and a fake runs API."""
    fake = Org()
    monkeypatch.setattr(loop_run, "judge", fake.judge)
    return fake


@pytest.fixture
def hosted(org) -> Hosted:
    return org.hosted()


def reader(model: str, system: str, user: str) -> str:
    """Answers "blue" when the memory mentions it -- a reader that only knows its context.

    The context is what precedes LoCoMo's official instruction (LOCOMO_READER)."""
    context = user.split("\n\nBased on the above context", 1)[0]
    return "blue" if "blue" in context else "I don't know"


def full_context(complete=reader) -> AgentSpec:
    fields = {**preset_mapping("openai-chat", {"model": "m"}), "base_url": "http://agent",
              "vars": {"model": "m"}}
    connector = HttpConnector(HttpSpec.model_validate(fields),
                              transport=asgi_transport(agent_app(complete, "m", LOCOMO_READER)))
    return AgentSpec(ref=AgentRef(name="full-context"), connector=connector, start=None,
                     description=None)


def quiet(echoed: list[str]) -> Instruments:
    ticks = itertools.count()
    # Unstyled, as a pipe sees each line: click strips the colour on its way to one.
    return Instruments(clock=lambda: float(next(ticks)), sleep=lambda s: None,
                       echo=lambda line: echoed.append(click.unstyle(line)))


class Rows(HTMLParser):
    """Counts the rows of the report's per-question table."""

    def __init__(self) -> None:
        super().__init__()
        self.inside, self.rows = False, 0

    def handle_starttag(self, tag, attrs):
        self.inside = self.inside or (tag == "table" and ("id", "questions") in attrs)
        self.rows += int(self.inside and tag == "tr")

    def handle_endtag(self, tag):
        self.inside = self.inside and tag != "table"


@pytest.mark.parametrize("options,model", [
    (LOCOMO, "claude-haiku-4-5"),
    (RunOptions(evaluation="locomo", judge_model="claude-sonnet-4-6"), "claude-sonnet-4-6"),
])
def test_a_judged_run_names_its_judge_haiku_unless_told(engine, tmp_path, hosted, options,
                                                        model):
    echoed: list[str] = []
    done = run_agent(engine, full_context(), options, tmp_path, hosted, quiet(echoed))
    assert done.result.identity.judge_model == model
    assert f"memrank: Judge {model} with acme's key" in echoed


def test_the_reference_agent_passes_the_leak_probe():
    leak_probe(full_context().connector, token="feedface")


def test_a_run_in_process_writes_its_folder(engine, tmp_path, hosted):
    echoed: list[str] = []
    done = run_agent(engine, full_context(), LOCOMO, tmp_path / "results", hosted, quiet(echoed))
    result, folder = done.result, done.folder
    assert done.url == f"https://memrank.test/acme/runs/{result.run_id}"
    answers = [q.answer for c in result.cases for q in c.questions]
    assert answers == ["blue", "blue", "I don't know"]
    assert result.score.mean == pytest.approx(2 / 3) and result.failed == 0
    assert json.loads((folder / "result.json").read_text())["run_id"] == result.run_id
    table = Rows()
    table.feed((folder / "report.html").read_text())
    assert table.rows == 1 + 3  # header plus one row per question
    assert any("ask 2/2" in line for line in echoed)


def test_the_same_run_over_http_writes_the_same_outputs(client, tmp_path, hosted):
    service = ServiceClient("http://service", transport=asgi_transport(client.app))
    done = run_agent(service, full_context(), LOCOMO, tmp_path / "out", hosted, quiet([]))
    result, folder = done.result, done.folder
    assert result.score.mean == pytest.approx(2 / 3)
    assert (folder / "report.html").is_file()


class Flaky:
    """Unreachable ``down`` times, then answers -- or fails once reached, when told to."""

    def __init__(self, down: int, fail: bool = False) -> None:
        self.down, self.fail, self.calls = down, fail, 0

    def ask(self, step: Step) -> str:
        self.calls += 1
        if self.calls <= self.down:
            raise AgentUnreachable("refused")
        if self.fail:
            raise AgentError("500")
        return "ok"


ASK = Step(step_id="s1", op=Op.ASK, question=Question(id="q", text="?"))


def test_only_an_unreached_call_is_retried_and_only_the_reaching_attempt_is_timed():
    agent = Flaky(down=2)
    result = execute(agent, ASK, quiet([]))
    assert agent.calls == 3 and result.answer == "ok" and result.elapsed_ms == 1000.0


def test_a_reached_failure_is_recorded_once_and_not_retried():
    agent = Flaky(down=0, fail=True)
    result = execute(agent, ASK, quiet([]))
    assert agent.calls == 1 and result.error == "500"


class GoesDown:
    """Passes the leak probe (four calls), then refuses every connection."""

    def __init__(self) -> None:
        self.calls = 0

    def _call(self, step: Step) -> str:
        self.calls += 1
        if self.calls > 4:
            raise AgentUnreachable("refused")
        return "I don't know"

    reset = feed = ask = _call

    def close(self) -> None:
        pass


def test_an_agent_that_stays_unreachable_aborts_a_resumable_run(engine, tmp_path, hosted):
    agent = AgentSpec(ref=AgentRef(name="a"), connector=GoesDown(), start=None, description=None)
    with pytest.raises(RunAborted, match="memrank run --resume"):
        run_agent(engine, agent, LOCOMO, tmp_path, hosted, quiet([]))


class CreditTooLow(Exception):
    """What Anthropic's SDK raises for a spent balance: a status and the provider's words."""

    status_code = 400


def test_a_model_failure_reaches_the_runner_with_its_reason(engine, tmp_path, hosted):
    def broke(model: str, system: str, user: str) -> str:
        raise CreditTooLow("Your credit balance is too low to access the Anthropic API.")

    with pytest.raises(ProbeIncomplete) as refused:
        run_agent(engine, full_context(broke), LOCOMO, tmp_path, hosted, quiet([]))
    lead, detail = str(refused.value).split("\n", 1)
    assert "no question was asked" in lead
    assert detail.startswith("The agent's model call was refused by its provider "
                             "(anthropic, 400) -- that is the model account, not memrank.")
    assert "502" in detail and "credit balance is too low" in detail  # the payload, kept whole


def test_a_failed_upload_keeps_the_judged_run_and_says_how_to_send_it(engine, tmp_path,
                                                                     monkeypatch, hosted):
    def refused(hosted, result, echo):
        raise RunApiError("the API is down (503)")

    monkeypatch.setattr(loop_run, "upload", refused)
    with pytest.raises(UploadIncomplete, match="(?s)API is down.*memrank run --resume"):
        run_agent(engine, full_context(), LOCOMO, tmp_path, hosted, quiet([]))
    saved = json.loads(next(tmp_path.glob("*/result.json")).read_text())
    assert saved["judged"] is True and len(saved["cases"]) == 2


def test_a_resume_sends_the_saved_run_without_judging_it_again(engine, tmp_path, org, hosted):
    first = run_agent(engine, full_context(), LOCOMO, tmp_path, hosted, quiet([]))
    resumed = RunOptions(resume=first.result.run_id)
    again = run_agent(engine, full_context(), resumed, tmp_path, hosted, quiet([]))
    assert org.keys == ["org-key"] and len(org.uploads) == 2
    assert again.result == first.result


def test_a_judged_run_without_the_orgs_key_is_refused_and_keeps_the_answers(engine, tmp_path,
                                                                            org, hosted):
    keyless = org.hosted(judge_key=None)
    with pytest.raises(JudgeKeyMissing, match="memrank secrets set ANTHROPIC_API_KEY --org acme"):
        run_agent(engine, full_context(), LOCOMO, tmp_path, keyless, quiet([]))
    saved = json.loads(next(tmp_path.glob("*/result.json")).read_text())
    assert saved["judged"] is False and org.uploads == []


def test_a_run_without_judging_is_uploaded_unscored(engine, tmp_path, org):
    options = RunOptions(evaluation="locomo", judge=False)
    done = run_agent(engine, full_context(), options, tmp_path, org.hosted(judge_key=None),
                     quiet([]))
    assert not done.result.judged and done.result.score.mean is None
    assert org.keys == [] and org.uploads == [done.result]


def test_a_longmemeval_run_gets_past_the_leak_probe(engine, tmp_path, hosted):
    """LongMemEval's official reader states the current date: an undated probe question stopped
    every `memrank run longmemeval` before its first question."""
    def dated_reader(model: str, system: str, user: str) -> str:
        context = user.split("History Chats:", 1)[-1].split("Current Date:", 1)[0]
        return "blue" if "blue" in context else "I don't know"

    fields = {**preset_mapping("openai-chat", {"model": "m"}), "base_url": "http://agent",
              "vars": {"model": "m"}}
    app = agent_app(dated_reader, "m", LONGMEMEVAL_READER)
    connector = HttpConnector(HttpSpec.model_validate(fields), transport=asgi_transport(app))
    agent = AgentSpec(ref=AgentRef(name="full-context"), connector=connector, start=None,
                      description=None)
    done = run_agent(engine, agent, RunOptions(evaluation="longmemeval:smoke"), tmp_path,
                     hosted, quiet([]))
    assert done.result.failed == 0
    assert [q.answer for c in done.result.cases for q in c.questions] == [
        "blue", "blue", "I don't know"]
