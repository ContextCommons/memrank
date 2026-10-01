"""Every way ``memrank run`` ends reads as done, a step for the reader, or a memrank bug.

Driven through the real CLI boundary (``runner.main``), which is where an ending is rendered.
The table is the list of endings: each is arranged by making the run stop the way it would --
the loop's own exception, with the trace filled as far as that run got -- and each must show
its look, the commands to copy on lines of their own, the run's link wherever a run exists, and
no escape codes when the output is not a terminal. Only a failure inside memrank may say "bug
in memrank"; a provider, the network or the user's agent never does.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Callable

import anthropic
import httpx
import pytest

from memrank import runner
from memrank.api_client import ApiUnreachable
from memrank.connect.base import AgentError, AgentUnreachable
from memrank.connect.probe import LeakDetected
from memrank.definitions.commands import CommandFailed
from memrank.loop import run as loop_run
from memrank.loop.explain import judge_failure
from memrank.loop.run import Finished, ProbeIncomplete, RunAborted, UploadIncomplete
from memrank.loop.trace import Stage
from memrank.placement import run_api_client
from memrank.service.protocol import RunResult
from memrank.service.scoring import apply_verdicts
from memrank.term import outcome, style
from tests.cli.test_run_missing_steps import ORG, api, world  # noqa: F401 - the fixture
from tests.service.test_scoring import PASS, recorded

RUN = "20260927-213555__locomo__76506d"
PAGE = f"https://memrank.test/{ORG}/runs/{RUN}"
RESUME = f"memrank run --resume {RUN} --agent full-context"
SAVE = f"memrank secrets set ANTHROPIC_API_KEY --org {ORG}"
REFUSED_BY_PROVIDER = "the agent's model provider refused the calls (credit balance too low)"
CREDIT = {"type": "error", "error": {"type": "invalid_request_error",
                                     "message": "Your credit balance is too low to access the "
                                                "Anthropic API."}}
MARKS = {"done": f"{outcome.CHECK} ", "action": "! ", "broken": "x "}


def refused(status: int, body: dict | None = None) -> BaseException:
    """What judging raises when Anthropic refuses: the SDK's own error, as the loop names it."""
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    body = body or {"type": "error", "error": {"type": "error", "message": f"refused {status}"}}
    response = httpx.Response(status, json=body, request=request,
                              headers={"request-id": "req_011CfUfagiGA7SyNv8YWc3BW"})
    sdk = anthropic.APIStatusError(f"Error code: {status}", response=response, body=body)
    failure = judge_failure(sdk)
    assert failure is not None
    return failure


def unanswered() -> BaseException:
    """Judging with an SDK that failed on its own side, wrapped as a connection error."""
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    try:
        try:
            raise TypeError("Client.__init__() got an unexpected keyword argument 'proxies'")
        except TypeError as inner:
            raise anthropic.APIConnectionError(request=request) from inner
    except anthropic.APIConnectionError as sdk:
        failure = judge_failure(sdk)
    assert failure is not None
    return failure


def with_cause(exc: BaseException, cause: BaseException) -> BaseException:
    exc.__cause__ = cause
    return exc


def answered() -> RunResult:
    return recorded().model_copy(update={"run_id": RUN})


def judged() -> RunResult:
    verdicts = {q.question_id: PASS for case in answered().cases for q in case.questions
                if q.status == "not_judged"}
    return apply_verdicts(answered(), verdicts, judge_model="claude-judge", judge_samples=1)


def stops(exc: BaseException, stage: Stage = Stage.STEPS, *, run: bool = True,
          steps_done: bool = False, failures: int = 0) -> Callable:
    """A run that gets as far as ``stage`` and then stops on ``exc``."""
    def run_agent(evaluator, spec, options, output_root, hosted, tools):
        trace = tools.trace
        if run:
            trace.run_id, trace.url, trace.folder = RUN, PAGE, output_root / RUN
            trace.folder.mkdir(parents=True, exist_ok=True)
        trace.stage = stage
        if steps_done:
            trace.recorded = answered()
        for _ in range(failures):
            trace.failed(REFUSED_BY_PROVIDER)
        raise exc
    return run_agent


def finishes(result: RunResult) -> Callable:
    def run_agent(evaluator, spec, options, output_root, hosted, tools):
        tools.trace.run_id, tools.trace.folder = RUN, output_root / RUN
        return Finished(result=result, folder=output_root / RUN, url=PAGE, org=hosted.org)
    return run_agent


PROVIDER_502 = AgentError("POST /v1/chat/completions returned 502: {...}", status=502, body={
    "detail": {"code": "provider_refused", "provider": "anthropic", "status": 400,
               "key_name": "ANTHROPIC_API_KEY", "message": "BadRequestError: Your credit "
                                                           "balance is too low"}})
UNREACHABLE = ApiUnreachable("cannot reach the memrank API at http://127.0.0.1:9 (from "
                             "MEMRANK_API_URL in the environment): nothing answered there.\n"
                             "  the underlying error: PUT /orgs/acme/runs/x: ConnectError")

#: (ending, how the run goes, extra args, look, commands to copy, words that must be shown)
CASES: list[tuple[str, Callable, tuple[str, ...], str, tuple[str, ...], tuple[str, ...]]] = [
    ("done-judged", finishes(judged()), (), "done", (),
     ("Done: 3 of 4 questions answered and judged", "Score", "View results: " + PAGE)),
    ("done-not-judged", finishes(answered()), ("--no-judge",), "done", (),
     ("answered, not judged", "not judged (--no-judge)")),
    ("judge-no-credit", stops(refused(400, CREDIT), Stage.JUDGING, steps_done=True, failures=1),
     (), "action", (SAVE, RESUME),
     ("Your agent answered all 4 questions (1 of them failed", "Judging couldn't run",
      "has no credit left", "1 answer failed: " + REFUSED_BY_PROVIDER,
      "Details from Anthropic:", "req_011CfUfagiGA7SyNv8YWc3BW")),
    ("judge-key-rejected", stops(refused(401), Stage.JUDGING, steps_done=True), (), "action",
     (SAVE, RESUME), ("the key was not accepted",)),
    ("judge-not-permitted", stops(refused(403), Stage.JUDGING, steps_done=True), (), "action",
     (SAVE, RESUME), ("not allowed",)),
    ("judge-rate-limited", stops(refused(429), Stage.JUDGING, steps_done=True), (), "action",
     (RESUME,), ("rate limits",)),
    ("judge-unanswered", stops(unanswered(), Stage.JUDGING, steps_done=True), (), "action",
     (RESUME,), ("could not get a verdict", "caused by TypeError")),
    ("agent-check-refused", stops(with_cause(ProbeIncomplete("check"), PROVIDER_502),
                                  Stage.CHECK), (), "action", (RESUME,),
     ("keeps each conversation separate", "credit balance too low",
      "Details from Anthropic (through your agent):")),
    ("agent-not-running", stops(AgentUnreachable("could not connect to http://127.0.0.1:9"),
                                Stage.CHECK), (), "action", (RESUME,),
     ("memrank could not connect to your agent",)),
    ("agent-mixes-conversations", stops(LeakDetected("Your agent was told a code word"),
                                        Stage.CHECK), (), "action", (RESUME,),
     ("mixed up two conversations",)),
    ("agent-went-away", stops(with_cause(RunAborted("gone"), AgentUnreachable("refused")),
                              failures=2), (), "action", (RESUME,),
     ("stopped answering",)),
    ("evaluation-program-failed",
     stops(CommandFailed("The grader command exited 1: python grade.py\nKeyError: 'score'"),
           Stage.JUDGING, steps_done=True), (), "action", (RESUME,),
     ("Your evaluation's program failed", "Details from your program:", "KeyError")),
    ("ctrl-c", stops(KeyboardInterrupt()), (), "action", (RESUME,), ("you pressed Ctrl-C",)),
    ("api-unreachable-at-start", stops(UNREACHABLE, Stage.SETUP, run=False), (), "action",
     ("memrank run locomo --agent full-context",),
     ("The run didn't start", "Details from the memrank API:")),
    ("upload-failed", stops(with_cause(UploadIncomplete("x"), UNREACHABLE), Stage.UPLOAD,
                            steps_done=True), (), "action", (RESUME,),
     ("couldn't be saved to acme's run history", "Cannot reach the memrank API")),
    ("memrank-bug", stops(RuntimeError("boom")), (), "broken",
     ("https://github.com/ContextCommons/memrank/issues",
      "MEMRANK_DEBUG=1 memrank run locomo --agent full-context", RESUME),
     ("bug in memrank", "RuntimeError: boom")),
]


@pytest.fixture
def keyed(request, monkeypatch):
    """The signed-in machine (``world``), its organisation with an Anthropic key saved."""
    request.getfixturevalue("world")
    monkeypatch.setattr(run_api_client, "authenticated_client",
                        lambda: api(resolved={"ANTHROPIC_API_KEY": "sk-org"}))
    return monkeypatch


def run_cli(mp, capsys, *argv: str) -> tuple[int, str]:
    """``memrank run ...`` through the real boundary; its exit code and all it printed."""
    mp.setattr(sys, "argv", ["memrank", "run", *argv])
    with pytest.raises(SystemExit) as exited:
        runner.main()
    shown = capsys.readouterr()
    return int(exited.value.code or 0), shown.out + shown.err


def assert_ending(shown: str, look: str, commands: tuple[str, ...]) -> None:
    lines = shown.splitlines()
    titles = [line for line in lines if line[:2] in MARKS.values()]
    assert titles and titles[-1].startswith(MARKS[look]), shown
    assert "\x1b" not in shown and not any(line.startswith("error:") for line in lines), shown
    assert ("bug in memrank" in shown) == (look == "broken" and "RuntimeError" in shown), shown
    for command in commands:  # its own line, indented, nothing else on it
        assert any(line.strip() == command and line.startswith("    ") for line in lines), \
            (command, shown)


@pytest.mark.parametrize(("arrange", "extra", "look", "commands", "says"),
                         [case[1:] for case in CASES], ids=[case[0] for case in CASES])
def test_every_ending_says_where_you_are_and_what_to_do(keyed, capsys, arrange, extra, look,
                                                        commands, says):
    keyed.setattr(loop_run, "run_agent", arrange)
    code, shown = run_cli(keyed, capsys, "locomo", "--agent", "full-context", *extra)
    assert code == {"done": 0, "action": 1, "broken": 1}[look] or "Ctrl-C" in shown
    assert_ending(shown, look, commands)
    assert all(part in shown for part in says), shown
    if "run=False" not in repr(arrange) and RUN in shown:
        assert re.search(rf"^View (the run|results): {re.escape(PAGE)}$", shown, re.M), shown


def test_ctrl_c_exits_as_an_interrupt(keyed, capsys):
    keyed.setattr(loop_run, "run_agent", stops(KeyboardInterrupt()))
    code, _ = run_cli(keyed, capsys, "locomo", "--agent", "full-context")
    assert code == 130


def test_a_5xx_from_memranks_api_is_memranks_side_but_not_called_a_bug(keyed, capsys):
    keyed.setattr(run_api_client, "authenticated_client", lambda: api(runs=503))
    code, shown = run_cli(keyed, capsys, "locomo", "--agent", "full-context")
    assert code == 1 and shown.startswith("x memrank's service failed")
    assert "bug in memrank" not in shown and "Details from the memrank API:" in shown


def test_a_terminal_gets_colour_and_a_link_with_the_same_words(keyed, capsys, monkeypatch):
    keyed.setattr(loop_run, "run_agent", stops(refused(400, CREDIT), Stage.JUDGING,
                                               steps_done=True))
    _, plain = run_cli(keyed, capsys, "locomo", "--agent", "full-context")
    monkeypatch.setattr(style, "decorates", lambda stream: True)
    monkeypatch.setattr("click.utils.should_strip_ansi", lambda *a, **k: False)
    _, fancy = run_cli(keyed, capsys, "locomo", "--agent", "full-context")
    assert "\x1b]8;;" + PAGE in fancy and "\x1b[" in fancy
    visible = re.sub(r"\x1b\]8;;[^\x1b]*\x1b\\|\x1b\[[0-9;]*m", "", fancy)
    assert visible == plain


def test_resuming_a_run_that_is_not_there_says_so_without_a_bug(keyed, capsys):
    code, shown = run_cli(keyed, capsys, "--resume", "20260101-000000__locomo__000000",
                          "--agent", "full-context")
    assert code == 1 and shown.startswith("! The run didn't start")
    assert "no run '20260101-000000__locomo__000000'" in shown
    assert "bug in memrank" not in shown


def test_a_run_saved_in_a_shape_it_cannot_read_is_refused_plainly(keyed, capsys, tmp_path):
    run_dir = tmp_path / "results" / RUN
    run_dir.mkdir(parents=True)
    run_dir.joinpath("run.json").write_text('{"request": {}, "state": {}}', encoding="utf-8")
    code, shown = run_cli(keyed, capsys, "--resume", RUN, "--agent", "full-context")
    assert code == 1 and shown.startswith("! The run didn't start")
    assert "cannot read, so it cannot be resumed" in shown and "bug in memrank" not in shown


def test_resuming_with_another_agent_file_names_the_one_it_started_with(keyed, capsys):
    from memrank.service.protocol import AgentRef, RunCreate
    from memrank.service.store import FolderRunStore

    started = RunCreate(evaluation="locomo", agent=AgentRef(
        name="mine", spec_path="/work/mine.yaml", spec_sha256="abc"))
    keyed.setattr(FolderRunStore, "load", lambda self, run_id: (started, None))
    code, shown = run_cli(keyed, capsys, "--resume", RUN, "--agent", "full-context")
    assert code == 1 and shown.startswith("! The run didn't start")
    assert "started with a different agent file: /work/mine.yaml" in shown
    assert_ending(shown, "action", (f"memrank run --resume {RUN} --agent /work/mine.yaml",))
