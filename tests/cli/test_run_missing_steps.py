"""Every missing step ``memrank run`` can hit reads as a step, and never as ``error:``.

Driven through the real CLI boundary (``runner.main``), because the boundary is where a
missing step and a failure part ways: a step renders as the "!" ending -- what happened, then
what to do, each command on its own indented line -- with no red and no ``error:`` label, and
still exits non-zero. The table is the list of missing-step cases; a case that ever renders
through the error path again fails here. ``test_run_endings`` covers every other ending.
"""

from __future__ import annotations

import sys
from collections.abc import Callable

import httpx
import pytest

from memrank import runner, settings
from memrank.cli import agents as agents_cli
from memrank.judging import client as judging_client
from memrank.loop import run as loop_run
from memrank.loop.run import Finished
from memrank.placement import run_api_client
from memrank.placement.run_api_client import RunApiError
from memrank.service import engine as service_engine
from tests.loop.conftest import WHOAMI
from tests.service.conftest import PLAN
from tests.service.test_scoring import recorded

ORG = "acme"
RERUN = "memrank run locomo --agent full-context --no-judge"
SAVE = f"memrank secrets set ANTHROPIC_API_KEY --org {ORG}"


def api(runs: int = 200, resolved: int | dict = 200, saved: list | None = None) -> httpx.Client:
    """The runs API: the status its runs listing answers, and the org's saved secrets."""
    def handle(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/secrets/resolved"):
            return httpx.Response(resolved, json={}) if isinstance(resolved, int) \
                else httpx.Response(200, json=resolved)
        if path == "/whoami":
            return httpx.Response(runs, json=WHOAMI)
        if path.endswith("/secrets") and saved is not None:
            saved.append(request.content)
            return httpx.Response(201, json={})
        return httpx.Response(runs, json={"runs": []})

    return httpx.Client(base_url="http://api", transport=httpx.MockTransport(handle))


@pytest.fixture
def world(monkeypatch, tmp_path):
    """A signed-in machine with a default org, no terminal, results under tmp_path."""
    monkeypatch.chdir(tmp_path)
    real_get = settings.get
    monkeypatch.setattr(settings, "get",
                        lambda key: ORG if key == "defaults.org" else real_get(key))
    monkeypatch.setattr(run_api_client, "authenticated_client", lambda: api())
    monkeypatch.setattr(agents_cli, "_interactive", lambda: False)
    return monkeypatch


def memrank(mp, capsys, *argv: str) -> tuple[int, str]:
    """Run ``memrank run ...`` through the real boundary; its exit code and stderr."""
    mp.setattr(sys, "argv", ["memrank", "run", *argv])
    with pytest.raises(SystemExit) as exited:
        runner.main()
    return int(exited.value.code or 0), capsys.readouterr().err


def signed_out(mp):
    def refuse():
        raise RunApiError("not signed in", code="no_session")

    mp.setattr(run_api_client, "authenticated_client", refuse)


def never_ready_agent(mp, tmp_path) -> str:
    mp.setattr(service_engine, "build_plan", lambda request: PLAN)
    spec = tmp_path / "agent.yaml"
    spec.write_text("name: dead\nconnector: openai-chat\nbase_url: http://127.0.0.1:9\n"
                    "vars: {model: m}\nstart:\n"
                    "  argv: ['{python}', '-c', 'import sys; sys.exit(3)']\n"
                    "  ready: http://127.0.0.1:9/health\n", encoding="utf-8")
    return str(spec)


def offers(shown: str, command: str) -> bool:
    """Whether ``command`` is on a line of its own, indented to be copied whole."""
    return any(line.strip() == command and line.startswith("    ")
               for line in shown.splitlines())


Setup = Callable[[pytest.MonkeyPatch], None]

#: (case, arrange the world, what `memrank run` is given, the lines that must be shown)
CASES: list[tuple[str, Setup, tuple[str, ...], tuple[str, ...]]] = [
    ("not-signed-in", signed_out, (), ("memrank auth login",)),
    ("session-rejected",
     lambda mp: mp.setattr(run_api_client, "authenticated_client", lambda: api(runs=401)),
     (), ("memrank auth login",)),
    ("not-a-member",
     lambda mp: mp.setattr(run_api_client, "authenticated_client", lambda: api(runs=403)),
     (), ("memrank config set defaults.org <slug>",)),
    ("no-default-org", lambda mp: mp.setattr(settings, "get", lambda key: None),
     (), ("memrank auth login", "memrank config set defaults.org <slug>")),
    ("no-judge-key-without-a-terminal",
     lambda mp: mp.setattr(run_api_client, "authenticated_client", lambda: api(resolved={})),
     ("--cases", "2"), (SAVE, "memrank run locomo --agent full-context --cases 2 --no-judge")),
    ("not-an-owner",
     lambda mp: mp.setattr(run_api_client, "authenticated_client", lambda: api(resolved=403)),
     (), (SAVE, RERUN)),
]


@pytest.mark.parametrize(("setup", "extra", "commands"), [case[1:] for case in CASES],
                         ids=[case[0] for case in CASES])
def test_a_missing_step_is_shown_as_a_step(world, capsys, setup, extra, commands):
    setup(world)
    code, shown = memrank(world, capsys, "locomo", "--agent", "full-context", *extra)
    assert code == 1 and "error:" not in shown and shown.startswith("! ")
    assert all(offers(shown, command) for command in commands), shown


def test_an_unknown_agent_is_a_step(world, capsys):
    code, shown = memrank(world, capsys, "locomo", "--agent", "nope")
    assert code == 1 and "error:" not in shown and offers(shown, "memrank agents ls")


def test_an_agent_that_never_starts_points_at_its_log(world, capsys, tmp_path):
    code, shown = memrank(world, capsys, "locomo", "--agent", never_ready_agent(world, tmp_path),
                          "--no-judge")
    # The run's opening lines come first: it had started when the agent failed to.
    ending = shown[shown.index("\n! ") + 1:]
    assert code == 1 and "error:" not in shown and ending.startswith("! Your agent didn't start")
    assert any(line.startswith("    ") and line.strip().startswith("results/")
               and line.endswith("/agent.log")
               for line in shown.splitlines()), shown
    assert "Details from your agent's output:" in shown


def test_a_real_failure_still_reads_as_a_failure(world, capsys):
    world.setattr(run_api_client, "authenticated_client", lambda: api(runs=503))
    code, shown = memrank(world, capsys, "locomo", "--agent", "full-context")
    assert code == 1 and shown.startswith("x ")


def test_cancelling_the_key_prompt_starts_nothing(world, capsys, tmp_path):
    world.setattr(run_api_client, "authenticated_client", lambda: api(resolved={}))
    world.setattr(agents_cli, "_interactive", lambda: True)
    world.setattr(agents_cli, "_read_key", lambda name: "")
    code, shown = memrank(world, capsys, "locomo", "--agent", "full-context")
    assert code == 1 and "error:" not in shown and offers(shown, RERUN)
    assert not (tmp_path / "results").exists()


def test_a_key_typed_at_the_prompt_is_saved_and_the_run_goes_on(world, capsys, tmp_path):
    saved: list[bytes] = []
    world.setattr(run_api_client, "authenticated_client",
                  lambda: api(resolved={}, saved=saved))
    world.setattr(agents_cli, "_interactive", lambda: True)
    world.setattr(agents_cli, "_read_key", lambda name: "sk-typed")
    world.setattr(judging_client, "key_refusal", lambda key: None)
    used: list[str | None] = []

    def run_agent(evaluator, spec, options, output_root, hosted, tools):
        used.append(hosted.judge_key)
        return Finished(result=recorded(), folder=tmp_path / "results" / "r", url=None,
                        org=hosted.org)

    world.setattr(loop_run, "run_agent", run_agent)
    code, shown = memrank(world, capsys, "locomo", "--agent", "full-context")
    assert code == 0 and used == ["sk-typed"] and len(saved) == 1
    assert "Saved ANTHROPIC_API_KEY to acme." in shown and "sk-typed" not in shown


def test_ctrl_c_at_the_hidden_prompt_reads_as_a_cancel(monkeypatch):
    def abort(*args, **kwargs):
        raise agents_cli.typer.Abort()

    monkeypatch.setattr(agents_cli.typer, "prompt", abort)
    assert agents_cli._read_key("ANTHROPIC_API_KEY") == ""
