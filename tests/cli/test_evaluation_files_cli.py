"""``memrank evals new|check`` and ``memrank run <file>``: a person's own evaluation file.

A malformed file is refused before anything runs -- before sign-in, before the agent is read --
naming each case and field, as a step for the reader.
"""

from __future__ import annotations

import sys

import pytest
from typer.testing import CliRunner

from memrank.cli import agents as agents_cli
from memrank.loop import upload
from memrank.placement import run_api_client
from memrank.runner import app, main
from tests.cli.test_run_endings import run_cli
from tests.cli.test_run_missing_steps import world  # noqa: F401 - the fixture
from tests.definitions.conftest import FIXTURES

runner = CliRunner()


def boundary(mp, capsys, *argv: str) -> tuple[int, str]:
    """``memrank ...`` through the real boundary, where a refusal is rendered."""
    mp.setattr(sys, "argv", ["memrank", *argv])
    with pytest.raises(SystemExit) as exited:
        main()
    shown = capsys.readouterr()
    return int(exited.value.code or 0), shown.out + shown.err


def test_new_writes_a_starter_that_checks_clean(tmp_path):
    path = tmp_path / "mine.yaml"
    wrote = runner.invoke(app, ["evals", "new", str(path)])
    checked = runner.invoke(app, ["evals", "check", str(path)])
    assert wrote.exit_code == 0 and "# A memrank evaluation" in path.read_text()
    assert checked.exit_code == 0, checked.output
    for row in ("cases", "questions", "categories", "grading", "judge model", "size",
                "fingerprint"):
        assert row in checked.stdout


@pytest.mark.parametrize("name, refusal", [("exists.yaml", "already exists"),
                                           ("mine.json", "is not a .yaml file")])
def test_new_refuses_to_overwrite_or_to_write_other_formats(monkeypatch, capsys, tmp_path, name,
                                                            refusal):
    (tmp_path / "exists.yaml").write_text("keep me")
    code, shown = boundary(monkeypatch, capsys, "evals", "new", str(tmp_path / name))
    assert code == 1 and refusal in shown and "error:" not in shown
    assert (tmp_path / "exists.yaml").read_text() == "keep me"


def test_check_summarises_a_file_and_runs_its_case_program():
    result = runner.invoke(app, ["evals", "check", str(FIXTURES / "counting.yaml"),
                                 "--seed", "3"])
    assert result.exit_code == 0, result.output
    assert "counting@1" in result.stdout and "--seed 3" in result.stdout
    assert "not used; no key needed" in result.stdout


def test_check_refuses_a_broken_file_as_a_step(monkeypatch, capsys):
    code, shown = boundary(monkeypatch, capsys, "evals", "check", str(FIXTURES / "broken.yaml"))
    assert code == 1 and "! The evaluation file needs fixing" in shown
    assert "case 1 (one), question 1, `answer`: is required by the exact grader" in shown


def test_run_refuses_a_broken_file_before_sign_in(world, capsys):  # noqa: F811
    def never():
        raise AssertionError("signed in before the file was checked")

    world.setattr(run_api_client, "authenticated_client", never)
    code, shown = run_cli(world, capsys, str(FIXTURES / "broken.yaml"), "--agent",
                          "full-context")
    assert code == 1 and "! The run didn't start" in shown
    assert "5 problems, so nothing ran" in shown and "memrank evals check" in shown


@pytest.mark.parametrize("evaluation, needs_key", [("counting.yaml", False),
                                                   ("travel.yaml", True)])
def test_only_an_evaluation_the_model_grades_asks_for_the_key(world, capsys,  # noqa: F811
                                                             evaluation, needs_key):
    asked: list[bool] = []

    def connect(*, judge, ask=None, rerun=None):
        asked.append(judge)
        raise upload.LoginRequired("stop here")

    world.setattr(upload, "connect", connect)
    world.setattr(agents_cli, "_interactive", lambda: False)
    run_cli(world, capsys, str(FIXTURES / evaluation), "--agent", "full-context")
    assert asked == [needs_key]
