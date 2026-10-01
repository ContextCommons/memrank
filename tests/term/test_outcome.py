"""The ending renderer: one shape for every ending, and the same words with or without colour."""

from __future__ import annotations

import io

import pytest

from memrank.outcome import Details, Kind, Link, Outcome, Step
from memrank.term import outcome, style

ACTION = Outcome(kind=Kind.ACTION, title="Judging couldn't run", achieved="Answered all 3",
                 happened="Anthropic refused.",
                 steps=(Step("Save a key:", ("memrank secrets set K --org o",)),
                        Step("Then resume:", ("memrank run --resume r --agent a",))),
                 note="1 answer failed.", saved="results/r/", link=Link("View the run", "https://x/r"),
                 details=Details("Anthropic", ("400 invalid_request_error: no credit",)))


def test_an_action_ending_reads_top_to_bottom():
    assert outcome.plain(ACTION).splitlines() == [
        f"{outcome.CHECK} Answered all 3", "! Judging couldn't run", "",
        "What happened", "  Anthropic refused.", "",
        "What to do", "  1. Save a key:", "       memrank secrets set K --org o",
        "  2. Then resume:", "       memrank run --resume r --agent a", "",
        "Note: 1 answer failed.", "",
        "Saved locally: results/r/", "View the run: https://x/r", "",
        "Details from Anthropic:", "  400 invalid_request_error: no credit"]


def test_one_step_is_not_numbered_and_a_clean_finish_is_one_block():
    one = Outcome(kind=Kind.ACTION, title="t", steps=(Step("Sign in:", ("memrank auth login",)),))
    assert outcome.plain(one).splitlines()[-2:] == ["  Sign in:", "    memrank auth login"]
    done = Outcome(kind=Kind.DONE, title="Done", facts=(("Score", "40.0%"), ("Failures", "none")),
                   link=Link("View results", "https://x/r"))
    assert outcome.plain(done).splitlines() == [
        f"{outcome.CHECK} Done", "  Score     40.0%", "  Failures  none", "View results: https://x/r"]


class Tty(io.StringIO):
    def isatty(self) -> bool:
        return True


@pytest.mark.parametrize(("env", "decorated"), [({}, True), ({"NO_COLOR": "1"}, False),
                                                ({"CI": "true"}, False),
                                                ({"TERM": "dumb"}, False)],
                         ids=["terminal", "no-color", "ci", "dumb"])
def test_only_a_live_terminal_is_decorated(monkeypatch, env, decorated):
    for name in ("NO_COLOR", "CI", "TERM"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    assert style.decorates(Tty()) is decorated
    assert style.decorates(io.StringIO()) is False
    assert (style.link("https://x", Tty()) != "https://x") is decorated
