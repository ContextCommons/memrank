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
"""``memrank run`` says which organisation a run is recorded in, and which run it is.

Through the real CLI boundary with a model-free command agent: the opening lines come before
the first step, the ending names the org and the run again, and the words are the same whether
or not the terminal gets colour.
"""

from __future__ import annotations

import re
import sys

import pytest

from memrank import runner
from memrank.service import engine as service_engine
from memrank.term import style
from tests.cli.test_run_missing_steps import ORG, world  # noqa: F401 - the fixture
from tests.service.conftest import PLAN

#: Every ANSI escape memrank emits: SGR colour and OSC 8 links.
ANSI = re.compile(r"\x1b\]8;;[^\x1b]*\x1b\\|\x1b\[[0-9;]*m")


@pytest.fixture
def agent(world, tmp_path) -> str:  # noqa: F811 - the fixture, used by name
    """A command agent that needs no model: every answer is ``blue``."""
    world.setattr(service_engine, "build_plan", lambda request: PLAN)
    spec = tmp_path / "agent.yaml"
    spec.write_text("name: blue\nconnector: command\n"
                    "reset: ['{python}', '-c', 'pass']\n"
                    "feed: ['{python}', '-c', 'pass']\n"
                    "ask: ['{python}', '-c', 'print(\"blue\")']\n", encoding="utf-8")
    return str(spec)


def run(mp, capsys, agent: str) -> tuple[str, str]:
    """``memrank run locomo --agent <agent> --no-judge``; what it wrote to stdout and stderr.

    One lane: two cases side by side interleave their progress lines in whatever order the
    threads reach them, and two runs' transcripts are compared line for line.
    """
    mp.setattr(sys, "argv", ["memrank", "run", "locomo", "--agent", agent, "--no-judge",
                             "--concurrency", "1"])
    with pytest.raises(SystemExit) as exited:
        runner.main()
    shown = capsys.readouterr()
    assert exited.value.code in (0, None), shown.out + shown.err
    return shown.out, shown.err


def test_the_run_opens_with_the_account_the_org_and_the_run(monkeypatch, capsys, agent):
    out, err = run(monkeypatch, capsys, agent)
    lines = err.splitlines()
    run_line = next(line for line in lines if line.startswith("memrank: Run "))
    run_id = run_line.split()[2]
    assert lines[:3] == ["memrank: Signed in as ada <ada@example.test>",
                         f"memrank: Recording to org {ORG} (Acme Labs)",
                         f"memrank: Run {run_id} \u00b7 locomo \u00b7 agent blue \u00b7 "
                         f"2 cases, 3 questions"]
    assert f"memrank: Saving locally to results/{run_id}/" in lines
    assert not any(line.startswith("memrank: Judge ") for line in lines)  # --no-judge
    first_step = next(i for i, line in enumerate(lines) if line.startswith("case 1/2"))
    assert lines.index(run_line) < first_step
    assert re.search(rf"^  Org +{ORG}$", out, re.M) and re.search(rf"^  Run +{run_id}$", out,
                                                                   re.M), out


def test_progress_is_narration_and_the_ending_is_the_answer(monkeypatch, capsys, agent):
    out, err = run(monkeypatch, capsys, agent)
    assert re.search(r"^case 1/2 c1  ask 1/2 +ok  \d+\.\d\ds$", err, re.M), err
    assert "case 1/2" not in out and "memrank:" not in out
    assert out.startswith("\u2713 Done: 3 questions answered, not judged")


def test_a_pipe_gets_no_colour_and_a_terminal_gets_the_same_words(monkeypatch, capsys, agent):
    plain_out, plain_err = run(monkeypatch, capsys, agent)
    assert "\x1b" not in plain_out + plain_err
    monkeypatch.setattr("click.utils.should_strip_ansi", lambda *a, **k: False)
    monkeypatch.setattr(style, "decorates", lambda stream: True)
    out, err = run(monkeypatch, capsys, agent)
    assert style.identity(ORG) in err and style.identity(ORG) in out  # bold yellow
    run_id = next(line for line in ANSI.sub("", err).splitlines()
                  if line.startswith("memrank: Run ")).split()[2]
    assert style.identity(run_id) in err and style.identity(run_id) in out
    assert style.good("ok") in err
    assert _same_words(ANSI.sub("", err), plain_err) and _same_words(ANSI.sub("", out), plain_out)


def test_no_color_gets_the_same_words_plain_even_on_a_terminal(monkeypatch, capsys, agent):
    monkeypatch.setattr("click.utils.should_strip_ansi", lambda *a, **k: False)
    monkeypatch.setenv("NO_COLOR", "1")
    out, err = run(monkeypatch, capsys, agent)
    assert "\x1b" not in out + err and f"memrank: Recording to org {ORG}" in err


def _same_words(one: str, other: str) -> bool:
    """Equal once run ids, timings and the run's folder are masked: two runs, same transcript."""
    def mask(text: str) -> str:
        text = re.sub(r"\d{8}-\d{6}__\w+__\w+", "<run>", text)
        text = re.sub(r"\d+\.\d\ds", "<s>", text)
        return re.sub(r"\d+ ms|\d+\.\d s|\d+ min \d+ s", "<s>", text)
    return mask(one) == mask(other)
