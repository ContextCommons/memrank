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
"""The pre-agent surface warns where it is entered, and the current interface does not.

A runtime DeprecationWarning at the in-process Python route's verb, a one-line stderr notice on
each hidden engine-hosting command, and silence on import and on the commands `memrank --help`
lists.
"""

from __future__ import annotations

import subprocess
import sys

import pytest
from typer.testing import CliRunner

from memrank import deprecation
from memrank.runner import app

runner = CliRunner()


def test_evaluation_run_warns_and_still_runs():
    from memrank.evaluations import Demo
    from memrank.systems import TFIDF

    with pytest.warns(DeprecationWarning, match=r"evaluation\.run\(system=\.\.\.\) is deprecated"):
        result = Demo().run(system=TFIDF())
    assert result.traces


def test_importing_memrank_and_the_cli_does_not_warn():
    code = ("import warnings; warnings.simplefilter('always'); "
            "import memrank, memrank.runner, memrank.loop, memrank.connect, memrank.service")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          check=True)
    assert deprecation.REPLACEMENT not in done.stderr


@pytest.mark.parametrize("args", [["ps", "--help"], ["submit", "--help"]])
def test_hidden_commands_still_show_their_help_without_the_notice(args):
    assert runner.invoke(app, args).exit_code == 0


def test_a_hidden_command_prints_the_deprecation_notice(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMRANK_HOME", str(tmp_path))
    done = runner.invoke(app, ["targets", "ls"])
    assert "`memrank targets` is deprecated" in done.stderr


@pytest.mark.parametrize("args", [["--help"], ["run", "--help"], ["evals", "ls"],
                                  ["agents", "ls"], ["version"]])
def test_current_commands_print_no_deprecation(args):
    done = runner.invoke(app, args)
    assert "deprecated" not in done.output
