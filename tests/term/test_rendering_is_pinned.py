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
"""The guard on the rendering pin that ``tests/conftest.py`` sets at import time.

Thirteen tests once asserted plain substrings on CLI output and failed only on GitHub
Actions, because typer detects ``GITHUB_ACTIONS`` and forces coloured rendering (ATO-1931).
The pin closes that for every CLI test at once; these two assert the pin is actually in
effect, so a future test cannot silently inherit a chokepoint that has stopped working.

The failure mode being guarded against is quiet: the pin is an environment variable read at
``typer.rich_utils`` import time, so anything that imports typer before the root conftest
runs -- a new plugin, a reordered import -- disarms it with no error anywhere.
"""
from __future__ import annotations

import os

import typer.rich_utils
from typer.testing import CliRunner

from memrank.runner import app

#: What a rich console emits when it believes it is writing to a terminal.
ANSI_PREFIX = "\033["

runner = CliRunner()


def test_typer_believes_it_is_not_writing_to_a_terminal():
    """The pin reached typer, whatever the CI detections say."""
    assert os.environ.get("_TYPER_FORCE_DISABLE_TERMINAL") == "1"
    assert typer.rich_utils.FORCE_TERMINAL is False


def test_cli_output_carries_no_escape_sequences():
    """The consequence, at the surface every CLI test reads: help and errors are plain.

    Both branches, because typer renders them through separate paths -- the help panels and
    the usage-error panel -- and only the second one broke the retired-flag assertions.
    """
    help_result = runner.invoke(app, ["--help"])
    assert ANSI_PREFIX not in help_result.output

    error_result = runner.invoke(app, ["submit", "--no-such-flag"])
    assert error_result.exit_code != 0
    assert ANSI_PREFIX not in error_result.output
