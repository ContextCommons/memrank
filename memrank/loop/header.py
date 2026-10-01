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
"""What ``memrank run`` says before its first step: who, where, which run, and its link.

A run is recorded in an organisation's history, and a person who belongs to two -- their own
and a team's -- must be able to tell which one before anything is sent. So the signed-in
account, the organisation and the run's id open the run, each ``memrank:`` line styled through
:mod:`memrank.term.style`: a pipe, ``NO_COLOR`` or CI get the same words plain.
"""

from __future__ import annotations

import sys
from pathlib import Path

from memrank.loop.upload import Hosted
from memrank.service.protocol import RunStatus
from memrank.term.style import bold, dim, identity, link, prefixed

#: Between the parts of the run's line. A middle dot, written as an escape to keep the source
#: ASCII.
_SEP = " \u00b7 "


def opening(hosted: Hosted, status: RunStatus, agent: str, judge_model: str,
            folder: Path) -> list[str]:
    """The lines before the run registers online: account, organisation, run, judge, folder."""
    account = hosted.account
    run = _SEP.join((f"Run {identity(status.run_id)}", status.evaluation, f"agent {bold(agent)}",
                     f"{status.cases} cases, {status.questions} questions"))
    lines = [f"Signed in as {identity(account.login)} {dim(f'<{account.email}>')}",
             f"Recording to org {identity(hosted.org)} {dim(f'({account.org_name})')}",
             run]
    if status.judge and hosted.judge_key is not None:
        whose = f"with {hosted.org}'s key"
        lines.append(f"Judge {bold(judge_model)} {dim(whose)}")
    lines.append(f"Saving locally to {dim(f'{folder}/')}")
    return [prefixed(line) for line in lines]


def live(url: str | None) -> str:
    """The run's page, once it is registered; narration, so its link is judged on stderr."""
    if url is None:
        return prefixed("This deployment gave the run no page to view it live")
    return prefixed(f"View run live at {link(url, sys.stderr)}")
