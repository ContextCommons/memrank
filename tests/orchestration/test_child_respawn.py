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
"""The local background child must accept the command its parent renders for it.

The failure this pins happened on 2026-08-13: a parent derived tier/slice from the eval ref, and
`_child_argv` rendered them back as the retired `--tier`/`--slice` flags -- so the background child
refused its own respawn command at parse time and the run sat `queued` forever.

`memrank submit` is removed (ATO-2366). The parent that still spawns a child is the MCP server's
local placement, whose parameters come from `remote_params`; the child runs the hidden
`task-exec`. Parse-side and render-side are pinned to each other here, and so is the placement:
the child must execute under the placement its parent chose, and never under `cloud`.
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from memrank import runner
from memrank.application.planning import plan_sweep
from memrank.application.submission import remote_params
from memrank.application.types import Routing, SweepRequest
from memrank.runner import app

runner_cli = CliRunner()

_FIXTURE = [
    {"sample_id": "s1",
     "conversation": {
         "speaker_a": "A", "speaker_b": "B",
         "session_1": [{"dia_id": "D1:1", "speaker": "A", "text": "we met twice"}],
         "session_1_date_time": "1:00 pm on 8 May, 2023"},
     "qa": [{"category": 1, "question": "how many times?", "answer": "twice",
             "evidence": ["D1:1"]}]}
]


@pytest.fixture
def locomo(tmp_path, monkeypatch):
    data = tmp_path / "locomo.json"
    data.write_text(json.dumps(_FIXTURE), encoding="utf-8")
    monkeypatch.setenv("LOCOMO_DATA_PATH", str(data))
    monkeypatch.setenv("MEMRANK_RUNS_DIR", str(tmp_path / "runs"))


def _child_argv(placement: str, tmp_path) -> list[str]:
    """What the MCP server's local parent hands its child for one planned cell."""
    plan = plan_sweep(SweepRequest(targets=["no-context"], evals=["locomo:smoke"],
                                   routing=Routing(placement=placement)))
    params = remote_params(plan.experiments[0], plan, [])
    params["output_dir"] = tmp_path / "out"
    return runner._child_argv(params, ["c1"])


def _executed_opts(argv: list[str], monkeypatch) -> list:
    """Invoke the child's command with execution stubbed; return what it would have run."""
    seen: list = []
    monkeypatch.setattr(runner, "_run_local_sweep",
                        lambda factories, opts, run_ids: seen.append(opts))
    monkeypatch.setattr(runner, "_require_placement_ready", lambda on, target=None: None)
    child = runner_cli.invoke(app, argv)
    assert child.exit_code == 0, (
        f"the child refused the command its own parent rendered:\n{child.output}")
    return seen


def test_the_child_parses_the_argv_its_parent_renders(locomo, tmp_path, monkeypatch):
    """End-to-end over a ref that carries a slice, the shape that derived the retired flags."""
    argv = _child_argv("none", tmp_path)

    assert argv[0] == "task-exec"
    assert "--tier" not in argv and "--slice" not in argv, (
        f"the parent rendered retired flags into its child's command: {argv}")
    assert "--no-judge" in argv, "the child must be TOLD the decision, not left to re-derive it"
    assert len(_executed_opts(argv, monkeypatch)) == 1


def test_the_local_child_forwards_local_placement(locomo, tmp_path, monkeypatch):
    """A parent that chose `local` must have its child provision the engine, not assume one."""
    argv = _child_argv("local", tmp_path)

    on = argv.index("--on")
    assert argv[on + 1] == "local"
    assert [opts.on for opts in _executed_opts(argv, monkeypatch)] == ["local"]


def test_task_exec_refuses_cloud_placement(tmp_path, monkeypatch):
    """A task must never place another run: `--on cloud` is refused before anything is minted."""
    monkeypatch.setenv("MEMRANK_RUNS_DIR", str(tmp_path / "runs"))
    result = runner_cli.invoke(app, ["task-exec", "word-overlap", "demo", "--on", "cloud",
                                     "--run-id", "c1", "--output-dir", str(tmp_path / "out")])

    assert result.exit_code == 2
    assert "cloud" in result.output
    assert not (tmp_path / "runs").exists(), "nothing may be minted for a refused placement"


def test_task_exec_executes_only_runs_its_caller_minted(tmp_path, monkeypatch):
    """Without --run-id there is nothing to execute, and task-exec never mints a run itself."""
    monkeypatch.setenv("MEMRANK_RUNS_DIR", str(tmp_path / "runs"))
    result = runner_cli.invoke(app, ["task-exec", "word-overlap", "demo"])

    assert result.exit_code == 2
    assert "--run-id" in result.output
    assert not (tmp_path / "runs").exists()


def test_submit_is_not_a_command():
    """ATO-2366: not hidden, not deprecated -- gone."""
    result = runner_cli.invoke(app, ["submit", "word-overlap", "demo"])

    assert result.exit_code == 2
    assert "No such command" in result.output
