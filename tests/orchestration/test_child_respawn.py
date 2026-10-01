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
"""A submission's child must accept the command its parent renders for it.

The failure this pins happened on 2026-08-13: `submit word-overlap beam:100k-smoke --on none`
parsed the eval ref, DERIVED tier/slice into its locals, packed `dict(locals())` into params,
and `_child_argv` rendered them back as the retired `--tier`/`--slice` flags -- so the
background child refused its own respawn command at parse time and the run sat `queued`
forever, its only trace a usage error in run.log.

The test reproduces the production flow exactly: capture the params the parent actually packs
(by stubbing the spawn), render the child argv from them, and re-invoke the CLI on it with
execution stubbed. Parse-side and render-side are pinned to each other -- a future retirement
that forgets the renderer table fails here, deterministically.
"""

from __future__ import annotations

import json

from typer.testing import CliRunner

from memrank import runner
from memrank.orchestration import sweep
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


def test_the_child_parses_the_argv_its_parent_renders(tmp_path, monkeypatch):
    """End-to-end over a ref that carries a slice, the shape that derived the retired flags."""
    data = tmp_path / "locomo.json"
    data.write_text(json.dumps(_FIXTURE), encoding="utf-8")
    monkeypatch.setenv("LOCOMO_DATA_PATH", str(data))
    monkeypatch.setenv("MEMRANK_RUNS_DIR", str(tmp_path / "runs"))

    captured: dict = {}

    def capture_spawn(params, runs):
        captured["argv"] = runner._child_argv(params, [r.run_dir.name for r in runs])

    monkeypatch.setattr(sweep, "_spawn_child", capture_spawn)
    # `--no-judge` keeps this about PARSING: locomo judges by default now, and a judged run needs
    # a credential this test has no business holding. The judge decision still rides in the argv
    # either way, which is the half that matters here.
    submitted = runner_cli.invoke(app, ["submit", "no-context", "locomo:smoke", "--on", "none",
                                        "--no-judge", "--output-dir", str(tmp_path / "out")])
    assert submitted.exit_code == 0, submitted.output
    argv = captured["argv"]
    assert "--tier" not in argv and "--slice" not in argv, (
        f"the parent rendered retired flags into its child's command: {argv}")
    assert "--no-judge" in argv, "the child must be TOLD the decision, not left to re-derive it"

    # The child's half: the rendered argv must PARSE and reach execute mode. Execution itself
    # is stubbed -- what died in production was the parse, before any work.
    monkeypatch.setattr(runner, "_run_local_sweep", lambda *a, **k: None)
    child = runner_cli.invoke(app, argv)
    assert child.exit_code == 0, (
        f"the child refused the command its own parent rendered:\n{child.output}")
