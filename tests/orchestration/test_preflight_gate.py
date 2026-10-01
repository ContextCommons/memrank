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
"""An engine that is not there must be a sentence, before anything runs.

`memrank submit atomicmemory,mem0 demo` against no backends produced ~200 lines of httpx
traceback from deep inside ingest. `compare` had the gate that prevents this all along; the
product's central verb did not.
"""
from __future__ import annotations

import pytest
from typer.testing import CliRunner

from memrank.adapters.preflight import PreflightError, preflight
from memrank.runner import app
from memrank.runs import registry

runner = CliRunner()


class Unreachable:
    """An engine whose every call fails the way a dead HTTP backend does."""

    name = "unreachable"
    base_url = "http://localhost:9"
    # The variable that moves the address, which the failure quotes. Declared here because a real
    # adapter declares it: preflight reads it off the adapter rather than holding a table.
    base_url_env = "UNREACHABLE_API_URL"
    graph_capable = False

    def __init__(self) -> None:
        self.cleaned = False

    def prepare(self, isolation_unit: str) -> None:
        raise ConnectionRefusedError("[Errno 61] Connection refused")

    def retrieve(self, *a, **kw):
        raise ConnectionRefusedError("[Errno 61] Connection refused")

    def cleanup(self) -> None:
        # Real adapters reset their isolation source over HTTP here, so on a dead backend
        # cleanup fails too. That must not replace the diagnosis.
        self.cleaned = True
        raise ConnectionRefusedError("[Errno 61] Connection refused")

    def close(self) -> None:
        return None


def test_a_failing_cleanup_does_not_replace_the_diagnosis():
    """A raise from `finally` swaps the reason for a second copy of the symptom."""
    adapter = Unreachable()
    with pytest.raises(PreflightError) as caught:
        preflight(adapter)

    assert "unreachable" in str(caught.value)
    assert "http://localhost:9" in str(caught.value)
    assert adapter.cleaned, "cleanup must still be attempted"


def test_the_message_names_both_ways_out():
    """Point memrank at the engine that is running, or have memrank start one.

    Both remedies are things the reader can do with what they installed. The message this
    replaced named a shell script in the maintainers' own checkout, which the person who most
    needs it does not have (ATO-2125); `tests/repo/test_user_facing_paths.py` is what keeps a
    sibling message from doing it again.
    """
    with pytest.raises(PreflightError) as caught:
        preflight(Unreachable())

    message = str(caught.value)
    assert "UNREACHABLE_API_URL" in message, "name the variable, do not describe it"
    assert "base_url" in message
    assert "--on local" in message


#: Real refs for the stubbed factories to wear. `_validate_source_target` resolves every label
#: against the catalog to decide placement, so a made-up one dies there as an unknown target --
#: before the probe this module is about ever runs. The adapters below are still the fakes; only
#: the label is real, and the probe reports `adapter.name`, not the ref.
LABELS = ("word-overlap", "no-context")


class _Ran:
    """What `_submit` reports: the exit code and what the terminal actually showed."""

    def __init__(self, exit_code: int, output: str) -> None:
        self.exit_code = exit_code
        self.output = output


def _submit(monkeypatch, tmp_path, capsys, factories, *args):
    """Run `submit` with the target resolution stubbed to these factories.

    Driven through ``runner_module.main`` rather than ``CliRunner.invoke``, because the error
    boundary lives on ``_BoundedTyper.__call__`` and CliRunner calls the click command underneath
    it -- see tests/cli/test_cli_errors.py. A refusal reaches ``result.output`` only through the
    boundary; under CliRunner it lands in ``result.exception`` and the terminal stays empty, which
    is not what a user sees.
    """
    from memrank import runner as runner_module

    monkeypatch.setenv("MEMRANK_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("MEMRANK_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setattr(runner_module, "_run_targets",
                        lambda **kw: ("demo", [(LABELS[i], factory)
                                               for i, (_, factory) in enumerate(factories)]))
    monkeypatch.setattr("sys.argv", ["memrank", "submit", "x", "demo",
                                     "--output-dir", str(tmp_path / "out"), *args])
    with pytest.raises(SystemExit) as exited:
        runner_module.main()
    printed = capsys.readouterr()
    return _Ran(exited.value.code, printed.err + printed.out)


def test_an_unreachable_engine_stops_the_run_before_any_cell(monkeypatch, tmp_path, capsys):
    """The property the up-front gate buys over failing at first contact: nothing ran."""
    result = _submit(monkeypatch, tmp_path, capsys, [("unreachable", Unreachable)])

    assert result.exit_code == 1
    assert "is not answering at http://localhost:9" in result.output
    assert "Traceback" not in result.output
    assert not list(registry.runs_root().glob("*")), "no run may be recorded"


def test_a_dead_second_target_fails_before_the_first_one_ingests(monkeypatch, tmp_path, capsys):
    """compare's stated reason, now true for submit: an earlier engine must not pay for a
    later engine's absence."""
    ingested: list[str] = []

    class Reachable(Unreachable):
        name = "reachable"

        def prepare(self, isolation_unit: str) -> None:
            ingested.append(isolation_unit)

        def retrieve(self, *a, **kw):
            return []

        def cleanup(self) -> None:
            return None

    result = _submit(monkeypatch, tmp_path, capsys,
                     [("reachable", Reachable), ("unreachable", Unreachable)])

    assert result.exit_code == 1
    assert "unreachable" in result.output
    assert ingested == ["__memrank_preflight__"], "only the probe, never a real ingest"


def test_in_process_targets_are_unaffected(monkeypatch, tmp_path):
    """The probe must not become a new way for a local demo run to fail."""
    monkeypatch.setenv("MEMRANK_RUNS_DIR", str(tmp_path / "runs"))
    result = runner.invoke(app, ["submit", "word-overlap,no-context", "demo",
                                 "--output-dir", str(tmp_path / "out"),
                                 "--run-id", "p1", "--run-id", "p2"])
    assert result.exit_code == 0, result.output


def test_a_local_placement_is_not_probed_again(monkeypatch, tmp_path):
    """`--on local` provisions its own engine and blocks on readiness; probing here would
    assert what the placement has just proven -- and would run before provisioning."""
    probed: list = []
    from memrank import runner as runner_module
    from memrank.orchestration import sweep as sweep_module

    monkeypatch.setattr(sweep_module, "preflight", probed.append)
    monkeypatch.setattr(runner_module, "_run_targets", lambda **kw: ("demo", []))
    runner_module._preflight_engines([("x", Unreachable)], "demo", on="local")

    assert probed == []
