"""`memrank task-exec <target> <benchmark>` -- the manifest-driven run form a task executes."""
from __future__ import annotations

import json

from typer.testing import CliRunner

from memrank.adapters import get_adapter
from memrank.runner import app
from memrank.targets import resolve_target
from memrank.targets.factory import build_adapter
from tests import withheld

runner = CliRunner()

# The env route's own values, inlined. They lived in scripts/eval-configs/mem0-voyage.env until
# 2026-08-19, when that profile followed `mem0:voyage` into the research lane -- and a gate in this
# repo must not depend on a file in another one.
LEGACY_ENV = {"MEM0_EMBEDDER_PROVIDER": "voyage", "MEM0_EMBEDDER_MODEL": "voyage-4-large",
              "MEM0_EMBEDDING_DIMS": "1024", "MEM0_LLM_PROVIDER": "anthropic",
              "MEM0_LLM_MODEL": "claude-sonnet-4-5-20250929"}


def _run(tmp_path, *args, targets=1):
    """Invoke in EXECUTE mode (--run-id): submit itself now detaches by default, and these
    tests assert on files that exist only after the evaluation actually ran."""
    ids = [tok for i in range(targets) for tok in ("--run-id", f"{tmp_path.name}-{i}")]
    return runner.invoke(app, [*args, "--output-dir", str(tmp_path), *ids])


def test_positional_form_runs_and_writes_a_cell(tmp_path):
    result = _run(tmp_path, "task-exec", "word-overlap", "demo")
    assert result.exit_code == 0, result.output
    assert (tmp_path / "word-overlap__demo.json").exists()


def test_positional_target_without_benchmark_is_rejected(tmp_path):
    assert _run(tmp_path, "task-exec", "word-overlap").exit_code != 0


def test_unknown_target_ref_fails_rather_than_skipping(tmp_path):
    """The legacy form skips unknown adapters; a bad ref must be loud."""
    assert _run(tmp_path, "task-exec", "nope", "demo").exit_code != 0


def test_ref_with_preset_yields_a_filesystem_safe_cell_name(tmp_path):
    from memrank.runner import _safe_label

    assert _safe_label("mem0:voyage") == "mem0-voyage"
    assert _safe_label("atomicstrata/mem0:voyage") == "atomicstrata-mem0-voyage"


def test_comma_sweeps_targets(tmp_path):
    result = _run(tmp_path, "task-exec", "word-overlap,word-overlap", "demo",
                  targets=2)
    assert result.exit_code == 0, result.output


def test_manifest_route_matches_the_legacy_env_route(monkeypatch):
    """The milestone gate: the manifest path must be faithful to the env path it replaces."""
    for key, value in LEGACY_ENV.items():
        monkeypatch.setenv(key, value)
    withheld.require("mem0")
    legacy = get_adapter("mem0").effective_config()
    manifest = build_adapter(resolve_target("mem0", ["llm=anthropic/claude-sonnet-4-5-20250929",
                                                     "embedder=voyage/voyage-4-large",
                                                     "embedder.dims=1024"]),
                             verify_engine=False).effective_config()
    assert legacy == manifest


def test_conflicting_environment_aborts_the_run(tmp_path, monkeypatch):
    """A server configured with bge must not be evaluated under a target claiming otherwise."""
    monkeypatch.setenv("MEM0_EMBEDDER_MODEL", "BAAI/bge-small-en-v1.5")
    monkeypatch.setenv("MEM0_EMBEDDING_DIMS", "384")
    result = _run(tmp_path, "task-exec", "mem0", "demo")
    assert result.exit_code != 0
    assert not (tmp_path / "mem0__demo.json").exists()


def test_cell_json_records_manifest_components(tmp_path):
    _run(tmp_path, "task-exec", "word-overlap", "demo")
    cell = json.loads((tmp_path / "word-overlap__demo.json").read_text(encoding="utf-8"))
    components = cell["receipt"]["config"]["components"]
    assert components["engine"]["transport"] == "in-process"
