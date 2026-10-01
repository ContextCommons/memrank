"""Run identity, the recorded target ref, and the configurable reader."""
from __future__ import annotations

import json

from typer.testing import CliRunner

from memrank.runner import _judge_cfg_or_refuse, _split_overrides, app, run_identity

runner = CliRunner()


def _run(tmp_path, *args):
    """Execute mode (--run-id): these tests read output submit-mode only produces in a child."""
    return runner.invoke(app, [*args, "--output-dir", str(tmp_path),
                               "--run-id", f"{tmp_path.name}-0"])


def test_identity_is_a_readable_stem_plus_short_hash():
    assert run_identity("mem0:voyage", "4c1f9a" + "0" * 58) == "mem0-voyage-4c1f9a"


def test_identity_without_a_hash_is_just_the_stem():
    assert run_identity("word-overlap", None) == "word-overlap"


def test_identity_differs_when_the_config_hash_differs():
    assert run_identity("mem0", "aaa" + "0" * 61) != run_identity("mem0", "bbb" + "0" * 61)


def test_run_prints_the_identity_and_the_benchmark(tmp_path):
    """Both halves matter: the identity says WHICH config, the benchmark says on what."""
    result = _run(tmp_path, "submit", "word-overlap", "demo")
    assert result.exit_code == 0, result.output
    assert "[word-overlap-" in result.output          # stem + hash, not the bare label
    assert "× demo]" in result.output             # benchmark retained alongside it


def test_cell_records_the_target_ref(tmp_path):
    """The `adapter` field is the adapter NAME; without this the variant is unrecoverable."""
    _run(tmp_path, "submit", "word-overlap", "demo")
    cell = json.loads((tmp_path / "word-overlap__demo.json").read_text(encoding="utf-8"))
    assert cell["target"] == "word-overlap"
    assert cell["receipt"]["config_hash_version"] >= 1


def test_split_overrides_separates_run_level_from_component_keys():
    components, run_level = _split_overrides(["embedder.dims=1024", "reader=gpt-4o-mini"])
    assert components == ["embedder.dims=1024"]
    assert run_level == {"reader": "gpt-4o-mini"}


def test_reader_override_reaches_the_judge_config():
    cfg = _judge_cfg_or_refuse(enabled=True, samples=1,
                               no_cache=True, reader="gpt-4o-mini")
    assert cfg.answer_model == "gpt-4o-mini"


def test_reader_without_judge_is_rejected_not_ignored(tmp_path):
    """It previously ran happily and did nothing -- exit 0, identity unchanged, no warning."""
    result = _run(tmp_path, "submit", "word-overlap", "demo",
                  "reader=claude-haiku-4-5-20251001")
    assert result.exit_code != 0
    assert "no effect without" in result.output      # short: Rich wraps the full sentence


def test_non_anthropic_reader_is_rejected_at_parse_time(tmp_path):
    """It previously parsed, started the run, then died at the first API call."""
    result = _run(tmp_path, "submit", "word-overlap", "demo",
                  "--judge", "reader=gpt-4o-mini")
    assert result.exit_code != 0
    assert "not an Anthropic model" in result.output


def test_reader_defaults_are_unchanged_when_not_overridden():
    """Every existing comparison must remain valid."""
    cfg = _judge_cfg_or_refuse(enabled=True, samples=1,
                               no_cache=True)
    assert cfg.answer_model == "claude-sonnet-4-6"
