"""The `memrank targets` command surface."""
from __future__ import annotations

import json

from typer.testing import CliRunner

from memrank.runner import app
from tests import withheld

runner = CliRunner()

SOURCE_TARGET = """\
schema_version: 1
name: myengine:dev
kind: stack
interface:
  adapter: myengine
  transport: http
binding:
  kind: source
  root: /work/myengine
launch:
  command: "cargo run -- --bind 127.0.0.1:{port}"
  requires: [Cargo.toml]
network:
  port: 8080
  readiness: {path: /health}
components:
  llm: {provider: regex}
"""


def _source_target(tmp_path, monkeypatch):
    target_dir = tmp_path / "targets"
    target_dir.mkdir()
    (target_dir / "myengine-dev.yaml").write_text(SOURCE_TARGET, encoding="utf-8")
    monkeypatch.setenv("MEMRANK_CONFIG_DIR", str(tmp_path))


def test_ls_lists_every_target():
    result = runner.invoke(app, ["targets", "ls"])
    assert result.exit_code == 0
    assert "hindsight:matched" in result.stdout
    assert "word-overlap" in result.stdout


def test_show_json_emits_the_resolved_manifest():
    withheld.require("mem0")
    result = runner.invoke(app, ["targets", "show", "mem0", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["components"]["embedder"]["model"] == "text-embedding-3-small"
    assert payload["components"]["embedder"]["dims"] == 1536


def test_show_echoes_the_canonical_ref():
    result = runner.invoke(app, ["targets", "show", "hindsight:matched"])
    assert result.exit_code == 0
    assert "hindsight:matched" in result.stdout


def test_show_json_emits_one_complete_named_source_target(tmp_path, monkeypatch):
    _source_target(tmp_path, monkeypatch)

    result = runner.invoke(app, ["targets", "show", "myengine:dev", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["binding"] == {"kind": "source", "root": "/work/myengine"}
    assert payload["launch"]["requires"] == ["Cargo.toml"]
    assert "engine" not in payload and "workspace" not in payload


def test_retired_submit_source_points_to_named_target(tmp_path):
    result = runner.invoke(
        app, ["submit", "myengine", "demo", "--source", str(tmp_path), "--on", "local"])

    assert result.exit_code != 0
    assert "--source is retired" in result.output
    assert "binding.root" in result.output


def test_show_applies_overrides():
    withheld.require("mem0")
    result = runner.invoke(app, ["targets", "show", "mem0", "--json",
                                 "embedder=voyage/voyage-4-large", "embedder.dims=1024"])
    assert result.exit_code == 0
    assert json.loads(result.stdout)["components"]["embedder"]["dims"] == 1024


def test_show_unknown_target_exits_nonzero():
    assert runner.invoke(app, ["targets", "show", "nope"]).exit_code != 0


def test_show_embedder_override_without_dims_exits_nonzero():
    """The dims trap must be a loud CLI failure, not a plausible-looking manifest.

    Guarded even though an absent `mem0` also exits nonzero: that exit is "unknown target", so
    without the guard this passes on a tree where the dims trap is never reached at all."""
    withheld.require("mem0")
    result = runner.invoke(app, ["targets", "show", "mem0", "embedder=voyage/voyage-4-large"])
    assert result.exit_code != 0


def test_bad_ref_reports_a_message_not_a_traceback():
    """AGENTS.md asks for an actionable message; a Rich traceback is not one."""
    result = runner.invoke(app, ["targets", "show", "nope"])
    assert result.exit_code == 1
    assert "unknown target" in result.output
    assert "Traceback" not in result.output


def test_show_marks_which_required_secrets_are_present(monkeypatch):
    """What `memrank preflight <ref>` used to answer, from the plane that owns the fact: a
    target declares what it needs, so the catalog is where "do I have it" is read."""
    monkeypatch.setenv("OPENAI_API_KEY", "x")

    withheld.require("mem0")
    result = runner.invoke(app, ["targets", "show", "mem0"])

    assert result.exit_code == 0
    assert "OPENAI_API_KEY" in result.stdout
    assert "✔" in result.stdout


def test_show_marks_a_missing_secret_without_failing(tmp_path, monkeypatch):
    """`show` answers a question about the target, so a missing key is part of the ANSWER, not
    an error -- the run that needs it is what refuses, naming the same key."""
    monkeypatch.setenv("MEMRANK_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    withheld.require("mem0")
    result = runner.invoke(app, ["targets", "show", "mem0"])

    assert result.exit_code == 0
    assert "✘" in result.stdout
