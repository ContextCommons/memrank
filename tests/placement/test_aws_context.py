"""Locating the AWS context file.

`scripts/internal/aws-context.sh` writes `.aws-context.json` in the working directory. Requiring an
environment variable to point at a file we told you to create at that exact path is ceremony, not
safety -- and it blocked two real cloud runs.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from memrank import config
from memrank.config import DEFAULT_AWS_CONTEXT, ConfigError, aws_context

CONTEXT = {
    "generated_at": "2026-07-30T00:00:00+00:00", "region": "us-east-1",
    "cluster": "c", "subnet": "s", "security_group": "sg", "log_group": "lg",
    "execution_role_arn": "e", "task_role_arn": "t", "artifact_bucket": "b",
    "runner_repository": "rr", "engines_repository": "er",
    "secret_arns": {"ANTHROPIC_API_KEY": "arn:a"},
}


@pytest.fixture(autouse=True)
def _clean(monkeypatch, tmp_path):
    monkeypatch.delenv("MEMRANK_AWS_CONTEXT", raising=False)
    monkeypatch.chdir(tmp_path)


def test_the_default_file_is_found_without_any_env_var(tmp_path):
    (tmp_path / DEFAULT_AWS_CONTEXT).write_text(json.dumps(CONTEXT))
    assert aws_context()["cluster"] == "c"


def test_an_explicit_path_still_wins(tmp_path, monkeypatch):
    (tmp_path / DEFAULT_AWS_CONTEXT).write_text(json.dumps(CONTEXT))
    other = tmp_path / "staging.json"
    other.write_text(json.dumps({**CONTEXT, "cluster": "staging"}))
    monkeypatch.setenv("MEMRANK_AWS_CONTEXT", str(other))
    assert aws_context()["cluster"] == "staging"


def test_no_file_anywhere_says_how_to_make_one():
    """The FILE is the contract, so the message states its keys rather than a generator.

    The generator is operator tooling under `scripts/`, which does not ship; an installed copy
    that was sent there had nothing to run (ATO-2125). Every required key, named, is what an
    operator running their own ECS can actually act on.
    """
    with pytest.raises(ConfigError) as exc:
        aws_context()
    assert DEFAULT_AWS_CONTEXT in str(exc.value)
    for key in ("cluster", "subnet", "execution_role_arn", "secret_arns"):
        assert key in str(exc.value)


def test_an_explicit_path_that_does_not_exist_is_not_silently_replaced(tmp_path, monkeypatch):
    """Falling back to the default would run against different infrastructure than asked for."""
    (tmp_path / DEFAULT_AWS_CONTEXT).write_text(json.dumps(CONTEXT))
    monkeypatch.setenv("MEMRANK_AWS_CONTEXT", str(tmp_path / "typo.json"))
    with pytest.raises(ConfigError, match="typo.json"):
        aws_context()


def test_a_context_missing_a_field_names_it(tmp_path):
    partial = {k: v for k, v in CONTEXT.items() if k != "subnet"}
    (tmp_path / DEFAULT_AWS_CONTEXT).write_text(json.dumps(partial))
    with pytest.raises(ConfigError, match="subnet"):
        aws_context()


def test_the_default_matches_what_the_script_writes():
    """If the script's filename changes, this fails rather than the default going quietly stale.

    Skipped where the generator is absent rather than asserted unconditionally: it is operator
    tooling under ``scripts/internal/`` and does not exist in the published projection, where an
    unconditional read would fail a clean public checkout over a file that side never had.
    """
    script = Path(config.__file__).parents[1] / "scripts" / "internal" / "aws-context.sh"
    if not script.exists():
        pytest.skip("operator generator not present (public projection)")
    assert DEFAULT_AWS_CONTEXT in script.read_text(encoding="utf-8")
