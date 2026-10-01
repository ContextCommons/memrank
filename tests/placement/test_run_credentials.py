"""Which credentials a run must have BEFORE it starts, and whose business they are.

The distinction is who launches the engine. When memrank launches it (`--on local`) memrank must
hold the engine's keys, because it injects them into the container. When the engine is already
running (`--on none`) or ECS resolves them from SSM into the engine's own container (`--on cloud`),
the engine already has its credentials and the harness needs none of them.

Getting this wrong killed the first mem0 cloud run: the task ran `memrank submit mem0` with
`--on` stripped, defaulted to `none`, and demanded ANTHROPIC_API_KEY -- a key ECS had already
injected into the *engine* container, and which the harness never uses.

  error: missing credential(s) ANTHROPIC_API_KEY, and there is no terminal to ask on.
"""
from __future__ import annotations

from memrank.judging.client import JUDGE_SECRET
from memrank.runner import run_credentials
from tests import withheld


def test_provisioning_needs_the_engine_key():
    """--on local: memrank starts the engine, so it must hold what the engine needs."""
    withheld.require("mem0")
    assert run_credentials(["mem0"], judged=False, provisioning=True) == \
        ["OPENAI_API_KEY"]        # mem0's own configuration: OpenAI for both roles


def test_not_provisioning_needs_nothing_from_the_engine():
    """--on none / --on cloud: the engine already has its own credentials.

    Guarded despite passing either way: an absent `mem0` also contributes nothing, so without
    this the empty list would stop meaning what the name says."""
    withheld.require("mem0")
    assert run_credentials(["mem0"], judged=False, provisioning=False) == []


def test_the_judge_key_is_needed_either_way():
    """The judge is the HARNESS making calls, so it is memrank's requirement wherever it runs."""
    assert run_credentials(["word-overlap"], judged=True, provisioning=False) == [JUDGE_SECRET]
    assert run_credentials(["word-overlap"], judged=True, provisioning=True) == [JUDGE_SECRET]


def test_a_judged_provisioned_run_needs_both():
    withheld.require("mem0")
    got = run_credentials(["mem0"], judged=True, provisioning=True)
    assert JUDGE_SECRET in got
    assert "OPENAI_API_KEY" in got


def test_an_unknown_ref_contributes_nothing():
    """A legacy --adapter name with no manifest must not crash the preflight."""
    assert run_credentials(["not-a-target"], judged=False, provisioning=True) == []
