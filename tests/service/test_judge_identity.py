"""A recorded judge is named in full, and one recorded by model alone is never given a provider."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from memrank.service.protocol import Identity, JudgeIdentity
from tests.service.test_scoring import recorded


def _old_identity(judge_model: str | None) -> dict:
    """An identity as a run folder written before the judge was named in full holds it."""
    identity = recorded().identity.model_dump(mode="json")
    for name in ("judge", "judge_provider", "judge_endpoint", "judge_settings", "judge_source"):
        identity.pop(name)
    return {**identity, "judge_model": judge_model}


def test_an_old_record_reads_as_an_unknown_provider_never_anthropic():
    identity = Identity.model_validate(_old_identity("claude-haiku-4-5"))
    assert (identity.judge, identity.judge_provider, identity.judge_source) == (
        "unknown:claude-haiku-4-5", None, "backfill:unverified")
    assert identity.judge_label() == "claude-haiku-4-5 (provider not recorded)"


def test_an_old_record_no_judge_model_graded_names_no_judge():
    identity = Identity.model_validate(_old_identity(None))
    assert identity.judge is None and identity.judge_label() is None


@pytest.mark.parametrize(("judge", "label"), [
    (JudgeIdentity(judge="openai:gpt-5", judge_provider="openai", judge_model="gpt-5"),
     "openai:gpt-5"),
    (JudgeIdentity(judge="router:bedrock/claude-haiku-4-5", judge_provider="openai",
                   judge_model="bedrock/claude-haiku-4-5",
                   judge_source="backfill:translator-log"),
     "router:bedrock/claude-haiku-4-5 (attributed from the translator log)"),
], ids=["run", "translator-backfill"])
def test_each_judge_is_labelled_by_how_it_is_known(judge, label):
    assert judge.label() == label


@pytest.mark.parametrize("fields", [
    {"judge": "anthropic:x", "judge_provider": "anthropic", "judge_model": "y"},
    {"judge": "x", "judge_provider": "anthropic", "judge_model": "x"},
    {"judge": "unknown:x", "judge_provider": "anthropic", "judge_model": "x"},
    {"judge": "anthropic:x", "judge_provider": None, "judge_model": "x"},
], ids=["model-differs", "no-provider", "unknown-with-provider", "provider-missing"])
def test_a_judge_that_contradicts_itself_is_refused(fields):
    with pytest.raises(ValidationError):
        JudgeIdentity(**fields)


def test_an_identity_with_a_model_but_no_judge_is_refused():
    identity = recorded().identity.model_dump(mode="json")
    with pytest.raises(ValidationError, match="no judge"):
        Identity.model_validate({**identity, "judge": None, "judge_model": "x"})
