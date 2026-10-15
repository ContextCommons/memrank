"""The runner judges with the key it is given, through the benchmark's own judge shape."""

from __future__ import annotations

from memrank.judging.judge import JUDGE, JudgeConfig
from memrank.judging.prompts import JUDGE_PROMPT_VERSION
from memrank.judging.providers import ADAPTERS, Endpoint
from memrank.loop import judge as loop_judge
from tests.service.conftest import PASS
from tests.service.test_scoring import recorded


def test_the_orgs_key_reaches_the_anthropic_client_and_every_answer_is_judged(monkeypatch):
    keys: list[str | None] = []

    def anthropic(target, api_key, *, max_tokens):
        keys.append(api_key)
        return lambda model, system, user, sampling: PASS

    monkeypatch.setitem(ADAPTERS, "anthropic", anthropic)
    scored = loop_judge.judge(recorded(), "org-key", JudgeConfig(cache=False))
    assert keys == ["org-key"] and scored.judged
    assert [q.status for c in scored.cases for q in c.questions] == \
        ["judged", "judged", "judged", "failed"]
    assert scored.identity.judge_model == JudgeConfig().model(JUDGE)  # the config's, when given
    # with the model, the prompts' version names the judging (decision 0040)
    assert scored.identity.judge_prompt_version == JUDGE_PROMPT_VERSION


def test_without_a_config_the_judge_is_haiku(monkeypatch, tmp_path):
    # No config means the judge cache is on: keep it out of the real ~/.memrank.
    monkeypatch.setenv("MEMRANK_CACHE_DIR", str(tmp_path))
    monkeypatch.setitem(ADAPTERS, "anthropic",
                        lambda target, api_key, *, max_tokens: lambda m, s, u, sampling: PASS)
    scored = loop_judge.judge(recorded(), "org-key")
    assert scored.identity.judge == loop_judge.DEFAULT_JUDGE == "anthropic:claude-haiku-4-5"


def test_a_run_judged_by_your_own_completer_records_custom_and_its_name():
    """It used to record the default Claude model, which a custom completer never called."""
    cfg = JudgeConfig(completer=lambda model, system, user: PASS, completer_name="mine",
                      cache=False)
    identity = loop_judge.judge(recorded(), None, cfg).identity
    assert (identity.judge, identity.judge_provider, identity.judge_model) == (
        "custom:mine", "custom", "mine")


def test_a_run_judged_through_an_endpoint_records_its_fingerprint_not_its_url(monkeypatch):
    lab = Endpoint(name="lab", adapter="openai", api_base="https://lab.invalid/v1",
                   key_ref="LAB_KEY")
    keys: list[str] = []

    def openai(target, api_key, *, max_tokens):
        keys.append(api_key)
        return lambda m, s, u, sampling: PASS

    monkeypatch.setitem(ADAPTERS, "openai", openai)
    monkeypatch.delenv("LAB_KEY", raising=False)
    cfg = JudgeConfig(judge="lab:bedrock/claude-haiku-4-5", endpoints=(lab,), cache=False)
    # The key handed over is the org's LAB_KEY, the one the endpoint names: never this machine's.
    identity = loop_judge.judge(recorded(), "org-lab-key", cfg).identity
    assert keys == ["org-lab-key"]
    assert (identity.judge, identity.judge_provider, identity.judge_endpoint) == (
        "lab:bedrock/claude-haiku-4-5", "openai", "local:lab")
    assert "lab.invalid" not in identity.model_dump_json()
