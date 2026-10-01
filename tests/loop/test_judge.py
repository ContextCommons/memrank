"""The runner judges with the key it is given, through the benchmark's own judge shape."""

from __future__ import annotations

from memrank.judging.judge import JudgeConfig
from memrank.judging.prompts import JUDGE_PROMPT_VERSION
from memrank.loop import judge as loop_judge
from tests.service.conftest import PASS
from tests.service.test_scoring import recorded


def test_the_orgs_key_reaches_the_anthropic_client_and_every_answer_is_judged(monkeypatch):
    keys: list[str | None] = []

    def anthropic(cfg, *, api_key=None):
        keys.append(api_key)
        return lambda model, system, user: PASS

    monkeypatch.setattr(loop_judge, "anthropic_completer", anthropic)
    scored = loop_judge.judge(recorded(), "org-key", JudgeConfig(cache=False))
    assert keys == ["org-key"] and scored.judged
    assert [q.status for c in scored.cases for q in c.questions] == \
        ["judged", "judged", "judged", "failed"]
    assert scored.identity.judge_model == JudgeConfig.judge_model  # the config's, when given
    # with the model, the prompts' version names the judging (decision 0040)
    assert scored.identity.judge_prompt_version == JUDGE_PROMPT_VERSION


def test_without_a_config_the_judge_is_haiku(monkeypatch, tmp_path):
    # No config means the judge cache is on: keep it out of the real ~/.memrank.
    monkeypatch.setenv("MEMRANK_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(loop_judge, "anthropic_completer",
                        lambda cfg, *, api_key=None: lambda model, system, user: PASS)
    scored = loop_judge.judge(recorded(), "org-key")
    assert scored.identity.judge_model == loop_judge.DEFAULT_JUDGE_MODEL == "claude-haiku-4-5"
