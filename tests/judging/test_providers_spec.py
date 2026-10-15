"""Judge specs, endpoints, cache keys and what a configuration records about who judged."""

from __future__ import annotations

import hashlib

import pytest

from memrank.evaluation.judge_stage import _judge_receipt_config
from memrank.judging.client import build_completer, cache_prefix
from memrank.judging.judge import JUDGE, READER, JudgeConfig
from memrank.judging.prompts import JUDGE_PROMPT_VERSION
from memrank.judging.providers import ADAPTERS, Endpoint, parse_spec, resolve
from memrank.judging.record import receipt_fields

ROUTER = Endpoint(name="router", adapter="openai", api_base="https://router.invalid/v1",
                  key_ref="ROUTER_KEY", fingerprint="0f1e@3")


def test_a_spec_splits_on_the_first_colon_only():
    assert parse_spec("router:bedrock/claude-haiku-4-5") == ("router", "bedrock/claude-haiku-4-5")
    assert parse_spec("router:anthropic.claude-v2:1") == ("router", "anthropic.claude-v2:1")


@pytest.mark.parametrize("spec", ["claude-haiku-4-5", "", ":gpt-5", "openai:"])
def test_a_spec_without_both_sides_is_refused(spec):
    with pytest.raises(ValueError):
        parse_spec(spec)


def test_builtins_and_endpoints_resolve_to_adapter_model_endpoint_and_key():
    builtin = resolve("openai:gpt-5")
    assert (builtin.adapter, builtin.model, builtin.endpoint, builtin.key_name) == (
        "openai", "gpt-5", None, "OPENAI_API_KEY")
    named = resolve("router:bedrock/claude-haiku-4-5", [ROUTER])
    assert (named.adapter, named.model, named.endpoint, named.key_name) == (
        "openai", "bedrock/claude-haiku-4-5", ROUTER, "ROUTER_KEY")
    assert named.max_tokens_field == "max_tokens"
    assert builtin.max_tokens_field == "max_completion_tokens"


def test_an_unknown_provider_is_refused_naming_what_is_known():
    with pytest.raises(ValueError, match="known: anthropic, openai, typesafe, router"):
        resolve("nosuch:model", [ROUTER])


@pytest.mark.parametrize("spec", ["custom:mine", "unknown:claude-haiku-4-5"])
def test_reserved_providers_cannot_be_typed(spec):
    with pytest.raises(ValueError):
        resolve(spec)


def test_typesafe_is_a_builtin_that_decides_rather_than_completes():
    jev = resolve("typesafe:jev-1.14.0")
    assert (jev.adapter, jev.model, jev.endpoint, jev.key_name) == (
        "typesafe", "jev-1.14.0", None, "TYPESAFE_API_KEY")
    assert jev.decides and jev.api_base == "https://api.typesafe.ai"
    assert not resolve("openai:gpt-5").decides


@pytest.mark.parametrize("alias", ["jev-latest", "jev-preview"])
def test_a_moving_jev_alias_is_refused(alias):
    with pytest.raises(ValueError, match="moving alias"):
        resolve(f"typesafe:{alias}")


@pytest.mark.parametrize("name", ["anthropic", "custom", "unknown", "typesafe", "a:b", ""])
def test_an_endpoint_may_not_take_a_reserved_or_malformed_name(name):
    with pytest.raises(ValueError):
        Endpoint(name=name, adapter="openai", api_base="https://x.invalid", key_ref="K")


def test_a_code_endpoint_is_fingerprinted_by_name_and_never_shows_its_url():
    local = Endpoint(name="lab", adapter="openai", api_base="https://private.invalid",
                     key_ref="K")
    assert local.fingerprint == "local:lab"
    assert "private.invalid" not in repr(local)


def test_anthropic_cache_keys_are_byte_identical_to_the_pre_provider_key(tmp_path, monkeypatch):
    """Every cache entry written before providers existed must still hit, so old runs never pay
    again: the key string is pinned here exactly as it was built then."""
    monkeypatch.setenv("MEMRANK_CACHE_DIR", str(tmp_path))
    monkeypatch.setitem(ADAPTERS, "anthropic",
                        lambda target, api_key, *, max_tokens: lambda m, s, u, sampling: "ok")
    cfg = JudgeConfig(judge="anthropic:claude-haiku-4-5")
    complete, _ = build_completer(cfg, keys={"ANTHROPIC_API_KEY": "k"})
    complete(JUDGE, "the system", "the user")
    today = f"{JUDGE_PROMPT_VERSION}|claude-haiku-4-5|512|the system|the user"
    key = hashlib.sha256(today.encode()).hexdigest()
    assert [p.name for p in (tmp_path / "judge-cache").iterdir()] == [f"{key}.txt"]


def test_other_providers_key_their_cache_by_adapter_endpoint_and_fingerprint():
    assert cache_prefix(resolve("anthropic:claude-haiku-4-5")) == ""
    assert cache_prefix(resolve("openai:gpt-5")) == "openai|https://api.openai.com/v1|"
    assert cache_prefix(resolve("router:m", [ROUTER])) == \
        "openai|https://router.invalid/v1|0f1e@3|"


def test_a_custom_completers_cache_is_keyed_by_its_name(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMRANK_CACHE_DIR", str(tmp_path))
    answers = iter(["first", "second"])

    def completer(name):
        cfg = JudgeConfig(completer=lambda m, s, u: next(answers), completer_name=name)
        return build_completer(cfg)[0]

    assert completer("v1")(JUDGE, "s", "u") == "first"
    assert completer("v1")(JUDGE, "s", "u") == "first"  # replayed from the cache
    assert completer("v2")(JUDGE, "s", "u") == "second"  # a new name is a new completer


def test_a_custom_completer_needs_a_name():
    with pytest.raises(ValueError, match="completer_name"):
        JudgeConfig(completer=lambda m, s, u: "x")


def test_a_custom_completer_is_called_with_each_roles_model():
    seen: list[str] = []
    cfg = JudgeConfig(reader="router:bedrock/reader", judge="router:bedrock/judge",
                      completer=lambda m, s, u: seen.append(m) or "x", completer_name="mine",
                      cache=False)
    complete, _ = build_completer(cfg)
    complete(READER, "s", "u")
    complete(JUDGE, "s", "u")
    assert seen == ["bedrock/reader", "bedrock/judge"]


def test_a_completer_is_called_with_a_role_and_refuses_anything_else():
    complete, counter = build_completer(
        JudgeConfig(completer=lambda m, s, u: "x", completer_name="mine", cache=False))
    with pytest.raises(ValueError, match="role"):
        complete("claude-haiku-4-5", "s", "u")
    assert counter() == 0


def test_a_custom_completers_records_name_it_and_never_a_claude_model():
    cfg = JudgeConfig(completer=lambda m, s, u: "x", completer_name="mine")
    fields = receipt_fields(cfg)
    assert fields["judge"] == fields["reader"] == "custom:mine"
    assert fields["judge_provider"] == "custom" and fields["judge_endpoint"] is None
    assert _judge_receipt_config(cfg)["judge"] == "custom:mine"


def test_an_endpoint_judge_is_recorded_by_fingerprint_with_the_settings_sent():
    cfg = JudgeConfig(judge="router:bedrock/claude-haiku-4-5", endpoints=(ROUTER,))
    fields = receipt_fields(cfg)
    assert fields["judge"] == "router:bedrock/claude-haiku-4-5"
    assert (fields["judge_provider"], fields["judge_model"], fields["judge_endpoint"]) == (
        "openai", "bedrock/claude-haiku-4-5", "0f1e@3")
    assert fields["judge_settings"] == {"adapter": "openai", "max_tokens": 512,
                                        "max_tokens_field": "max_tokens"}
    assert fields["reader_settings"]["temperature"] == 0.0
    assert "router.invalid" not in repr(_judge_receipt_config(cfg))


def test_a_config_refuses_an_unresolvable_judge_and_duplicate_endpoints():
    with pytest.raises(ValueError, match="unknown provider"):
        JudgeConfig(judge="router:m")
    with pytest.raises(ValueError, match="unique"):
        JudgeConfig(endpoints=(ROUTER, ROUTER))
