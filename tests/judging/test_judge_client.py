
from memrank.judging.client import (
    build_completer,
    cached_completer,
    sampling_params,
)
from memrank.judging.judge import JUDGE, READER, JudgeConfig
from memrank.judging.providers import resolve
from memrank.judging.providers.anthropic import anthropic_send


def _counting_inner():
    calls = {"n": 0}

    def inner(model, system, user):
        calls["n"] += 1
        return f'{{"passed": true, "rationale": "{calls["n"]}"}}'

    return inner, calls


def test_cache_calls_inner_once_per_unique_key(tmp_path):
    inner, calls = _counting_inner()
    c = cached_completer(inner, cache_dir=tmp_path, prompt_version="v1")
    a = c("m", "sys", "user-A")
    b = c("m", "sys", "user-A")   # cache hit
    c("m", "sys", "user-B")       # miss
    assert a == b
    assert calls["n"] == 2  # A once, B once


def test_cache_persists_response_not_prompt(tmp_path):
    inner, _ = _counting_inner()
    c = cached_completer(inner, cache_dir=tmp_path, prompt_version="v1")
    c("m", "sys", "secret memory content")
    blobs = "\n".join(p.read_text() for p in tmp_path.iterdir())
    assert "secret memory content" not in blobs  # prompt body never written


def test_every_call_goes_out_and_is_counted():
    """This asserted a cap that refused the third call. There is no ceiling any more -- capping the
    judge would have made which queries got graded depend on where a counter ran out -- so what is
    left to check is that the counter reports every billable call, since it becomes the receipt's
    `judge_calls_made`."""
    inner, _ = _counting_inner()
    cfg = JudgeConfig(completer_name="fake", completer=inner, cache=False)
    complete, counter = build_completer(cfg)
    complete(JUDGE, "s", "u1")
    complete(JUDGE, "s", "u2")
    complete(READER, "s", "u3")
    assert counter() == 3


def test_answer_calls_are_pinned_and_judge_calls_are_not():
    """The reader runs at temperature 0 (the reproducibility pin BEAM's protocol specifies);
    the judge model rejects the parameter, so its calls must carry NO sampling kwargs -- sending
    one anyway would 400 every judge call at the API."""
    cfg = JudgeConfig()
    assert sampling_params(READER, cfg) == {"temperature": 0.0}
    assert sampling_params(JUDGE, cfg) == {}


def test_the_role_decides_sampling_even_when_reader_and_judge_share_a_model():
    """It used to be model equality, which gave a judge sharing the reader's model a temperature."""
    same = "anthropic:claude-haiku-4-5"
    cfg = JudgeConfig(reader=same, judge=same)
    assert sampling_params(READER, cfg) == {"temperature": 0.0}
    assert sampling_params(JUDGE, cfg) == {}


def test_cache_hits_bypass_cap_and_counter(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMRANK_CACHE_DIR", str(tmp_path))
    inner, calls = _counting_inner()
    # First run populates the cache (cap high enough).
    cfg1 = JudgeConfig(completer_name="fake", completer=inner, cache=True)
    c1, n1 = build_completer(cfg1)
    c1(JUDGE, "s", "u1")
    assert n1() == 1 and calls["n"] == 1
    # Second run with cap=1: the single call is a cache hit -> no egress, no cap trip.
    cfg2 = JudgeConfig(completer_name="fake", completer=inner, cache=True)
    c2, n2 = build_completer(cfg2)
    c2(JUDGE, "s", "u1")  # cache hit
    c2(JUDGE, "s", "u1")  # cache hit again -- would exceed cap=1 if hits counted
    assert n2() == 0 and calls["n"] == 1  # no new egress


def test_the_anthropic_adapter_sends_no_system_prompt_when_there_is_none(monkeypatch):
    """The official readers are one user message; an empty system field is not sent at all."""
    import anthropic

    sent: list[dict] = []

    class Messages:
        def create(self, **kwargs):
            sent.append(kwargs)
            return type("Reply", (), {"content": []})()

    class Client:
        def __init__(self, **_):
            self.messages = Messages()

    monkeypatch.setattr(anthropic, "Anthropic", Client)
    send = anthropic_send(resolve("anthropic:claude-haiku-4-5"), "k", max_tokens=512)
    send("claude-haiku-4-5", "", "the whole prompt", {})
    send("claude-haiku-4-5", "judge", "u", {})
    assert "system" not in sent[0] and sent[1]["system"] == "judge"
