import pytest

from memrank.judging.judge import (
    JudgeConfig,
    JudgeVerdict,
    _vote,
    generate_answer,
    judge_answer,
    judge_query,
    judge_sufficiency,
    parse_verdict,
)
from memrank.judging.prompts import (
    MEMRANK_READER,
)


def test_parse_verdict_strict_json():
    v = parse_verdict('{"passed": true, "rationale": "supported"}')
    assert v.passed is True and v.rationale == "supported"


def test_parse_verdict_malformed_raises():
    with pytest.raises(ValueError):
        parse_verdict("totally not json")


def test_parse_verdict_missing_field_raises():
    with pytest.raises(ValueError):
        parse_verdict('{"passed": true}')


def test_vote_majority_passed():
    vs = [JudgeVerdict(True, "a"), JudgeVerdict(False, "b"), JudgeVerdict(True, "c")]
    out = _vote(vs)
    assert out.passed is True and out.samples == 3


def test_config_defaults():
    cfg = JudgeConfig()
    assert cfg.answer_model == "claude-sonnet-4-6"
    assert cfg.judge_model == "claude-opus-4-8"
    assert cfg.no_context_control is True


def _scripted_completer(script):
    """Return a completer that responds based on the system prompt it receives."""
    calls = []

    def complete(model, system, user):
        calls.append((model, system, user))
        for key, resp in script.items():
            if key in system:
                return resp
        raise AssertionError(f"no scripted response for system: {system[:40]}")

    complete.calls = calls  # type: ignore[attr-defined]
    return complete


def test_generate_answer_passes_through_text():
    c = _scripted_completer({"answer the user's question": "the blue whale"})
    assert generate_answer(c, question="q", context="ctx", model="m",
                           prompt=MEMRANK_READER) == "the blue whale"


def test_judge_sufficiency_parses_verdict():
    c = _scripted_completer({"grade whether the provided memory": '{"passed": true, "rationale": "ok"}'})
    v = judge_sufficiency(c, question="q", context="ctx", gold="g", model="m", samples=1)
    assert v.passed is True


def test_judge_answer_negative_uses_negative_system():
    c = _scripted_completer({"should NOT answer affirmatively": '{"passed": true, "rationale": "abstained"}'})
    v = judge_answer(c, question="q", answer="I don't know", gold="g", model="m",
                     samples=1, negative=True)
    assert v.passed is True


def test_judge_query_positive_runs_all_steps_and_control():
    c = _scripted_completer({
        "answer the user's question": "vegetarian sushi",
        "grade whether the provided memory": '{"passed": true, "rationale": "s"}',
        "matches the reference answer": '{"passed": true, "rationale": "c"}',
    })
    cfg = JudgeConfig(completer=c, samples=1)
    jq = judge_query(c, question="q", context="ctx", gold="vegetarian sushi", cfg=cfg, negative=False)
    assert jq.sufficiency is not None and jq.correctness.passed is True
    assert jq.generated_answer == "vegetarian sushi"
    assert jq.answered_without_context is True  # no-context answer also judged correct


def test_judge_query_negative_skips_sufficiency():
    c = _scripted_completer({
        "answer the user's question": "I don't know",
        "should NOT answer affirmatively": '{"passed": true, "rationale": "ok"}',
    })
    cfg = JudgeConfig(completer=c)
    jq = judge_query(c, question="q", context="ctx", gold="no peanuts", cfg=cfg, negative=True)
    assert jq.sufficiency is None and jq.correctness.passed is True


def test_parse_verdict_rejects_string_bool():
    with pytest.raises(ValueError):
        parse_verdict('{"passed": "false", "rationale": "x"}')


def test_parse_verdict_rejects_nonstring_rationale():
    with pytest.raises(ValueError):
        parse_verdict('{"passed": true, "rationale": 5}')


def test_samples_propagate_through_grading():
    # 3 sufficiency samples: 2 pass, 1 fail -> majority pass, samples recorded
    responses = iter(['{"passed": true, "rationale": "a"}',
                      '{"passed": false, "rationale": "b"}',
                      '{"passed": true, "rationale": "c"}'])

    def complete(model, system, user):
        return next(responses)

    v = judge_sufficiency(complete, question="q", context="c", gold="g", model="m", samples=3)
    assert v.passed is True and v.samples == 3
