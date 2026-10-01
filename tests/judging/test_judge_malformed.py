"""A judge that emits invalid JSON must cost one query, not a whole cell.

`supermemory x locomo --slice mini` ingested three units, retrieved all 385 queries, spent four
minutes judging, and then exited 1 on a single reply:

    {"passed": false, "rationale": "The candidate refers to passing interviews "last Friday"
     without providing the specific date (Friday before 15 July 2023) that the reference requires."}

The judge quoted the candidate verbatim, so raw double quotes landed inside its own JSON string.
`parse_verdict` raised, `_graded` called it inside a list comprehension, nothing caught it, and
~1,900 paid gradings were discarded along with every retrieval behind them.

The wording that triggers it comes from the benchmark data, not from any engine, so every cell is
exposed. These tests pin the two properties that matter: a transient malformation is retried, and a
persistent one costs exactly one query.
"""
from __future__ import annotations

import pytest

from memrank.judging.judge import (
    JudgeConfig,
    UnparseableVerdict,
    _graded,
    judge_query,
    parse_verdict,
)

# The real payload, reproduced exactly -- unescaped quotes around `last Friday`.
MALFORMED = ('{"passed": false, "rationale": "The candidate refers to passing interviews '
             '"last Friday" without providing the specific date that the reference requires."}')
VALID = '{"passed": true, "rationale": "ok"}'


def _scripted(*replies: str):
    """A completer returning each reply in turn, then repeating the last one forever."""
    calls: list[str] = []

    def complete(model: str, system: str, user: str) -> str:
        calls.append(user)
        return replies[min(len(calls) - 1, len(replies) - 1)]
    return complete, calls


def test_the_real_payload_is_genuinely_unparseable():
    """Guards the fixture: if this ever parses, these tests are asserting nothing."""
    with pytest.raises(ValueError, match="did not return JSON"):
        parse_verdict(MALFORMED)


def test_a_malformed_reply_is_retried_and_the_verdict_survives():
    complete, calls = _scripted(MALFORMED, VALID)

    verdict = _graded(complete, "m", "sys", "user", samples=1)

    assert verdict.passed is True
    assert len(calls) == 2, "the malformed reply should have been re-asked exactly once"


def test_the_retry_changes_the_prompt_or_the_cache_would_replay_the_bad_reply():
    """`cached_completer` keys on the prompt. Re-asking identical text returns the same broken
    response from cache forever, so the retry must vary the prompt to be a retry at all."""
    complete, calls = _scripted(MALFORMED, VALID)

    _graded(complete, "m", "sys", "user", samples=1)

    assert calls[0] != calls[1]


def test_a_reply_that_parses_is_never_re_asked():
    """Retrying a parseable verdict would be verdict-shopping, and would inflate the call count
    the preflight promised."""
    complete, calls = _scripted(VALID)

    _graded(complete, "m", "sys", "user", samples=1)

    assert len(calls) == 1


def test_a_persistently_broken_judge_raises_a_typed_error():
    """Typed so the runner can catch exactly this and skip one query. A bare ValueError would be
    indistinguishable from a genuine programming error and should not be swallowed."""
    complete, calls = _scripted(MALFORMED)

    with pytest.raises(UnparseableVerdict):
        _graded(complete, "m", "sys", "user", samples=1)

    assert len(calls) == 3, "bounded: it must give up rather than retry forever"


def test_retries_are_billed_because_they_are_real_calls():
    """The cap counter wraps the completer beneath this, so a retry must reach it. Otherwise a
    run could exceed the budget the judge preflight quoted."""
    complete, calls = _scripted(MALFORMED, MALFORMED, VALID)

    _graded(complete, "m", "sys", "user", samples=1)

    assert len(calls) == 3


def test_judge_query_propagates_it_so_the_caller_decides():
    """`judge_query` grades up to three things; if any is ungradeable the query is unjudged. The
    decision to skip belongs to the runner, which owns the coverage metric."""
    complete, _ = _scripted(MALFORMED)

    with pytest.raises(UnparseableVerdict):
        judge_query(complete, question="q", context="c", gold="g",
                    cfg=JudgeConfig(no_context_control=False), negative=True)


# --- the runner: one bad query must not end the cell ------------------------------------------- #

def _unit(texts: list[str]):
    """A unit whose questions are distinguishable in the prompt, so a stub can fail exactly one."""
    from types import SimpleNamespace
    return SimpleNamespace(queries=[
        {"id": f"q{i}", "text": t, "gold_answers": ["g"], "category": "single-hop"}
        for i, t in enumerate(texts)])


def test_one_ungradeable_query_costs_coverage_not_the_run():
    """The supermemory failure at the level it actually bit: two queries grade, one cannot, and
    the cell still produces a result with the loss visible in the metrics rather than as exit 1."""
    from memrank.runner import _apply_judge

    def complete(model: str, system: str, user: str) -> str:
        return MALFORMED if "POISON" in user else VALID

    unit = _unit(["fine one", "POISON", "fine two"])
    rows = [{"query_id": f"q{i}", "retrieved": [{"content": "c"}]} for i in range(3)]
    metrics = _apply_judge([unit], rows, JudgeConfig(no_context_control=False), complete)

    assert metrics["n_judged"] == 2
    assert metrics["n_unjudged_unparseable_verdict"] == 1
    assert metrics["judged_coverage"] == pytest.approx(2 / 3)
    # The two that graded still contribute a real score -- the point of not discarding the cell.
    assert metrics["answer_correctness"] == 1.0


def test_the_unparseable_count_has_its_own_reason():
    """A benchmark gap and a judge malfunction are different problems; only one is worth
    re-running to fix, so they must not share a counter.

    Two queries with one survivor, where this once used a single query that failed. A cell where
    NOTHING grades now raises instead of returning these metrics (ATO-1885,
    `test_judge_fenced_verdict.py`), so the counter has to be read off a cell that lived.
    """
    from memrank.runner import _apply_judge

    def complete(model: str, system: str, user: str) -> str:
        return MALFORMED if "POISON" in user else VALID

    rows = [{"query_id": f"q{i}", "retrieved": [{"content": "c"}]} for i in range(2)]
    metrics = _apply_judge([_unit(["POISON", "fine"])], rows,
                           JudgeConfig(no_context_control=False), complete)

    assert metrics["n_unjudged_unparseable_verdict"] == 1
    assert metrics["n_unjudged"] == 1
    assert metrics["n_unjudged_no_gold"] == 0
    assert metrics["judged_coverage"] == pytest.approx(1 / 2)
    assert metrics["n_judged"] == 1
