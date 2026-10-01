"""A judge that fences its JSON, and a run that must not finish quietly when nothing graded.

ATO-1885. `claude-haiku-4-5` wraps every verdict in a markdown code fence, in both judge roles,
deterministically -- so `parse_verdict`'s bare-JSON-only reading made three billed attempts fail
per grading and left the whole arm at `judged_coverage` 0.0. The run then COMPLETED with a
normal-looking receipt, because the only signal on that path is `observer.warning`, which is a
no-op under the library default `NULL_OBSERVER`.

Two defects, one observable. These tests pin both: a fenced object is a reply the parser reads,
and a run where nothing at all could be graded raises instead of returning a result.
"""
from __future__ import annotations

import pytest

from memrank.judging.judge import JudgeConfig, parse_events, parse_nugget_score, parse_verdict

# Reproduced from the shared judge cache, byte for byte: the fence carries a `json` tag, the
# object sits on its own lines, and there is no trailing newline after the closing delimiter.
FENCED_VERDICT = '```json\n{"passed": false, "rationale": "The candidate omitted the date."}\n```'
BARE_VERDICT = '{"passed": true, "rationale": "ok"}'


def test_a_fenced_verdict_parses():
    """The defect itself: this reply is a complete, unambiguous verdict, and it was unreadable."""
    verdict = parse_verdict(FENCED_VERDICT)

    assert verdict.passed is False
    assert verdict.rationale == "The candidate omitted the date."


def test_a_bare_verdict_still_parses_unchanged():
    """The incumbent judges return bare JSON; accepting a fence must not cost that."""
    assert parse_verdict(BARE_VERDICT).passed is True


@pytest.mark.parametrize("text", [
    '```\n{"passed": true, "rationale": "ok"}\n```',      # no language tag
    '```json\n{"passed": true, "rationale": "ok"}\n```\n',  # trailing newline
    '  ```json\n{"passed": true, "rationale": "ok"}\n```  ',  # surrounding whitespace
])
def test_the_fence_variants_a_model_actually_emits(text):
    assert parse_verdict(text).passed is True


@pytest.mark.parametrize("text", [
    '```json\n{"passed": true, "rationale": "ok"}',        # opened, never closed
    '```python\n{"passed": true, "rationale": "ok"}\n```',  # not a JSON block
    '```json{"passed": true, "rationale": "ok"}```',        # no delimiter newline
    'Here is my verdict:\n```json\n{"passed": true, "rationale": "ok"}\n```',  # prose around it
])
def test_anything_that_is_not_exactly_one_json_fence_still_raises(text):
    """Not a fallback chain: a reply is bare JSON or one whole JSON fence, and nothing else is
    salvaged. A model that narrates around its verdict is malformed and must be re-asked."""
    with pytest.raises(ValueError):
        parse_verdict(text)


def test_the_other_two_judge_parsers_read_a_fence_too():
    """Same judge model, same habit: a rubric grading and an event extraction fence exactly as a
    verdict does, so fixing one parser and not its neighbours leaves BEAM failing identically."""
    assert parse_nugget_score('```json\n{"score": 0.5, "rationale": "partial"}\n```') == (
        0.5, "partial")
    assert parse_events('```json\n{"events": ["a", "b"]}\n```') == ["a", "b"]


# --- the second defect: a run where nothing graded must not complete ---------------------------- #

def _unit(n: int):
    from types import SimpleNamespace
    return SimpleNamespace(queries=[
        {"id": f"q{i}", "text": f"q{i}", "gold_answers": ["g"], "category": "single-hop"}
        for i in range(n)])


def test_a_run_that_judged_nothing_raises_rather_than_returning_a_receipt():
    """The reachable state the ticket names: every grading unparseable, every call billed, and a
    completed run whose receipt looks normal at `judged_coverage` 0.0. No observer is wired here,
    which is the library default and exactly where the warning went nowhere."""
    from memrank.evaluation.judge_stage import AllVerdictsUnparseable, _apply_judge

    rows = [{"query_id": f"q{i}", "retrieved": [{"content": "c"}]} for i in range(2)]

    with pytest.raises(AllVerdictsUnparseable, match="claude-opus-4-8"):
        _apply_judge([_unit(2)], rows, JudgeConfig(no_context_control=False),
                     lambda m, s, u: "not json at all")


def test_one_survivor_keeps_the_run_alive():
    """The gate is 'nothing graded', not 'anything failed'. One unparseable reply in a cell that
    otherwise graded fine must still cost coverage rather than the run -- that is the incident
    `_judge_one_query` was written for and it must not be undone here."""
    from memrank.evaluation.judge_stage import _apply_judge

    def complete(model: str, system: str, user: str) -> str:
        return "not json at all" if "q1" in user else BARE_VERDICT

    rows = [{"query_id": f"q{i}", "retrieved": [{"content": "c"}]} for i in range(2)]
    metrics = _apply_judge([_unit(2)], rows, JudgeConfig(no_context_control=False), complete)

    assert metrics["n_judged"] == 1
    assert metrics["n_unjudged_unparseable_verdict"] == 1
