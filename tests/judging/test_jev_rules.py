"""The code that turns Jev's probabilities into verdicts: binary, rubric and ordering."""

from __future__ import annotations

import pytest

from memrank.judging.jev import questions as jq
from memrank.judging.jev import rules
from memrank.judging.judge import UnparseableVerdict
from memrank.metrics.ordering import REVERSED, SAME, UNORDERED, score_relations


def _nouls(**values: float) -> dict:
    return {key: {"type": "noul", "noul": value} for key, value in values.items()}


@pytest.mark.parametrize(("states", "contradicts", "declines", "passed"), [
    (0.9, 0.1, 0.1, True),
    (0.4, 0.1, 0.1, False),   # does not state the reference
    (0.9, 0.6, 0.1, False),   # states it and contradicts it
    (0.9, 0.1, 0.7, False),   # "I don't know" with the facts listed anyway
    (0.5, 0.5, 0.5, True),    # each exactly at the threshold, on the passing side
])
def test_a_binary_answer_passes_only_when_every_condition_does(states, contradicts, declines,
                                                               passed):
    answers = _nouls(states_reference=states, contradicts_reference=contradicts,
                     declines=declines)
    outcome = rules.binary(jq.GENERIC, answers)
    assert outcome.passed is passed
    assert outcome.probability == min(states, 1 - contradicts, 1 - declines)
    assert outcome.passed == (outcome.probability >= rules.THRESHOLD)


def test_an_abstention_question_flips_the_rule_to_declining():
    abstention = jq.LONGMEMEVAL["abstention"]
    assert rules.binary(abstention, _nouls(declines=0.8)).passed is True
    refused = rules.binary(abstention, _nouls(declines=0.2))
    assert refused.passed is False and refused.probability == 0.2
    assert refused.rule == "declines, each at 0.5"


def test_a_later_threshold_is_a_rescoring_of_the_same_probability():
    answers = _nouls(states_reference=0.7, contradicts_reference=0.2, declines=0.1)
    assert rules.binary(jq.GENERIC, answers).passed is True
    assert rules.binary(jq.GENERIC, answers, threshold=0.75).passed is False


@pytest.mark.parametrize("answers", [
    {}, _nouls(states_reference=1.2, contradicts_reference=0.0, declines=0.0),
    {"states_reference": {"type": "choice", "choice": "x"}},
])
def test_a_missing_or_malformed_answer_is_unparseable_not_a_verdict(answers):
    with pytest.raises(UnparseableVerdict):
        rules.binary(jq.GENERIC, answers)


@pytest.mark.parametrize(("expected", "score"), [
    (0.0, 0.0), (0.49, 0.0), (0.5, 0.5), (1.0, 0.5), (1.49, 0.5), (1.5, 1.0), (1.63, 1.0),
    (2.0, 1.0),
])
def test_a_rubric_score_maps_its_rounded_expected_level_to_beams_scale(expected, score):
    answers = {"criterion_0": {"type": "score", "score": expected,
                               "probabilities": {"0": 0.1, "1": 0.3, "2": 0.6}}}
    [outcome] = rules.rubric(["names the city"], answers)
    assert outcome.score == score
    assert outcome.expected_level == expected
    assert outcome.probabilities == {"0": 0.1, "1": 0.3, "2": 0.6}


def test_score_relations_counts_coverage_order_and_extras():
    # Four reference events, three asserted; pairs: (0,1) same, (0,3) reversed, (1,3) unordered.
    relations = {(0, 1): SAME, (0, 3): REVERSED, (1, 3): UNORDERED}
    scored = score_relations(4, [0, 1, 3], relations, extras=False)
    assert scored.recall == 0.75 and scored.precision == 1.0
    assert scored.f1 == pytest.approx(2 * 0.75 / 1.75)
    # tau_b = (C - D) / sqrt(n0 * (n0 - T)) = (1 - 1) / sqrt(3 * 2) = 0
    assert scored.tau_norm == 0.5
    assert scored.score == pytest.approx(0.5 * scored.f1)


def test_a_perfect_answer_scores_one_and_extras_cap_precision():
    relations = {(0, 1): SAME, (0, 2): SAME, (1, 2): SAME}
    assert score_relations(3, [0, 1, 2], relations, extras=False).score == 1.0
    padded = score_relations(3, [0, 1, 2], relations, extras=True)
    assert padded.precision == 0.75  # three found, at least one more listed: 3 / (3 + 1)
    assert padded.tau_norm == 1.0 and padded.score < 1.0


def test_a_reversed_answer_scores_zero_order_and_nothing_found_scores_zero():
    relations = {(0, 1): REVERSED, (0, 2): REVERSED, (1, 2): REVERSED}
    assert score_relations(3, [0, 1, 2], relations, extras=False).tau_norm == 0.0
    assert score_relations(3, [], {}, extras=True).score == 0.0
    single = score_relations(3, [1], {}, extras=False)
    assert single.tau_norm == 0.5  # no pair to order: tau-b undefined, recorded as 0


def test_ordering_reads_alternated_pairs_against_the_reference_order():
    # Pair (0, 1) is shown as (b, a): event_a is reference event 1.
    assert jq.pair_events(0, 1) == (1, 0) and jq.pair_events(0, 2) == (0, 2)
    answers = {**_nouls(mentions_0=0.9, mentions_1=0.8, mentions_2=0.1, lists_other_events=0.2),
               "order_0_1": {"type": "choice", "choice": "b_first"}}
    outcome = rules.ordering(3, answers)
    assert outcome.present == [0, 1] and outcome.relations == {"0,1": SAME}
    assert outcome.extras is False
    assert outcome.scored.tau_norm == 1.0 and outcome.scored.recall == pytest.approx(2 / 3)
