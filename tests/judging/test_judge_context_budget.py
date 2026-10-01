# Copyright 2026 AtomicStrata
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.
"""Is a low score memory failure, or a context budget too small to hold the evidence?

`_judge_one_query` has always written `context_truncated` and `context_tokens_sent` onto each
drill row, and nothing ever aggregated them -- so a run could be context-starved with no figure
in the artifact saying so.

It matters most on LongMemEval, where the arithmetic is stark: sessions average 2,290 tokens, so
`k=10` returns ~22,900 against a 5,000-token cap that keeps about 2.2 of them, while 300 of 470
non-abstention questions need >=2 evidence sessions and 71 need >=3. The cap can therefore starve
the reader of evidence retrieval already found, and that is scored as a memory failure.

The cross-tabulation against evidence count is the discriminator: if truncation concentrates on
multi-evidence questions, the score is measuring ranking precision more than memory.
"""
from memrank.judging.judge import JudgeConfig
from memrank.runner import _judge_metrics


def _metrics(truncation):
    cfg = JudgeConfig(no_context_control=False)
    return _judge_metrics(cfg, [], [1.0] * len(truncation), [], len(truncation), {}, [],
                          len(truncation), 0, truncation=truncation)


def test_truncation_is_absent_when_nothing_was_measured():
    """A run whose rows carry no truncation flag reports None, not a false zero."""
    metrics = _metrics([])

    assert metrics["context_truncation_rate"] is None
    assert metrics["context_truncation_by_evidence_count"] == {}


def test_truncation_rate_is_the_fraction_of_judged_queries_capped():
    metrics = _metrics([
        {"truncated": True, "tokens_sent": 5000, "n_evidence": 3},
        {"truncated": False, "tokens_sent": 1200, "n_evidence": 1},
        {"truncated": True, "tokens_sent": 5000, "n_evidence": 2},
        {"truncated": False, "tokens_sent": 900, "n_evidence": 1},
    ])

    assert metrics["context_truncation_rate"] == 0.5
    assert metrics["context_tokens_sent_mean"] == 3025.0


def test_truncation_is_cross_tabulated_against_evidence_count():
    """The hypothesis under test: failures concentrate where the evidence spans more sessions."""
    metrics = _metrics([
        {"truncated": False, "tokens_sent": 900, "n_evidence": 1},
        {"truncated": False, "tokens_sent": 950, "n_evidence": 1},
        {"truncated": True, "tokens_sent": 5000, "n_evidence": 3},
        {"truncated": True, "tokens_sent": 5000, "n_evidence": 3},
    ])

    cells = metrics["context_truncation_by_evidence_count"]

    assert cells["1"] == {"rate": 0.0, "n": 2}
    assert cells["3"] == {"rate": 1.0, "n": 2}


def test_the_cross_tab_cells_reconstruct_the_overall_rate():
    rows = [{"truncated": i % 3 == 0, "tokens_sent": 100, "n_evidence": (i % 2) + 1}
            for i in range(12)]

    metrics = _metrics(rows)
    cells = metrics["context_truncation_by_evidence_count"]

    weighted = sum(c["rate"] * c["n"] for c in cells.values())
    assert weighted / sum(c["n"] for c in cells.values()) == metrics["context_truncation_rate"]


def test_truncation_is_measurable_without_a_judge():
    """The M4 question is about retrieval and the cap, so asking it must not cost a grading pass.

    `_drill_unit` records `context_truncated` on every run, judged or not, which is what makes a
    retrieval-only run enough to decide whether the budget starves the reader.
    """
    from memrank.core import Document
    from memrank.metrics.cost import context_truncated

    small = [Document(id="a", content="word " * 10)]
    large = [Document(id="a", content="word " * 5000)]

    assert context_truncated(small, token_budget=5000) is False
    assert context_truncated(large, token_budget=5000) is True
    # An uncapped arm is never truncated -- the same answer _context_text gives.
    assert context_truncated(large, token_budget=5000, budget_mode="uncapped") is False


def test_drill_rows_carry_the_truncation_flag():
    from memrank.core import AdapterResponse, BenchmarkUnit, Document
    from memrank.runner import _drill_unit

    unit = BenchmarkUnit(
        unit_id="u", isolation_id="u",
        documents=[], queries=[{"id": "q", "text": "t", "gold_ids": ["a"]}])
    responses = [AdapterResponse(query_id="q",
                                 documents=[Document(id="a", content="word " * 5000)])]

    rows = _drill_unit(unit, responses, token_budget=5000)

    assert rows[0]["context_truncated"] is True
