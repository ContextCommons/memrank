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
"""LoCoMo evidence recall -- the contract from the design spec.

The official `recall_acc` at session granularity via deterministic content matching
(docs/superpowers/specs/2026-08-13-evidence-recall-design.md). The paraphrase case is asserted
as a DOCUMENTED limitation: fixing it into a fuzzy matcher would recreate for retrieval the
judge-dependence dispute this metric exists to escape.
"""
from memrank.core import AdapterResponse, BenchmarkUnit, Document
from memrank.metrics.evidence_recall import evidence_recall

_SESSION_TEXT = ("Caroline: I went to the support group on Tuesday and it was inspiring. "
                 "Melanie: That sounds wonderful, I am so glad you found a community there.")


def _unit(evidence_ids=("s1_session_1",)):
    doc = Document(id="s1_session_1", content="[]", user_id="s1",
                   messages=[{"role": "user", "content": _SESSION_TEXT, "speaker": "Caroline"}])
    return BenchmarkUnit(
        unit_id="s1", isolation_id="s1", documents=[doc],
        queries=[{"id": "s1_q0", "text": "when?", "user_id": "s1", "category": "temporal",
                  "gold_answers": ["tuesday"], "evidence_doc_ids": list(evidence_ids)}],
    )


def _response(*contents):
    return [AdapterResponse(query_id="s1_q0",
                            documents=[Document(id=f"e{i}", content=c)
                                       for i, c in enumerate(contents)])]


def test_verbatim_echo_scores_full_recall():
    result = evidence_recall(_unit(), _response(_SESSION_TEXT))

    assert result["evidence_recall"] == 1.0
    assert result["per_query"][0]["matched"] == ["s1_session_1"]


def test_disjoint_retrieval_scores_zero():
    result = evidence_recall(_unit(), _response("The weather in Paris is lovely in spring."))

    assert result["evidence_recall"] == 0.0
    assert result["per_query"][0]["unmatched"] == ["s1_session_1"]


def test_paraphrase_misses_and_that_is_the_documented_limitation():
    """An extracted memory is a legitimate miss under the content matcher (spec: 'declared
    limitation'). This pin exists so the matcher is never silently made fuzzy."""
    result = evidence_recall(_unit(), _response("Caroline attended a support group recently."))

    assert result["evidence_recall"] == 0.0


def test_partial_snippet_matches_via_shingles():
    snippet = _SESSION_TEXT[: len(_SESSION_TEXT) * 3 // 4]

    assert evidence_recall(_unit(), _response(snippet))["evidence_recall"] == 1.0


def test_empty_evidence_queries_are_excluded_not_zeroed():
    """The 4 real empty-evidence queries (all category 3) must not drag the mean; a unit with
    nothing but empty-evidence queries has no recall to report -- None, never 0.0."""
    result = evidence_recall(_unit(evidence_ids=()), _response(_SESSION_TEXT))

    assert result["evidence_recall"] is None
    assert result["n_queries_with_evidence"] == 0
