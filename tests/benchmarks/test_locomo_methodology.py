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
"""LoCoMo methodology pins -- the Tier-A assertions from protocol-fidelity section 5.1.

The category integer->name mapping is the single most miscopied fact about LoCoMo: the
paper's prose order and the data's integer order differ, and memrank shipped the wrong one for
weeks -- which the judge gate silently turned into a 39.2% denominator. These tests pin the
mapping to the data's authority (snap-research task_eval/evaluation.py), pin judgeability at
100% of categories 1-4, and pin
the dataset digest check so upstream drift can never be silent again.
"""
import json

import pytest

from memrank.benchmarks.locomo import _CATEGORY_NAMES, LoCoMoBenchmark


def _write_fixture(tmp_path, monkeypatch, qa):
    fixture = [{
        "sample_id": "s1",
        "conversation": {
            "speaker_a": "A", "speaker_b": "B",
            "session_1": [{"dia_id": "D1:1", "speaker": "A", "text": "we met in Paris"}],
            "session_1_date_time": "1:00 pm on 8 May, 2023",
        },
        "qa": qa,
    }]
    path = tmp_path / "locomo.json"
    path.write_text(json.dumps(fixture), encoding="utf-8")
    monkeypatch.setenv("LOCOMO_DATA_PATH", str(path))


def _one_qa_per_category():
    return [
        {"category": 1, "question": "q1", "answer": "a", "evidence": ["D1:1"]},
        {"category": 2, "question": "q2", "answer": "a", "evidence": ["D1:1"]},
        {"category": 3, "question": "q3", "answer": "a", "evidence": ["D1:1"]},
        {"category": 4, "question": "q4", "answer": "a", "evidence": ["D1:1"]},
        {"category": 5, "question": "q5", "adversarial_answer": "trap"},
    ]


def test_category_mapping_matches_the_data_not_the_paper():
    # Authority: snap-research task_eval/evaluation.py dispatch (card section 2). The paper's prose
    # lists single/multi/temporal/open -- the data's integers do not.
    assert _CATEGORY_NAMES == {1: "multi-hop", 2: "temporal",
                               3: "open-domain", 4: "single-hop"}


def test_loaded_queries_carry_the_data_order_names(tmp_path, monkeypatch):
    _write_fixture(tmp_path, monkeypatch, _one_qa_per_category())

    names = [q["category"] for q in LoCoMoBenchmark().load()[0].queries]

    assert names == ["multi-hop", "temporal", "open-domain", "single-hop"]


def test_all_four_categories_are_judgeable(tmp_path, monkeypatch):
    """Tier A: judged_coverage on categories 1-4 is 1.0 -- no silent denominator."""
    _write_fixture(tmp_path, monkeypatch, _one_qa_per_category())

    bench = LoCoMoBenchmark()
    queries = bench.load()[0].queries
    shape = bench.judge_shape()

    assert len(queries) == 4  # cat 5 dropped at load
    assert all(shape.is_judgeable(q) for q in queries)


def test_download_digest_is_verified(tmp_path):
    bad = tmp_path / "locomo10.json"
    bad.write_text("[]", encoding="utf-8")

    with pytest.raises(RuntimeError, match="sha256"):
        LoCoMoBenchmark._verify_digest(bad)


def test_fingerprint_rejects_wrong_counts():
    """The canonical file's shape is asserted, not assumed: 10 conversations, 1,986 questions,
    category counts {1: 282, 2: 321, 3: 96, 4: 841, 5: 446}."""
    truncated = [{"sample_id": "conv-26", "qa": [{"category": 1}]}]

    with pytest.raises(RuntimeError, match="fingerprint"):
        LoCoMoBenchmark._verify_fingerprint(truncated)


def test_local_override_skips_digest_and_fingerprint(tmp_path, monkeypatch):
    """LOCOMO_DATA_PATH is the deliberate escape hatch (it already flips
    question_text_public); a one-conversation fixture must load without pin checks."""
    _write_fixture(tmp_path, monkeypatch, _one_qa_per_category())

    assert len(LoCoMoBenchmark().load()) == 1


def test_temporal_questions_carry_the_official_date_suffix(tmp_path, monkeypatch):
    """The published protocol appends this wording to category-2 question text itself
    (task_eval/gpt_utils.py); other categories are untouched."""
    _write_fixture(tmp_path, monkeypatch, _one_qa_per_category())

    queries = {q["category"]: q["text"] for q in LoCoMoBenchmark().load()[0].queries}

    suffix = " Use DATE of CONVERSATION to answer with an approximate date."
    assert queries["temporal"].endswith(suffix)
    assert not any(text.endswith(suffix) for cat, text in queries.items() if cat != "temporal")


def test_sessions_load_in_chronological_not_lexicographic_order(tmp_path, monkeypatch):
    """sorted() put session_9 after session_19; found when the reader anchored every relative
    date to a mid-conversation "now" (M3 hand verification)."""
    import json as _json
    fixture = [{
        "sample_id": "s1",
        "conversation": {
            "speaker_a": "A", "speaker_b": "B",
            "session_2": [{"dia_id": "D2:1", "speaker": "A", "text": "second"}],
            "session_2_date_time": "1:00 pm on 8 May, 2023",
            "session_10": [{"dia_id": "D10:1", "speaker": "A", "text": "tenth"}],
            "session_10_date_time": "1:00 pm on 22 October, 2023",
        },
        "qa": [{"category": 2, "question": "when?", "answer": "2023", "evidence": ["D2:1"]}],
    }]
    path = tmp_path / "locomo.json"
    path.write_text(_json.dumps(fixture), encoding="utf-8")
    monkeypatch.setenv("LOCOMO_DATA_PATH", str(path))

    unit = LoCoMoBenchmark().load()[0]

    assert [d.id for d in unit.documents] == ["s1_session_2", "s1_session_10"]
    assert unit.queries[0]["query_timestamp"].startswith("2023-10-22")


def test_document_content_is_the_dated_rendering_not_a_json_blob(tmp_path, monkeypatch):
    """The official context is dated, speaker-attributed text; a raw JSON blob (audit F10) is
    what no published system ever saw, and it starves the full-context reader of dates."""
    _write_fixture(tmp_path, monkeypatch, _one_qa_per_category())

    content = LoCoMoBenchmark().load()[0].documents[0].content

    assert content.startswith("Conversation between A and B")
    assert "1:00 pm on 8 May, 2023" in content
    assert "A: we met in Paris" in content
    assert '"dia_id"' not in content
