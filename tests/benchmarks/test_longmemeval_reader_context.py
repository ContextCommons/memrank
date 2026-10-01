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
"""What the reader is handed -- dates in band, and timestamps that keep their time.

The official harness renders each session as
`### Session {i}\\nSession Date: {date}\\nSession Content:\\n{json}`. memrank shipped a bare
`json.dumps(turns)`, so the date reached engines only via `Document.timestamp` and `context` --
channels a content-reading engine and the full-context arm never see.

Two question types are DEFINED by what was missing. `knowledge-update` (78 questions) asks which
of two contradictory facts is current: presented undated, it is not answerable in principle.
`temporal-reasoning` (133) is specified as reasoning over "the timestamp in metadata".

Separately, `_parse_date` split on "(" and then fell through to a bare `%Y/%m/%d`, silently
discarding the time of day on EVERY record -- so all timestamps were midnight and the reader's
`Current date` block lost its hour.
"""
import pytest

from memrank.benchmarks.longmemeval import LongMemEvalBenchmark
from tests.benchmarks.test_longmemeval_methodology import _item, _one_per_type, _write_fixture


def test_parse_date_keeps_the_time_of_day():
    assert LongMemEvalBenchmark._parse_date("2023/05/30 (Tue) 23:40") == \
        "2023-05-30T23:40:00+00:00"


def test_parse_date_raises_on_an_unknown_format():
    """AGENTS.md forbids degraded modes. The old bare-except plus `%Y/%m/%d` fallback is what
    turned a format mismatch into a silent midnight."""
    with pytest.raises(ValueError, match="30 May 2023"):
        LongMemEvalBenchmark._parse_date("30 May 2023")


def test_missing_date_is_absent_not_invented():
    assert LongMemEvalBenchmark._parse_date("") is None


def test_content_leads_with_the_session_date(tmp_path, monkeypatch):
    _write_fixture(tmp_path, monkeypatch, [_item("q1", "knowledge-update", n_sessions=3)])

    docs = LongMemEvalBenchmark().load()[0].documents

    for doc, expected in zip(docs, ["2023/05/10 (Wed) 10:30", "2023/05/11 (Wed) 11:30",
                                    "2023/05/12 (Wed) 12:30"], strict=True):
        assert doc.content.startswith(f"Session date: {expected}")


def test_content_is_a_transcript_not_a_json_blob(tmp_path, monkeypatch):
    _write_fixture(tmp_path, monkeypatch, [_item("q1", "multi-session")])

    content = LongMemEvalBenchmark().load()[0].documents[0].content

    assert "user: turn 0 user" in content
    assert "assistant: turn 0 assistant" in content
    assert not content.lstrip().startswith("["), "content must not be a JSON array"
    assert '"role"' not in content


def test_has_answer_never_reaches_the_rendered_content(tmp_path, monkeypatch):
    """The evidence label is stripped before prompting (run_generation.py:178-190)."""
    _write_fixture(tmp_path, monkeypatch, _one_per_type())

    for unit in LongMemEvalBenchmark().load():
        for doc in unit.documents:
            assert "has_answer" not in doc.content
            assert all("has_answer" not in m for m in doc.messages)


def test_context_carries_the_date_but_still_no_identifier(tmp_path, monkeypatch):
    """Dates are a legitimate input -- the protocol renders `Session Date:` to the reader.
    Session IDS are not (audit F1), so `context` gains the date and keeps the position."""
    _write_fixture(tmp_path, monkeypatch, [_item("q1", "temporal-reasoning", n_sessions=2)])

    unit = LongMemEvalBenchmark().load()[0]
    real_ids = set(unit.metadata["session_handles"].values())

    for i, doc in enumerate(unit.documents):
        assert doc.context.startswith(f"Session {i + 1}")
        assert "2023/05/" in doc.context
        assert not any(sid in doc.context for sid in real_ids)


def test_timestamps_are_distinct_within_a_day(tmp_path, monkeypatch):
    """211 of 500 instances have intra-day inversions; midnight timestamps made array order
    and true order indistinguishable to any engine sorting by time."""
    item = _item("q1", "temporal-reasoning", n_sessions=2)
    item["haystack_dates"] = ["2023/05/10 (Wed) 02:00", "2023/05/10 (Wed) 21:00"]

    _write_fixture(tmp_path, monkeypatch, [item])
    stamps = [d.timestamp for d in LongMemEvalBenchmark().load()[0].documents]

    assert stamps[0] != stamps[1]
    assert stamps == sorted(stamps)
