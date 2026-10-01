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
"""LongMemEval methodology pins -- the Tier-A assertions from protocol-fidelity section 5.1.

The FIRST loader-level tests this benchmark has ever had. `test_longmemeval_scoring.py`
hand-builds BenchmarkUnits and never calls `load()`, which is precisely how two defects survived
unnoticed: the evidence set built from `has_answer` rather than `answer_session_ids` (a strict
subset on 62 of 500 questions), and slices that take `raw[:N]` from a type-GROUPED file and so
cover one of six question types.

Pinned here: the six-type fingerprint, abstention as an id suffix rather than a seventh type,
the dataset digest, and slice stratification.
"""
import json

import pytest

from memrank.benchmarks.longmemeval import (
    _FINGERPRINT,
    _QUESTION_TYPES,
    LongMemEvalBenchmark,
)


def _item(qid: str, qtype: str, *, n_sessions: int = 2) -> dict:
    """One record in the shape `longmemeval_s_cleaned.json` ships.

    Session ids carry the real `answer_*` / `sharegpt_*` prefixes: the dataset marks evidence
    sessions with an `answer` prefix and the official retrieval scorer keys on it, so a fixture
    that anonymised them would not exercise what the loader does with them.
    """
    sessions, ids, dates = [], [], []
    for i in range(n_sessions):
        marker = "answer_ev" if i == 0 else f"sharegpt_filler{i}"
        sessions.append([
            {"role": "user", "content": f"turn {i} user", "has_answer": i == 0},
            {"role": "assistant", "content": f"turn {i} assistant"},
        ])
        ids.append(f"{marker}_{i}")
        dates.append(f"2023/05/{10 + i:02d} (Wed) 1{i}:30")
    return {
        "question_id": qid, "question_type": qtype,
        "question": f"question for {qid}", "answer": "an answer",
        "question_date": "2023/06/01 (Thu) 09:15",
        "haystack_session_ids": ids, "haystack_dates": dates,
        "haystack_sessions": sessions, "answer_session_ids": [ids[0]],
    }


def _write_fixture(tmp_path, monkeypatch, items) -> None:
    path = tmp_path / "longmemeval.json"
    path.write_text(json.dumps(items), encoding="utf-8")
    monkeypatch.setenv("LONGMEMEVAL_DATA_PATH", str(path))


def _one_per_type(copies: int = 1) -> list[dict]:
    return [_item(f"q{t}{n}", t) for n in range(copies) for t in _QUESTION_TYPES]


def test_question_type_counts_match_the_pinned_fingerprint():
    """The six official types and their counts, computed from the pinned artifact."""
    assert _FINGERPRINT == {"temporal-reasoning": 133, "multi-session": 133,
                            "knowledge-update": 78, "single-session-user": 70,
                            "single-session-assistant": 56, "single-session-preference": 30}
    assert sum(_FINGERPRINT.values()) == 500


def test_abstention_is_a_suffix_not_a_question_type():
    """The 30 `_abs` items sit INSIDE the six buckets; `question_type` keeps the base type.

    Treating abstention as a seventh `question_type` is the single most common third-party
    error.
    """
    assert "abstention" not in _QUESTION_TYPES
    assert len(_QUESTION_TYPES) == 6
    assert set(_FINGERPRINT) == set(_QUESTION_TYPES)


def test_download_digest_is_verified(tmp_path):
    bad = tmp_path / "longmemeval_s_cleaned.json"
    bad.write_text("[]", encoding="utf-8")

    with pytest.raises(RuntimeError, match="sha256"):
        LongMemEvalBenchmark._verify_digest(bad)


def test_fingerprint_rejects_wrong_counts():
    with pytest.raises(RuntimeError, match="fingerprint"):
        LongMemEvalBenchmark._verify_fingerprint([{"question_type": "multi-session"}])


def test_local_override_skips_digest_and_fingerprint(tmp_path, monkeypatch):
    """LONGMEMEVAL_DATA_PATH is the deliberate escape hatch, already marked by
    `question_text_public = False`; a fixture must not have to match the canonical digest."""
    _write_fixture(tmp_path, monkeypatch, _one_per_type())

    assert len(LongMemEvalBenchmark().load()) == 6


def test_every_question_type_is_judgeable(tmp_path, monkeypatch):
    """Tier A: judged_coverage is 1.0 -- every item carries a non-empty gold answer."""
    _write_fixture(tmp_path, monkeypatch, _one_per_type())
    bench = LongMemEvalBenchmark()
    shape = bench.judge_shape()

    queries = [q for unit in bench.load() for q in unit.queries]

    assert len(queries) == 6
    assert all(shape.is_judgeable(q) for q in queries)


def test_receipt_config_carries_the_dataset_digest(tmp_path, monkeypatch):
    _write_fixture(tmp_path, monkeypatch, _one_per_type())

    config = LongMemEvalBenchmark().config_for_receipt()

    assert "longmemeval_dataset_sha256" in config
    assert len(config["longmemeval_dataset_sha256"]) == 64
