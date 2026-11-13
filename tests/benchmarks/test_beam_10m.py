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
"""BEAM-10M: a separate dataset whose chat is plan -> batch -> exchange -> turn.

Pinned here: the mapping onto the other tiers' shape (one session per batch, plan order then
batch order, exchanges concatenated); that anything else is refused rather than guessed at; and
that the per-conversation cache lets a slice read only its own conversations.
"""

import json
import sys

import pytest

from memrank.adapters import transcript
from memrank.benchmarks import beam_10m
from memrank.benchmarks.beam import BEAMBenchmark
from memrank.benchmarks.beam_10m import Beam10MShapeError, sessions_of
from memrank.errors import MissingOptionalDependency

_PLAN_KEYS = ("plan-1", "plan-2")


def _turn(turn_id, role, content):
    return {"role": role, "id": turn_id, "index": None, "question_type": None,
            "time_anchor": "July-01-2024", "content": content}


def _batch(number, *exchanges):
    return {"batch_number": number, "time_anchor": None, "turns": [list(e) for e in exchanges]}


def _plan_element(key, batches):
    """One `chat` element: every plan key present, only its own non-null (as HF stores it)."""
    return {k: (batches if k == key else None) for k in _PLAN_KEYS}


def _chat():
    return [
        _plan_element("plan-1", [
            _batch(1, [_turn(0, "user", "p1b1 ask"), _turn(1, "assistant", "p1b1 reply")],
                   [_turn(2, "user", "p1b1 follow-up"), _turn(3, "assistant", "ok")]),
            _batch(2, [_turn(4, "user", "p1b2 ask"), _turn(5, "assistant", "p1b2 reply")]),
        ]),
        _plan_element("plan-2", [
            _batch(1, [_turn(6, "user", "p2b1 ask"), _turn(7, "assistant", "p2b1 reply")]),
        ]),
    ]


def _row(conv_id):
    return {"conversation_id": conv_id, "chat": _chat(),
            "user_profile": {"user_info": "dossier never ingested"},
            "probing_questions": repr({
                "event_ordering": [{"question": "Order?", "answer": "a then b",
                                    "rubric": ["a", "b"], "ordering_tested": ["a", "b"],
                                    "plan_reference": "Plan 0-1"}],
                "abstention": [{"question": "URL?", "ideal_response": "Unknown.",
                                "rubric": ["abstains"], "plan_reference": "None"}],
            })}


def test_one_session_per_batch_in_plan_then_batch_order():
    sessions = sessions_of(_chat())
    assert [[t["id"] for t in s] for s in sessions] == [[0, 1, 2, 3], [4, 5], [6, 7]]


@pytest.mark.parametrize("mutate, message", [
    (lambda c: c[0].update({"plan-2": c[1]["plan-2"]}), "expected exactly"),
    (lambda c: c.reverse(), "expected exactly"),
    (lambda c: c[0]["plan-1"].reverse(), "batch_number"),
    (lambda c: c[1]["plan-2"][0].update({"turns": [_turn(6, "user", "flat")]}), "exchanges"),
])
def test_any_other_layout_is_refused_rather_than_guessed(mutate, message):
    chat = _chat()
    mutate(chat)
    with pytest.raises(Beam10MShapeError, match=message):
        sessions_of(chat)


def test_local_file_loads_role_content_documents_and_the_10m_dataset(tmp_path, monkeypatch):
    path = tmp_path / "beam10m.json"
    path.write_text(json.dumps([_row("1"), _row("2")]), encoding="utf-8")
    monkeypatch.setenv("BEAM_DATA_PATH", str(path))
    bench = BEAMBenchmark(tier="10m", slice="smoke")

    units = bench.load()

    assert bench.dataset_version == "Mohammadta/BEAM-10M@v1"
    assert BEAMBenchmark(tier="1m").dataset_version == "Mohammadta/BEAM@v1"
    assert [u.unit_id for u in units] == ["1"]
    unit = units[0]
    assert [d.id for d in unit.documents] == ["1_s0_0", "1_s1_0", "1_s2_0"]
    rendered = "\n".join(transcript.render(d) for d in unit.documents)
    assert "p1b1 follow-up" in rendered and "2024" not in rendered
    assert "dossier" not in rendered
    assert all(d.context is None and d.timestamp is None for d in unit.documents)
    assert {q["category"]: q["kind"] for q in unit.queries} == {
        "abstention": "negative", "event_ordering": "positive"}
    assert next(q for q in unit.queries if q["category"] == "event_ordering")[
        "ordering_tested"] == ["a", "b"]


def _seed_cache(root, conv_ids, manifest=True):
    cache = root / "beam" / "10m"
    cache.mkdir(parents=True)
    names = []
    for position, conv_id in enumerate(conv_ids):
        names.append(f"{position:02d}.json")
        (cache / names[-1]).write_text(json.dumps(_row(conv_id)), encoding="utf-8")
    if manifest:
        (cache / "manifest.json").write_text(json.dumps(names), encoding="utf-8")
    return cache


def test_a_slice_reads_only_its_own_conversations_from_the_cache(tmp_path, monkeypatch):
    monkeypatch.delenv("BEAM_DATA_PATH", raising=False)
    monkeypatch.setenv("MEMRANK_CACHE_DIR", str(tmp_path))
    cache = _seed_cache(tmp_path, [str(i) for i in range(7)])
    (cache / "06.json").write_text("not json", encoding="utf-8")   # read it and this fails

    mini = BEAMBenchmark(tier="10m", slice="mini").load()

    assert [u.unit_id for u in mini] == ["0", "1", "2", "3", "4"]
    with pytest.raises(json.JSONDecodeError):
        BEAMBenchmark(tier="10m").load()


def test_a_cache_without_its_manifest_is_downloaded_again(tmp_path, monkeypatch):
    """The manifest is written last, so its absence means an interrupted download."""
    monkeypatch.delenv("BEAM_DATA_PATH", raising=False)
    monkeypatch.setenv("MEMRANK_CACHE_DIR", str(tmp_path))
    _seed_cache(tmp_path, ["stale"], manifest=False)
    fetched = []

    def fake_download(cache_dir):
        fetched.append(cache_dir)
        (cache_dir / "00.json").write_text(json.dumps(_row("fresh")), encoding="utf-8")
        (cache_dir / "manifest.json").write_text(json.dumps(["00.json"]), encoding="utf-8")

    monkeypatch.setattr(beam_10m, "_download", fake_download)
    assert [u.unit_id for u in BEAMBenchmark(tier="10m").load()] == ["fresh"]
    assert fetched == [tmp_path / "beam" / "10m"]


def test_an_uncached_10m_without_the_benchmarks_extra_names_the_install(tmp_path, monkeypatch):
    monkeypatch.delenv("BEAM_DATA_PATH", raising=False)
    monkeypatch.setenv("MEMRANK_CACHE_DIR", str(tmp_path))
    monkeypatch.setitem(sys.modules, "datasets", None)
    monkeypatch.setattr("memrank.provenance.install._direct_url", lambda distribution: None)
    monkeypatch.setattr("memrank.provenance.install._installed_as_uv_tool",
                        lambda distribution: False)

    with pytest.raises(MissingOptionalDependency, match=r"memrank\[benchmarks\]"):
        BEAMBenchmark(tier="10m", slice="smoke").load()

    assert not (tmp_path / "beam" / "10m" / "manifest.json").exists()
