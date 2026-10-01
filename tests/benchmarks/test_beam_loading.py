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
"""What BEAM hands an engine, and what it asks the scorer to do with it.

The load-bearing pin here is that a BEAM document is role+content and NOTHING else. The
harness survey (docs/research/2026-08-13-how-beam-is-actually-evaluated.md) showed every
published BEAM harness -- including BEAM's own baselines -- hands systems only the turns:
no rendered session date, no machine timestamp, no asked-at time. memrank briefly forwarded
the dataset's `time_anchor` through `Document.context`/`timestamp`/`query_timestamp` (audit
F6 read the dropped anchor as a defect), which handed our engines temporal signal no other
harness's systems had; decision-beam-runs-its-own-protocol reversed that, and these tests
pin the reversal so the anchor cannot quietly grow back.

Also pinned, from the same audit: F3 -- abstention probes were emitted as positives, so the
judge graded them with the wrong prompt and ran a retrieval-sufficiency check on questions
BEAM builds to be unanswerable. The trap that shaped that fix -- tagging abstention
`kind="negative"` WITHOUT `forbidden_spans` made the substring scorer return a hit
unconditionally -- is now historical: that scorer no longer runs on BEAM at all, and the
ability is counted rather than excluded to work around it.
"""

import json
import sys

import pytest

from memrank.adapters import transcript
from memrank.benchmarks import beam
from memrank.benchmarks.beam import BEAMBenchmark
from memrank.errors import MissingOptionalDependency
from memrank.judging.judge import JudgeConfig, calls_per_query

#: Shaped like the published dataset: `chat` is a list of sessions, each a list of turns, with
#: the session's `time_anchor` on its first turn only. `probing_questions` ships as a PYTHON
#: REPR string rather than JSON -- single quotes -- which is why the loader needs its
#: `ast.literal_eval` fallback. Pinning that here keeps the fallback honest.
_FIXTURE = [{
    "conversation_id": "c1",
    "user_profile": {"user_info": "builds a budget tracker"},
    "chat": [
        [
            {"role": "user", "id": 0, "time_anchor": "March-15-2024",
             "content": "My first sprint ends on March 29."},
            {"role": "assistant", "id": 1, "content": "Noted."},
        ],
        [
            {"role": "user", "id": 2, "time_anchor": "April-05-2024",
             "content": "Security review comes after transactions."},
            {"role": "assistant", "id": 3, "content": "Understood."},
        ],
    ],
    "probing_questions": repr({
        "information_extraction": [
            {"question": "When does my first sprint end?", "answer": "March 29",
             "rubric": ["March 29"]},
        ],
        "abstention": [
            {"question": "What is my production deployment URL?",
             "ideal_response": "I don't have that information.",
             "why_unanswerable": "Never discussed in chat",
             "rubric": ["should abstain or state information is unavailable"]},
        ],
    }),
}]


def _load(tmp_path, monkeypatch, fixture=None):
    """Load BEAM from a local fixture. The env var must be set BEFORE construction --
    `BEAMBenchmark.__init__` snapshots it."""
    path = tmp_path / "beam.json"
    path.write_text(json.dumps(fixture or _FIXTURE), encoding="utf-8")
    monkeypatch.setenv("BEAM_DATA_PATH", str(path))
    return BEAMBenchmark(tier="100k").load()


def test_documents_are_role_content_only_and_no_anchor_reaches_an_engine(tmp_path, monkeypatch):
    """The dataset's `time_anchor` must not reach an engine on ANY channel -- not rendered
    (transcript.header is gated on `context`), not machine-readable (`timestamp`). Both fixture
    sessions carry anchors, so a regression on either session or either channel trips this."""
    unit = _load(tmp_path, monkeypatch)[0]
    first, second = unit.documents[0], unit.documents[1]
    assert first.messages, "no messages means transcript.render falls back to raw content"
    for doc in (first, second):
        assert doc.context is None
        assert doc.timestamp is None
        rendered = transcript.render(doc)
        assert "2024" not in rendered, "an anchor date leaked into ingested text"
        assert not rendered.startswith("["), "a transcript header rendered for BEAM"


def test_the_user_profile_never_reaches_an_ingested_document(tmp_path, monkeypatch):
    """`user_info` is a dossier -- name, age, gender, location, profession -- and it used to sit in
    `context`. That was inert only while BEAM set no `messages`, because transcript.header() is
    gated on them. Once messages landed, the header rendered into EVERY chunk, which hands an
    engine a free answer sheet for the information-extraction probes. BEAM's own baselines are
    given the chat, not the profile."""
    unit = _load(tmp_path, monkeypatch)[0]
    rendered = "\n".join(transcript.render(d) for d in unit.documents)
    assert "budget tracker" not in rendered, "user_profile.user_info leaked into ingest"
    assert all("USER PROFILE" not in (d.context or "") for d in unit.documents)


def test_the_reference_answer_survives_under_the_shared_name(tmp_path, monkeypatch):
    """`gold_answers` is not BEAM's scoring key -- the rubric is -- but it IS the cross-benchmark
    field other surfaces read as "the reference answer for this query". Renaming it for BEAM alone
    made `arena/context_source.py` find no gold and return no packet, silently dropping every BEAM
    positive query from arena battles."""
    unit = _load(tmp_path, monkeypatch)[0]
    extraction = next(q for q in unit.queries if q["category"] == "information_extraction")
    assert extraction["gold_answers"] == ["March 29"]
    assert extraction["rubric"], "and the rubric, which is what actually grades it"


def test_queries_carry_no_asked_at_time(tmp_path, monkeypatch):
    """No published harness tells the system WHEN a probe is asked, so neither do we. The key is
    absent rather than None: adapters gate native query-time features on its presence."""
    unit = _load(tmp_path, monkeypatch)[0]
    assert all("query_timestamp" not in q for q in unit.queries)


def test_abstention_is_tagged_negative_and_the_rest_positive(tmp_path, monkeypatch):
    """Polarity decides which correctness prompt the judge uses and whether sufficiency runs."""
    unit = _load(tmp_path, monkeypatch)[0]
    kinds = {q["category"]: q["kind"] for q in unit.queries}
    assert kinds["abstention"] == "negative"
    assert kinds["information_extraction"] == "positive"


def test_abstention_is_counted_now_that_no_proxy_can_award_it_a_free_hit(tmp_path, monkeypatch):
    """Abstention used to be dropped from the denominator; with the proxy gone it is just counted.

    The exclusion existed because a negative spec carrying no forbidden spans hits
    unconditionally -- `any(...)` over an empty list is False for every document, so the loop fell
    through to True even against an empty retrieval, and scoring it read as retrieval succeeding
    when nothing was retrieved. That was a distortion the substring metric forced on the data.
    BEAM's rubric judge grades abstention on its own terms, so the ability now sits in the counts
    beside every other, with its size reported rather than silently removed.
    """
    unit = _load(tmp_path, monkeypatch)[0]

    score = BEAMBenchmark(tier="100k").score(unit, [])

    assert score["n_queries"] == 2, "both the extraction and the abstention probe are counted"
    assert score["n_abstention"] == 1
    assert score["per_ability_counts"]["abstention"] == 1


def test_abstention_costs_fewer_judge_calls_than_a_positive(tmp_path, monkeypatch):
    """Skipping sufficiency on an unanswerable question is the point of the tag, not a side effect.

    Sufficiency asks whether the retrieved snippets are enough to answer. For a probe BEAM
    built to be unanswerable that check can only fail, so it spent calls to learn nothing and
    depressed `retrieval_sufficiency` by a fixed fraction while it did.
    """
    unit = _load(tmp_path, monkeypatch)[0]
    cfg = JudgeConfig(samples=1)
    cost = {q["category"]: calls_per_query(cfg, negative=q["kind"] == "negative")
            for q in unit.queries}
    assert cost["abstention"] < cost["information_extraction"]


def test_a_long_session_splits_and_every_chunk_stays_bare(tmp_path, monkeypatch):
    """`messages` must survive a split, and the split must not reintroduce per-chunk metadata:
    every chunk of an anchored session stays role+content only."""
    monkeypatch.setattr(beam, "_MAX_DOC_CHARS", 200)
    turns = [{"role": "user", "id": i, "content": "x" * 60} for i in range(10)]
    turns[0]["time_anchor"] = "March-15-2024"
    unit = _load(tmp_path, monkeypatch, [dict(_FIXTURE[0], chat=[turns])])[0]
    assert len(unit.documents) > 1, "a session over the cap must split"
    assert all(len(d.content) <= 200 for d in unit.documents)
    assert all(d.messages for d in unit.documents)
    assert all(d.context is None and d.timestamp is None for d in unit.documents)


def test_an_uncached_tier_without_the_benchmarks_extra_names_the_install(tmp_path, monkeypatch):
    """`datasets` is the `benchmarks` extra, not a base dependency, so a released install that
    lacks it must be told the install that fixes it -- not that it is broken -- and must leave
    no half-written cache file behind for the next load to trust."""
    monkeypatch.delenv("BEAM_DATA_PATH", raising=False)
    monkeypatch.setenv("MEMRANK_CACHE_DIR", str(tmp_path))
    monkeypatch.setitem(sys.modules, "datasets", None)   # `import datasets` raises ImportError
    monkeypatch.setattr("memrank.provenance.install._direct_url", lambda distribution: None)
    monkeypatch.setattr("memrank.provenance.install._installed_as_uv_tool",
                        lambda distribution: False)

    with pytest.raises(MissingOptionalDependency) as excinfo:
        BEAMBenchmark(tier="100k").load()

    assert "pip install --upgrade 'memrank[benchmarks]'" in str(excinfo.value)
    assert not (tmp_path / "beam" / "100k.json").exists()
