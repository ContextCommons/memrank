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
"""BEAM benchmark (Mohammadta/BEAM).

100 conversations of 100K-10M tokens, 2000 probing questions across 10
memory ability categories. Memrank v0.1 supports the 100K, 500K, and 1M
tiers; 10M is reserved for v0.2 (memory + bandwidth requirements).

BEAM scores against a ``rubric`` of atomic nuggets -- one judge call each, scored
{0, 0.5, 1} and averaged within the question. ``judge_shape()`` returns that grader,
so ``--judge`` measures BEAM the way BEAM specifies. All ten abilities are covered:
nine by rubric averaging, and ``event_ordering`` by rank correlation over its ordered
rubric, because averaging that one would score a reversed answer identically to a
correct one.

Without ``--judge`` this module ships NO quality number. It used to compute a
retrieval-recall proxy while declaring it structurally invalid (the rubric is prose, not
verbatim spans) and withholding it from rankings -- but a number that is computed and
displayed per-ability gets quoted whatever the flag beside it says, and it distorted the
data it touched: abstention probes had to be dropped from its denominator entirely. So
BEAM quality requires ``--judge``, and ``score()`` reports only counts without one.

Engines are handed role+content ONLY -- no rendered time anchors, no machine ``timestamp``,
no ``query_timestamp`` -- and the reader's context is uncapped (``context_policy``). That is
what every published BEAM harness does, including BEAM's own baselines, and it is what makes
a memrank BEAM number a BEAM number rather than a memrank variant of one. The dataset's
``time_anchor`` field exists and is deliberately not forwarded: only 16 of 90 sessions at
100k restate their date in-band, so forwarding it would hand our engines temporal
information no other harness's systems had.
"""

from __future__ import annotations

import ast
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from memrank.benchmarks.answer_prompts import BEAM_READER
from memrank.core import AdapterResponse, Benchmark, BenchmarkUnit, Document, EvalInfo
from memrank.judging.shape import BeamJudgeShape, JudgeShape

_HF_DATASET = "Mohammadta/BEAM"

# HuggingFace exposes BEAM as one dataset with three named splits. The 10M bucket is NOT a
# fourth split of this dataset -- it lives in a separate repo (`Mohammadta/BEAM-10M`), so
# adding it is a new source, not a new key here.
#
# The paper and the repo call the smallest bucket 128K; the split is named 100K and every
# third-party writeup follows the split. Same 20 conversations either way.
_HF_SPLIT_MAP = {
    "100k": "100K",
    "500k": "500K",
    "1m": "1M",
}

#: The one ability BEAM builds to be unanswerable. Its probes carry `why_unanswerable` and no
#: `source_chat_ids` at all, so a correct response withholds rather than answers -- which makes it
#: a negative, and makes it unscoreable by evidence recall. Named because both facts are acted on.
_ABSTENTION = "abstention"

_CATEGORIES = [
    "abstention",
    "contradiction_resolution",
    "event_ordering",
    "information_extraction",
    "instruction_following",
    "knowledge_update",
    "multi_session_reasoning",
    "preference_following",
    "summarization",
    "temporal_reasoning",
]

#: Where a BEAM question keeps its REFERENCE answer -- drill-in material, not the scoring key.
#: BEAM does not use one name: the field is chosen per ability, so `contradiction_resolution` says
#: `ideal_answer` and `summarization` says `ideal_summary`. The two compliance abilities say
#: `expected_compliance`, which is not an answer at all but a description of the behaviour
#: expected ("Response should include code examples formatted with syntax highlighting") -- which
#: is exactly why grading against this list was wrong and why the rubric grades instead.
_ANSWER_FIELDS = (
    "ideal_response",
    "answer",
    "expected_answer",
    "expected",
    "gold_answer",
    "reference",
    "ideal_answer",
    "ideal_summary",
    "expected_compliance",
)

_MAX_DOC_CHARS = 100_000


class BEAMBenchmark(Benchmark):
    """BEAM benchmark loader + per-ability retrieval scorer."""

    name = "beam"
    dataset_version = "Mohammadta/BEAM@v1"
    # tiers mirrors _HF_SPLIT_MAP (pinned by test) and leads with the default tier.
    #: Per bucket, NOT per tier. BEAM's headline "100 conversations, 2000 questions" is the total
    #: across all four buckets -- 20/35/35/10 conversations at 128K/500K/1M/10M -- and every
    #: conversation carries exactly 20 questions, two per ability. Quoting the total as a per-tier
    #: figure overstated 100k by 5x, which is the number a reader would use to sanity-check
    #: coverage.
    info = EvalInfo(unit="conversation",
                    units_declared="20 conversations / 400 questions at 100k; 35 / 700 at 500k "
                                   "and 1m; 20 questions per conversation, 2 per ability",
                    slices=("smoke", "mini"),
                    tiers=("100k", "500k", "1m"))
    # BEAM gold answers are ideal_response prose, not verbatim spans, so the substring-recall
    # proxy was structurally ~0 (even for the word-overlap arm). It is no longer computed at
    # all -- see `score()`. BEAM quality is measured by the rubric judge (`--judge`).
    substring_recall_supported = False
    # Nothing here is rankable without a judge: display "n/a (judge required)" until --judge
    # produces scores, and rank on `judged_metrics` when it has.
    composite_rankable = False
    quality_metric = "judged_nugget_rubric"
    # BEAM's protocol hands the reader whatever retrieval returned, uncapped -- every published
    # harness including BEAM's own baselines runs this way, so on BEAM how much context to hand
    # the reader is part of the system under test, not a harness control. The runner promotes
    # matched arms to uncapped for benchmarks that declare this; the no-memory arm stays "none".
    context_policy = "uncapped"
    answer_prompt = BEAM_READER
    # 2026-08-13: loading stopped forwarding the dataset's time anchors (no rendered header, no
    # Document.timestamp, no query_timestamp) and the reader went uncapped -- scores across this
    # boundary are not comparable.
    VERSION = 1

    def __init__(self, tier: str = "100k", slice: str | None = None, k: int = 10) -> None:
        tier = tier.lower()
        if tier not in _HF_SPLIT_MAP:
            raise ValueError(f"BEAM tier must be one of {list(_HF_SPLIT_MAP)}, got {tier!r}")
        self.tier = tier
        self.slice = slice
        self.k = k
        env = os.environ.get("BEAM_DATA_PATH")
        self._local_path: Path | None = Path(env) if env else None
        # Question text is public only when using the canonical dataset (no local override).
        self.question_text_public = env is None

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #

    def _cache_path(self) -> Path:
        from memrank.benchmarks import cache_root

        cache = cache_root() / "beam"
        cache.mkdir(parents=True, exist_ok=True)
        return cache / f"{self.tier}.json"

    def _load_raw(self) -> list[dict[str, Any]]:
        if self._local_path and self._local_path.exists():
            with open(self._local_path, encoding="utf-8") as f:
                return json.load(f)
        path = self._cache_path()
        if not path.exists():
            from memrank.benchmarks import dataset_download_notice

            with dataset_download_notice(f"BEAM-{self.tier}", _HF_DATASET, path):
                self._download(path)
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def _download(self, dest: Path) -> None:
        """Download the requested tier from HuggingFace hub with ``datasets``."""
        # `datasets` is the `benchmarks` extra (pyproject.toml), not a base dependency: nothing
        # but this download needs it, and a quick-start install should not carry pyarrow and
        # pandas for an evaluation it never runs. Absent, this names the install that fixes it
        # before anything is fetched; a cached tier never reaches this line.
        from memrank.errors import optional_import

        load_dataset = optional_import("datasets", "benchmarks").load_dataset
        hf_split = _HF_SPLIT_MAP[self.tier]
        ds = load_dataset(_HF_DATASET, split=hf_split)
        rows = [dict(row) for row in ds]
        with open(dest, "w", encoding="utf-8") as f:
            json.dump(rows, f)

    @staticmethod
    def _parse_probing_questions(item: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
        pq = item.get("probing_questions", {})
        if isinstance(pq, str):
            try:
                pq = json.loads(pq)
            except Exception:
                try:
                    pq = ast.literal_eval(pq)
                except Exception:
                    pq = {}
        return pq if isinstance(pq, dict) else {}

    @staticmethod
    def _extract_answer(question_obj: dict[str, Any]) -> str:
        """The first gold answer that actually has content.

        Keeps scanning past a field that is present but empty or null. Stopping at the first
        name PRESENT let an empty `answer` mask a populated `ideal_answer` later in the list --
        the same query then reads as having no gold at all.
        """
        for field_name in _ANSWER_FIELDS:
            value = question_obj.get(field_name)
            if value is not None and str(value).strip():
                return str(value)
        return ""

    @staticmethod
    def _turns_to_messages(turns: list[Any]) -> list[dict[str, Any]]:
        """Role-tagged messages, the structured channel `adapters/transcript.py` renders from.

        No ``speaker`` key. BEAM is one user talking to an assistant -- there are no named
        participants to attribute, and inventing them would be fiction. LongMemEval makes the
        same call for the same reason; LoCoMo carries speakers because its questions name them.
        """
        return [
            {"role": str(t.get("role") or "user"), "content": str(t.get("content") or "")}
            for t in turns
            if isinstance(t, dict) and "role" in t
        ]

    @staticmethod
    def _format_chat(chat: list[Any]) -> str:
        """Flatten BEAM's nested chat structure into role-prefixed lines."""
        lines: list[str] = []
        for item in chat:
            if isinstance(item, dict) and "role" in item:
                lines.append(f"{item.get('role', 'unknown').capitalize()}: {item.get('content', '')}")
            elif isinstance(item, list):
                for turn in item:
                    if isinstance(turn, dict) and "role" in turn:
                        lines.append(
                            f"{turn.get('role', 'unknown').capitalize()}: {turn.get('content', '')}"
                        )
        return "\n\n".join(lines)

    def _chunk_session(self, turns: list[Any], *, doc_prefix: str,
                       user_id: str) -> list[Document]:
        """One session, split into sub-documents under _MAX_DOC_CHARS.

        ``content`` stays the flat rendering because that is what the substring scorer reads
        (memrank/metrics/scoring.py). ``messages`` is what engines actually receive, through
        `adapters/transcript.py`. No ``context`` and no ``timestamp`` -- a BEAM document is
        role+content and nothing else (see the module docstring), so with ``context`` unset the
        transcript header renders nothing and engines ingest exactly the turns.
        """
        docs: list[Document] = []
        chunk_start = 0
        chunk_idx = 0
        while chunk_start < len(turns):
            chunk_end = chunk_start + 1
            while chunk_end < len(turns):
                candidate = self._format_chat([turns[chunk_start : chunk_end + 1]])
                if len(candidate) > _MAX_DOC_CHARS:
                    break
                chunk_end += 1
            chunk_turns = turns[chunk_start:chunk_end]
            docs.append(
                Document(
                    id=f"{doc_prefix}_{chunk_idx}",
                    content=self._format_chat([chunk_turns]),
                    messages=self._turns_to_messages(chunk_turns),
                    user_id=user_id,
                )
            )
            chunk_idx += 1
            chunk_start = chunk_end
        return docs

    def _conv_documents(self, conv_id: str, chat: list[Any]) -> list[Document]:
        """Split one conversation into per-session sub-documents under _MAX_DOC_CHARS.

        The dataset's per-session ``time_anchor`` is read by nobody here, deliberately: BEAM's
        own baselines and every published harness hand systems role+content only, so forwarding
        the anchor (rendered or as a machine ``timestamp``) would give our engines information
        no other harness's systems had. Only 16 of 90 sessions at 100k restate their date
        in-band -- the anchor is real extra signal, not redundancy.
        """
        sessions = [s for s in chat if isinstance(s, list)]
        if not sessions:
            flat = [t for t in chat if isinstance(t, dict) and "role" in t]
            return [
                Document(
                    id=conv_id,
                    content=self._format_chat(chat),
                    messages=self._turns_to_messages(flat),
                    user_id=conv_id,
                )
            ]
        docs: list[Document] = []
        for s_idx, session in enumerate(sessions):
            turns = [t for t in session if isinstance(t, dict) and "role" in t]
            if not turns:
                continue
            docs.extend(self._chunk_session(
                turns,
                doc_prefix=f"{conv_id}_s{s_idx}",
                user_id=conv_id,
            ))
        return docs

    def _conv_queries(self, conv_id: str, item: dict[str, Any]) -> list[dict[str, Any]]:
        """One query per probing question, in a stable ability-then-index order.

        ``kind`` is set on every query rather than left to default. Abstention is the negative:
        BEAM builds those probes to be unanswerable, so a correct response withholds, and the
        judge needs the negative correctness prompt and no sufficiency check. Everything else is
        an ordinary positive, said out loud so the polarity is a property of the data rather than
        an absence somebody has to know about.
        """
        queries: list[dict[str, Any]] = []
        pq = self._parse_probing_questions(item)
        for category in _CATEGORIES:
            for idx, q_obj in enumerate(pq.get(category, [])):
                text = q_obj.get("question") or ""
                if not text:
                    continue
                queries.append(
                    {
                        "id": f"{conv_id}_{category}_{idx}",
                        "text": text,
                        "user_id": conv_id,
                        "evidence_doc_ids": [conv_id],
                        # NOT the scoring key -- `rubric` below is, and `judge_shape()` says so.
                        # It stays under the SHARED name because `gold_answers` is a
                        # cross-benchmark field other surfaces legitimately read as "the reference
                        # answer for this query" (memrank/arena/context_source.py builds its
                        # reference from it, and returns no packet without one). Renaming it for
                        # BEAM alone silently dropped every BEAM positive query from arena.
                        #
                        # What made it harmful was never that it existed: it was that grading and
                        # JUDGEABILITY keyed off it, so two abilities that ship no answer at all
                        # -- instruction_following, preference_following -- were graded against a
                        # description of expected behaviour. Both now key off the rubric.
                        "gold_answers": [self._extract_answer(q_obj)],
                        "category": category,
                        "kind": "negative" if category == _ABSTENTION else "positive",
                        "rubric": q_obj.get("rubric") or [],
                        "ordering_tested": q_obj.get("ordering_tested") or [],
                    }
                )
        return queries

    def judge_shape(self) -> JudgeShape:
        """BEAM is scored per rubric nugget, not by one verdict against a reference answer.

        This is the benchmark's own protocol, and the reason the shape seam exists. It also
        decides judgeability: a BEAM question is gradeable when it has a rubric, which every one
        of them does -- where "has a gold answer" left
        `instruction_following` and `preference_following` graded against a behaviour description.

        `event_ordering` is dispatched to rank correlation rather than nugget averaging: its rubric
        is an ORDERED list, so averaging it would score a reversed answer identically to a correct
        one.
        """
        return BeamJudgeShape()

    def config_for_receipt(self) -> dict[str, Any]:
        """Adds which BEAM protocol produced the run.

        BEAM's reference harness contradicts BEAM's paper in three places, so "a BEAM score" names
        two different metrics; `spec` is the documented one, which is the one memrank targets.
        In the HASHED config, not
        `receipt.extra`: an artifact has to be classifiable years later by someone who did not run
        it, and two runs differing only in protocol must not collide on one config hash.

        `time_anchors` and `context_policy` are stated explicitly rather than left to the config
        hash: a pre- and post-2026-08-13 artifact both say `beam_protocol: spec`, and the run
        protocol they differ on (anchors rendered + reader capped vs role+content + uncapped)
        must be readable off the artifact, not reconstructed from a hash-version changelog.
        """
        return {**super().config_for_receipt(), "beam_protocol": "spec",
                "time_anchors": "none", "context_policy": self.context_policy}

    def load(self) -> list[BenchmarkUnit]:
        """Return one BenchmarkUnit per conversation in the chosen tier."""
        raw = self._load_raw()
        if self.slice == "smoke":
            raw = raw[:1]
        elif self.slice == "mini":
            raw = raw[:5]
        units: list[BenchmarkUnit] = []
        for item in raw:
            conv_id = str(item.get("conversation_id") or f"conv_{len(units)}")
            # `user_profile.user_info` is deliberately NOT ingested. It is a multi-line dossier --
            # name, age, gender, location, profession -- that BEAM's own baselines never see (they
            # are given `chat` only), and it would hand every engine a free answer sheet for the
            # information-extraction probes. Same protocol rule as the anchors: engines get the
            # conversation, nothing about it.
            documents = self._conv_documents(conv_id, item.get("chat") or [])
            queries = self._conv_queries(conv_id, item)
            units.append(
                BenchmarkUnit(
                    unit_id=conv_id,
                    isolation_id=conv_id,
                    documents=documents,
                    queries=queries,
                    metadata={"tier": self.tier},
                )
            )
        return units

    # ------------------------------------------------------------------ #
    # Scoring
    # ------------------------------------------------------------------ #

    def score(self, unit: BenchmarkUnit, responses: list[AdapterResponse]) -> dict[str, Any]:
        """Report what was asked and what came back; BEAM's verdict is the rubric judge's.

        No composite. The substring proxy that used to live here asked whether an ability's gold
        answer appeared verbatim in retrieved text, which BEAM's prose ``ideal_response`` cannot
        satisfy -- the module has always declared it structurally invalid -- and it forced its own
        distortions: abstention probes had to be excluded from the denominator entirely, because
        a negative carrying no ``forbidden_spans`` scores an unconditional hit. Withholding the
        number was never enough while it was still computed and still shown per-ability.

        ``judge_shape()`` is where BEAM is actually scored: one call per rubric nugget on
        {0, 0.5, 1}, averaged within the question.
        """
        answered = {r.query_id for r in responses if r.documents}
        per_ability: dict[str, int] = defaultdict(int)
        for query in unit.queries:
            per_ability[query["category"]] += 1
        return {
            "composite": None,
            "per_ability_counts": dict(per_ability),
            "n_queries": len(unit.queries),
            "n_abstention": per_ability.get(_ABSTENTION, 0),
            "n_retrieved_nothing": sum(1 for q in unit.queries if q["id"] not in answered),
            "metric": "judged_nugget_rubric (judge required)",
        }

    # ------------------------------------------------------------------ #
    # Reporting
    # ------------------------------------------------------------------ #

    def report_template(self) -> str:
        return (
            "# BEAM-{tier} report -- {adapter}\n"
            "\n"
            "- Composite: **{composite}**\n"
            "- Dataset: {dataset_version}\n"
            "- k: {k}\n"
            "\n"
            "## Per-ability score\n"
            "{per_ability_md}\n"
        )
