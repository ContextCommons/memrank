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
"""LoCoMo benchmark (snap-research/locomo).

10 multi-session conversations x ~199 QA pairs each = 1986 questions in
4 scored categories -- in the DATA's integer order: 1=multi-hop (282),
2=temporal (321), 3=open-domain (96), 4=single-hop (841). Adversarial
questions (category 5, 446) are excluded: 444 of them ship no `answer`
field, so the official scorer itself cannot run on them.

LoCoMo's published protocol scores the GENERATED answer against the reference --
token-level F1, or a binary LLM-judge verdict. This module therefore ships no
self-contained quality score: ``score()`` reports the question counts and nothing
else, and quality comes from ``--judge``.

Two retrieval-only proxies were tried here before and both were withdrawn. Recall
of gold doc ids failed because engines return their own opaque ids and do not echo
``doc_id`` back, so it scored a constant 0.0 for every engine. Its replacement asked
whether the reference answer appeared as a verbatim SPAN of a retrieved document,
which LoCoMo does not define and which its derived answers ("3 times", "2022")
cannot satisfy at any retrieval quality. A proxy that measures storage literalness
is worse than no number, because it ranks.

``gold_answers`` and ``evidence_doc_ids`` are still loaded: the first is what the
judge grades against, the second is LoCoMo's own evidence annotation.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from memrank.benchmarks.answer_prompts import LOCOMO_READER
from memrank.core import AdapterResponse, Benchmark, BenchmarkUnit, Document, EvalInfo

_DATA_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"

# The dataset has exactly one upstream version (single data commit, 2024-08-10), so the pin is
# stable. Verified on every canonical load -- a cache that drifted from upstream, or an upstream
# that silently revised, must fail loudly, never score (protocol-fidelity section 6 item 5).
_DATA_SHA256 = "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4"

# Integer->name mapping in the DATA's order (authority: snap-research task_eval/evaluation.py
# dispatch), NOT the paper's prose order (single/multi/temporal/open) -- the two differ, and
# applying the prose order to the integers mislabels three of four categories.
_CATEGORY_NAMES = {
    1: "multi-hop",
    2: "temporal",
    3: "open-domain",
    4: "single-hop",
}
_SKIP_CATEGORIES = {5}  # adversarial: 444/446 released items have no `answer` field

# The canonical file's shape, asserted at load. These five counts are LoCoMo's fingerprint --
# any harness printing per-category numbers with different n has mislabeled or refiltered it.
_FINGERPRINT = {"conversations": 10, "questions": 1986,
                "per_category": {1: 282, 2: 321, 3: 96, 4: 841, 5: 446}}


class LoCoMoBenchmark(Benchmark):
    """LoCoMo benchmark loader + retrieval-recall scorer.

    Set ``LOCOMO_DATA_PATH`` to use a local JSON file (matches the path the
    AMB harness uses, so existing caches Just Work). Otherwise the file is
    downloaded from snap-research/locomo on first use and cached under
    ``~/.memrank/datasets/locomo/``.
    """

    name = "locomo"
    dataset_version = "snap-research/locomo10@v1"
    # 2026-08-13 (v1): the category integer->name mapping was corrected (3 of 4 labels were the
    # paper's prose order, not the data's) and all four categories became judge-valid -- the
    # judged denominator moved 603->1,540 and every earlier per-category label is void. Scores
    # across this boundary are not comparable.
    # 2026-08-14 (v2): sessions load in chronological order (the old lexicographic sort ingested
    # session_9 after session_19 and anchored query_timestamp mid-conversation), and
    # Document.content is the dated speaker-attributed rendering instead of a JSON turn blob --
    # both found by the M3 hand verification of judged temporal verdicts. Ingest input changed
    # for every engine: v1 scores are not comparable to v2.
    VERSION = 2
    info = EvalInfo(unit="conversation",
                    units_declared="10 multi-session conversations, ~199 QA pairs each "
                                   "(adversarial excluded)",
                    slices=("smoke", "mini"))
    # LoCoMo's published protocol scores the GENERATED answer: token-F1 against the reference, or
    # a binary LLM-judge verdict. Neither asks whether the reference text appears verbatim inside
    # retrieved context -- which is what the substring proxy measured, and what LoCoMo's derived
    # answers ("3 times", "2022") can never satisfy however well retrieval worked. The proxy is
    # therefore not computed here at all; quality on LoCoMo requires `--judge`.
    substring_recall_supported = False
    composite_rankable = False
    quality_metric = "judged_answer_correctness"
    answer_prompt = LOCOMO_READER

    def __init__(self, slice: str | None = None, k: int = 10) -> None:
        self.slice = slice  # "smoke" | "mini" | None (full)
        self.k = k
        env_path = os.environ.get("LOCOMO_DATA_PATH")
        self._local_path: Path | None = Path(env_path) if env_path else None
        # Question text is public only when using the canonical dataset (no local override).
        self.question_text_public = env_path is None

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #

    def _data_path(self) -> Path:
        if self._local_path and self._local_path.exists():
            return self._local_path
        from memrank.benchmarks import cache_root, dataset_download_notice

        cache = cache_root() / "locomo"
        cache.mkdir(parents=True, exist_ok=True)
        path = cache / "locomo10.json"
        if not path.exists():
            with dataset_download_notice("LoCoMo", _DATA_URL, path):
                urllib.request.urlretrieve(_DATA_URL, path)
        # On EVERY read, not only after download: an indefinitely-cached file that upstream (or
        # the disk) changed must fail before a single unit loads from it.
        self._verify_digest(path)
        return path

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @classmethod
    def _verify_digest(cls, path: Path) -> None:
        actual = cls._file_sha256(path)
        if actual != _DATA_SHA256:
            raise RuntimeError(
                f"LoCoMo dataset sha256 mismatch at {path}: expected {_DATA_SHA256}, "
                f"got {actual}. Upstream drifted or the cache is corrupt; delete the file "
                "to re-download, or point LOCOMO_DATA_PATH at a deliberate local variant.")

    @staticmethod
    def _verify_fingerprint(raw: list[dict[str, Any]]) -> None:
        counts: dict[int, int] = defaultdict(int)
        for item in raw:
            for qa in item.get("qa", []):
                counts[qa.get("category")] += 1
        actual = {"conversations": len(raw), "questions": sum(counts.values()),
                  "per_category": dict(counts)}
        if actual != _FINGERPRINT:
            raise RuntimeError(
                f"LoCoMo dataset fingerprint mismatch: expected {_FINGERPRINT}, got {actual}")

    def _load_raw(self) -> list[dict[str, Any]]:
        path = self._data_path()
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        # Canonical data only -- LOCOMO_DATA_PATH is the deliberate escape hatch, and it is
        # already marked (question_text_public flips false, and the receipt digest differs).
        if path is not self._local_path:
            self._verify_fingerprint(raw)
        return raw

    def config_for_receipt(self) -> dict[str, Any]:
        """The digest of the file actually read joins the HASHED config: two artifacts share a
        config hash only when they scored identical data (protocol-fidelity section 6 item 5)."""
        return {**super().config_for_receipt(),
                "locomo_dataset_sha256": self._file_sha256(self._data_path())}

    def judge_shape(self):
        """All four scored categories are plain single-gold QA, judged binary.

        The declaration IS the loader's vocabulary (`_CATEGORY_NAMES.values()`), one object, so
        the category names the loader emits and the set the judge gate accepts cannot drift --
        drifting apart is exactly how 55% of this benchmark went silently unjudged (audit F2).
        """
        from memrank.judging.shape import BinaryJudgeShape
        return BinaryJudgeShape(frozenset(_CATEGORY_NAMES.values()))

    @staticmethod
    def _session_keys(conv: dict[str, Any]) -> list[str]:
        """Session keys in CHRONOLOGICAL order -- numeric, never lexicographic.

        Plain ``sorted()`` ordered ``session_9`` after ``session_19``, so conversations were
        ingested out of chronological order and the "last session" scan behind
        ``query_timestamp`` landed mid-conversation (conv-26: 17 July instead of 22 October).
        Found by the M3 hand verification: the reader anchored every relative date to the
        wrong "now".
        """
        return sorted(
            (k
             for k in conv
             if k.startswith("session_") and not k.endswith("_date_time")
             and isinstance(conv[k], list)),
            key=lambda k: int(k.rsplit("_", 1)[1]),
        )

    @staticmethod
    def _parse_date(value: str | None) -> str | None:
        """Parse LoCoMo's '1:56 pm on 8 May, 2023' into ISO-8601."""
        if not value:
            return None
        try:
            dt = datetime.strptime(value, "%I:%M %p on %d %B, %Y")
            return dt.replace(tzinfo=timezone.utc).isoformat()
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _turns_to_messages(turns: list[Any], speaker_a: str) -> list[dict[str, Any]]:
        """Render LoCoMo turns as role-tagged, speaker-attributed messages.

        LoCoMo is a dialogue between two NAMED humans, and its questions name them
        ("When did Caroline go to the support group?"), so attribution is part of the
        answer rather than presentation. Three fields, each load-bearing:

        ``role`` is a wire-valid API role -- engines expect OpenAI-shaped messages, and
        ``"Caroline"`` is not a role. Speaker A maps to ``user``, B to ``assistant``.

        ``content`` carries the speaker name inline because adapters forward only role
        and content (memrank/adapters/mem0.py), so a name held anywhere else is dropped
        before it reaches the engine. This is also what Hindsight's ingest guidance asks
        for: text conveying who said what.

        ``speaker`` is kept structured so an adapter can render better than the prefix.

        Image captions are part of the conversation, not decoration: 20.8% of turns carry a
        ``blip_caption`` (1,226 of 5,882), and some gold answers appear ONLY there. Dropping
        them cost ~3 points of recall when this helper first shipped without them. The image-search
        ``query`` field is excluded by contrast -- it describes how the dataset was built
        and holds no answer that is not already in the text or the caption. ``dia_id`` and
        ``img_url`` are join keys and addresses, never content.
        """
        messages: list[dict[str, Any]] = []
        for turn in turns:
            if not isinstance(turn, dict):
                continue
            speaker = str(turn.get("speaker") or "")
            text = str(turn.get("text") or "")
            caption = str(turn.get("blip_caption") or "")
            if caption:
                text = f"{text} [shared an image: {caption}]" if text else f"[shared an image: {caption}]"
            messages.append({
                "role": "user" if speaker == speaker_a else "assistant",
                "content": f"{speaker}: {text}" if speaker else text,
                "speaker": speaker,
            })
        return messages

    def load(self) -> list[BenchmarkUnit]:
        """Return one BenchmarkUnit per LoCoMo conversation."""
        raw = self._load_raw()
        if self.slice == "smoke":
            raw = raw[:1]
        elif self.slice == "mini":
            raw = raw[:3]
        units: list[BenchmarkUnit] = []
        for item in raw:
            sample_id = item["sample_id"]
            conv = item["conversation"]
            session_keys = self._session_keys(conv)
            speaker_a = conv.get("speaker_a", "A")
            speaker_b = conv.get("speaker_b", "B")
            documents, dia_to_session = self._conv_documents(
                conv, sample_id, session_keys, speaker_a, speaker_b)
            queries = self._conv_queries(item, conv, sample_id, session_keys, dia_to_session)
            units.append(
                BenchmarkUnit(
                    unit_id=sample_id,
                    isolation_id=sample_id,
                    documents=documents,
                    queries=queries,
                    metadata={"speaker_a": speaker_a, "speaker_b": speaker_b},
                )
            )
        return units

    def _conv_documents(self, conv: dict[str, Any], sample_id: str, session_keys: list[str],
                        speaker_a: str, speaker_b: str
                        ) -> tuple[list[Document], dict[str, str]]:
        """One Document per session, plus the dia_id -> session join map for evidence ids."""
        documents: list[Document] = []
        dia_to_session: dict[str, str] = {}
        for sk in session_keys:
            turns = conv.get(sk) or []
            if not turns:
                continue
            for turn in turns:
                if isinstance(turn, dict) and "dia_id" in turn:
                    dia_to_session[turn["dia_id"]] = sk
            # Two date fields on purpose: the ISO one is machine-readable and goes to
            # engines that accept a native ingest timestamp; the raw string is the
            # dataset's OWN wording ("1:56 pm on 8 May, 2023") and rides in `context`,
            # which is the only channel most engines have. Passing the source's format
            # through is deliberate -- reformatting it to resemble a gold answer would
            # be tuning to the scorer.
            raw_when = conv.get(f"{sk}_date_time")
            # Comma, not " on ": LoCoMo's own string already reads "1:56 pm on 8 May, 2023".
            when = f", {raw_when}" if raw_when else ""
            header = f"Conversation between {speaker_a} and {speaker_b} ({sk}){when}"
            messages = self._turns_to_messages(turns, speaker_a)
            # `content` is the dated, speaker-attributed rendering -- the same text shape the
            # official harness put in front of its readers -- NOT a JSON blob (audit F10). The
            # date in the header is load-bearing: it is the only in-band channel by which
            # "Use DATE of CONVERSATION" is answerable for content-reading engines and for the
            # full-context arm. Found by the M3 hand verification.
            documents.append(
                Document(
                    id=f"{sample_id}_{sk}",
                    content="\n".join([header] + [m["content"] for m in messages]),
                    messages=messages,
                    user_id=sample_id,
                    timestamp=self._parse_date(raw_when),
                    context=header,
                )
            )
        return documents, dia_to_session

    def _conv_queries(self, item: dict[str, Any], conv: dict[str, Any], sample_id: str,
                      session_keys: list[str],
                      dia_to_session: dict[str, str]) -> list[dict[str, Any]]:
        """One query per non-adversarial QA pair, dated at the last session."""
        queries: list[dict[str, Any]] = []
        last_ts: str | None = None
        for sk in reversed(session_keys):
            last_ts = self._parse_date(conv.get(f"{sk}_date_time"))
            if last_ts:
                break
        for qi, qa in enumerate(item.get("qa", [])):
            cat_int = qa.get("category")
            if cat_int in _SKIP_CATEGORIES:
                continue
            question = qa.get("question", "")
            if cat_int == 2:
                # Temporal: the official protocol appends this to the QUESTION TEXT itself
                # (task_eval/gpt_utils.py), so it reaches the reader as published. Declared
                # side-effect: memrank's query text also drives retrieval, which the official
                # RAG harness's retrieval query did not carry (protocol-fidelity section 1.4).
                question += " Use DATE of CONVERSATION to answer with an approximate date."
            # LoCoMo answers can be numeric (counts, years) -- coerce to str.
            answer = str(qa.get("answer", ""))
            evidence = qa.get("evidence") or []
            gold_session_keys = {dia_to_session[e] for e in evidence if e in dia_to_session}
            queries.append(
                {
                    "id": f"{sample_id}_q{qi}",
                    "text": question,
                    "user_id": sample_id,
                    "evidence_doc_ids": sorted(f"{sample_id}_{sk}" for sk in gold_session_keys),
                    "gold_answers": [answer],
                    "category": _CATEGORY_NAMES.get(cat_int, str(cat_int)),
                    "query_timestamp": last_ts,
                }
            )
        return queries

    # ------------------------------------------------------------------ #
    # Scoring
    # ------------------------------------------------------------------ #

    def score(self, unit: BenchmarkUnit, responses: list[AdapterResponse]) -> dict[str, Any]:
        """Report what was asked and what came back; the quality verdict is the judge's.

        No composite. This used to be recall of the reference answer as a verbatim SPAN of a
        retrieved document -- a metric LoCoMo does not define, and one its numeric and derived
        answers can never satisfy. Publishing it made every engine's score a property of how
        literally it stored transcripts. The counts stay because a denominator nobody can see
        is how the previous broken metric survived unnoticed.
        """
        from memrank.metrics.evidence_recall import evidence_recall

        answered = {r.query_id for r in responses if r.documents}
        per_category: dict[str, int] = defaultdict(int)
        for q in unit.queries:
            per_category[q["category"]] += 1
        # LoCoMo's own retrieval metric (recall_acc at session granularity), judge-free and
        # deterministic -- a retrieval diagnostic reported beside the counts, never a composite
        # (docs/superpowers/specs/2026-08-13-evidence-recall-design.md).
        return {"composite": None, "per_category_counts": dict(per_category),
                "n_queries": len(unit.queries),
                "n_retrieved_nothing": sum(1 for q in unit.queries if q["id"] not in answered),
                "metric": "judged_answer_correctness (judge required)",
                **evidence_recall(unit, responses)}

    # ------------------------------------------------------------------ #
    # Reporting
    # ------------------------------------------------------------------ #

    def rollup(self, per_unit_scores: list[dict[str, Any]], *,
               ranked: bool = True) -> dict[str, Any]:
        """`recall_acc` as one number, micro-averaged over the queries that HAVE evidence.

        Same gap LongMemEval had: the metric was computed per conversation and never reduced, so
        the only official LoCoMo metric no other harness computes was also one memrank could not
        read. Weighted by each unit's `n_queries_with_evidence`, so a 199-question conversation
        outweighs a 96-question one and a unit with no evidence contributes nothing.
        """
        from memrank.metrics.retrieval import weighted_mean

        rows = [(u.get("evidence_recall"), int(u.get("n_queries_with_evidence") or 0))
                for u in per_unit_scores]
        return {"evidence_recall": weighted_mean(rows),
                "n_queries_with_evidence": sum(w for _, w in rows)}

    def report_template(self) -> str:
        return (
            "# LoCoMo report -- {adapter}\n"
            "\n"
            "- Composite: **{composite}**\n"
            "- Dataset: {dataset_version}\n"
            "- Slice: {slice}\n"
            "- k: {k}\n"
            "\n"
            "## Per-category\n"
            "{per_category_md}\n"
        )
