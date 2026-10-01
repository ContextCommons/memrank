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
"""LongMemEval benchmark (xiaowu0162/longmemeval-cleaned).

~500 questions across 6 question types, each with its own haystack of
conversation sessions. Each item is its own isolation unit (one bank per
question).

LongMemEval's published metric is QA accuracy: generate an answer from the
retrieved context, then grade it with an LLM judge. The paper rejects string
matching outright -- "as the correct answers can take flexible forms, an exact
matching strategy can result in inaccurate evaluations" -- and instructs its judge
to weigh semantic and temporal consistency instead. So this module ships no
self-contained quality score; ``score()`` reports counts, and quality comes from
``--judge``.

Two proxies were tried here and both were withdrawn. Gold-session-id recall scored
a constant 0.0 because engines return their own opaque ids. Its replacement asked
whether the reference answer appeared as a verbatim SPAN of retrieved text -- the
very strategy the paper rejects, and one that scored every abstention item 0 for
every engine by construction, since an abstention has no answer text to find.

``gold_answers`` and the ``has_answer`` session annotations are still loaded: the
first is what the judge grades against, the second is the dataset's own evidence
labelling.

**Ingest is sequential, in array order, matching the official harness.** The paper's task
definition is that a system "parse the dynamic interactions online for memorization, and answer
the question after all the interaction sessions", and the runner ingests one document at a time
in list order before querying. That list is the file's own `haystack_sessions` order, which is
sorted by DATE but not by time of day: on 211 of 500 instances the times run backwards within a
day. This is deliberately NOT re-sorted -- `run_generation.py` walks the sessions in the same
array order and re-sorts only the RETRIEVED chunks at read time (line 225), so array order is
the faithful ingest and chronological order would be the divergence.

**Dates are rendered into the text, and ordering is not enforced.** The official harness renders
``Session Date: {date}`` above every session and then sorts retrieved items by timestamp
(``run_generation.py:225``). It can sort because it *is* the retriever. memrank cannot: real
engines return their own documents -- ``mem0.py:566`` builds ``Document(id, content, user_id,
metadata)``, and no adapter carries our ``Document.timestamp`` back out -- so only the control
arms (which return our own objects, already in ingest order) could be sorted, and sorting just
those would make context order engine-dependent. That is the shape of defect the opaque-handle
work exists to remove, so it is not reintroduced here. The date therefore travels in ``content``,
which every engine preserves or paraphrases honestly, and the reader is left to order by it.
Recorded as a declared divergence rather than half-solved.
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

from memrank.benchmarks.answer_prompts import LONGMEMEVAL_READER
from memrank.core import AdapterResponse, Benchmark, BenchmarkUnit, Document, EvalInfo

_DATA_URL = (
    "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned"
    "/resolve/main/longmemeval_s_cleaned.json"
)

#: Digest of `longmemeval_s_cleaned.json` (277,383,467 bytes), the artifact the official
#: LongMemEval protocol runs against. Unlike LoCoMo -- one
#: upstream commit, ever -- this dataset MOVES: the original release now carries a deprecation
#: banner, the 2025/09 cleanup renamed the files, and a third-party re-review found 77 of 500
#: records (15.4%) wrong even in the cleaned file. A tag resolving to "whatever HuggingFace
#: serves today" is not a version.
_DATA_SHA256 = "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"

#: The six official `question_type` values. **Abstention is not among them**: it is an orthogonal
#: `question_id` suffix (`_abs`), and its 30 items keep their parent type here while being graded
#: by their own prompt and reported separately (card section 2, section 3). Treating abstention as a seventh
#: type is the single most common third-party error in this benchmark's literature.
_QUESTION_TYPES = [
    "single-session-user",
    "single-session-assistant",
    "multi-session",
    "temporal-reasoning",
    "knowledge-update",
    "single-session-preference",
]

#: The one date format the dataset ships, e.g. "2023/05/30 (Tue) 23:40". `%a` parses the weekday
#: rather than the old code's discard-everything-after-"(", which took the time with it.
_DATE_FORMAT = "%Y/%m/%d (%a) %H:%M"

#: Suffix marking a false-premise question. The official grader dispatches on this BEFORE
#: question_type (`evaluate_qa.py::get_anscheck_prompt`), so it takes precedence.
_ABSTENTION_SUFFIX = "_abs"

#: The canonical file's shape, asserted at load. Any harness printing per-type numbers with
#: different n has refiltered or mislabeled it.
_FINGERPRINT = {"temporal-reasoning": 133, "multi-session": 133, "knowledge-update": 78,
                "single-session-user": 70, "single-session-assistant": 56,
                "single-session-preference": 30}

#: Slice sizes, stratified across (question_type, is-abstention) rather than taken as `raw[:N]`.
#: The file is GROUPED BY TYPE and the `_abs` items sit at the end of each group, so the LoCoMo
#: idiom (safe there -- its unit is a conversation carrying every category) yielded slices that
#: were 100% single-session-user with zero abstention items. Eight covers the six types plus two
#: abstention items, so a smoke run exercises every judge path; twenty-four is roughly two per
#: stratum.
_SLICE_SIZES = {"smoke": 8, "mini": 24}


def _mean(scores: list[float]) -> float | None:
    """None for an empty bucket, never 0.0 -- "not measured" and "measured zero" differ."""
    return sum(scores) / len(scores) if scores else None


class _LongMemEvalJudgeShape:
    """Mixin supplying LongMemEval's two extra published aggregates.

    Lives here rather than in `judge_shape.py` because the knowledge is protocol knowledge: what
    LongMemEval publishes belongs beside how LongMemEval loads.
    """

    def aggregates(self, per_category: dict[str, list[float]],
                   per_prompt_key: dict[str, list[float]]) -> dict[str, Any]:
        """Task-averaged accuracy and abstention accuracy, beside the micro-mean.

        Task-averaged is the UNWEIGHTED mean of the six type means, so a 30-question type counts
        as much as a 133-question one. That is the protocol's definition and it is also why micro
        is the headline (the ADR): the macro hands 16.7% of the number to
        `single-session-preference`, which is the type with the weakest judge-human agreement.

        Abstention is reported over its own 30 items while those items ALSO remain inside their
        parent type's cell -- the protocol counts them twice on purpose, and a reader who removes
        the overlap is reporting something the benchmark does not define.
        """
        per_type = [_mean(scores) for scores in per_category.values() if scores]
        abstention = per_prompt_key.get("abstention") or []
        return {
            "answer_correctness_task_averaged": _mean([m for m in per_type if m is not None]),
            "abstention_accuracy": _mean(abstention),
            "n_abstention": len(abstention),
        }


class LongMemEvalBenchmark(Benchmark):
    """LongMemEval-S loader + per-question-type retrieval scorer."""

    name = "longmemeval"
    dataset_version = "xiaowu0162/longmemeval-cleaned@s"
    # 2026-08-14 (v1): documents are keyed by opaque positional handles instead of by the
    # dataset's own session ids, which prefix every evidence session with `answer` and so handed
    # every engine a noise-free retrieval oracle (audit F1). Scores across this boundary are not
    # merely non-comparable -- every VERSION-0 LongMemEval number is INVALID, by an amount that
    # varies per engine according to whether it indexed document ids and context strings.
    # 2026-08-14 (v2): Document.content is the dated transcript rendering instead of a JSON turn
    # blob, and _parse_date keeps the time of day instead of silently flooring every timestamp to
    # midnight (audit F9, F10, F15). The reader now receives information it did not have --
    # `knowledge-update` in particular was not answerable in principle before.
    # 2026-08-14 (v3): the six official judge prompts replace one generic prompt, with abstention
    # dispatched ahead of question type (audit F2-F5, F17). 12% of the benchmark stops being
    # graded against the wrong object and 43% gains the tolerance its type is owed.
    VERSION = 3

    def judge_shape(self):
        """Every question carries one gold answer, judged binary, across the six official
        `question_type` values -- declared from `_QUESTION_TYPES` so the loader's vocabulary and
        the judge gate are one object. An item whose `question_type` falls outside the official
        six (the loader defaults such to "unknown") now raises at judge time instead of being
        silently dropped from coverage.

        Since 2026-08-14 the shape also carries LongMemEval's SIX OFFICIAL judge prompts, keyed
        by `judge_prompt_key` -- the five per-type ones plus abstention, which overrides type.
        They are the only prompts in this repo with a published human-agreement figure (97-98%),
        and adopting them is what stops 12% of the benchmark being graded against the wrong
        object: preference ships a rubric and abstention an explanation, neither of which is an
        answer to match.
        """
        from memrank.judging.prompts import LME_PROMPTS
        from memrank.judging.shape import BinaryJudgeShape

        class _Shape(_LongMemEvalJudgeShape, BinaryJudgeShape):
            pass

        return _Shape(frozenset(_QUESTION_TYPES), prompts=LME_PROMPTS)
    info = EvalInfo(unit="question",
                    units_declared="~500 questions across 6 question types, "
                                   "each with its own haystack",
                    slices=("smoke", "mini"))
    # The published metric is a judged QA accuracy, and the paper explicitly rejects the exact
    # matching the substring proxy performed. No proxy is computed here; quality requires --judge.
    # LongMemEval's own protocol hands the reader everything retrieval returned -- it truncates
    # only at the MODEL WINDOW (~126k for GPT-4o), never at a fairness budget -- so it takes the
    # exception docs/methodology.md section "Retrieval token budget" defines and BEAM already uses.
    #
    # This was not a preference. Measured on a smoke run at the 5,000-token default: retrieved
    # sessions average 2,935 tokens, so k=10 needs ~29k to deliver what it retrieves, the cap bit
    # on 8/8 queries, and the evidence reached the reader on **0 of 5** scoreable questions --
    # while retrieval itself scored recall_all@10 = 1.0. A judged run under that cap would have
    # measured the truncation, not the memory.
    #
    # What is given up is budget-normalisation against LoCoMo rows, which
    # decision-beam-runs-its-own-protocol already accepts as the price. Token efficiency remains
    # visible as its own axis (`context_tokens_mean`, `est_dollars_per_query`), which is how the
    # whole field reports it -- Zep's own headline is 71.2% at 1.6k against full-context's 60.2%
    # at 115k.
    context_policy = "uncapped"
    answer_prompt = LONGMEMEVAL_READER

    substring_recall_supported = False
    composite_rankable = False
    quality_metric = "judged_answer_correctness"

    def __init__(self, slice: str | None = None, k: int = 10) -> None:
        self.slice = slice
        self.k = k
        env = os.environ.get("LONGMEMEVAL_DATA_PATH")
        self._local_path: Path | None = Path(env) if env else None
        # Question text is public only when using the canonical dataset (no local override).
        self.question_text_public = env is None

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #

    def _data_path(self) -> Path:
        if self._local_path and self._local_path.exists():
            return self._local_path
        from memrank.benchmarks import cache_root, dataset_download_notice

        cache = cache_root() / "longmemeval"
        cache.mkdir(parents=True, exist_ok=True)
        path = cache / "longmemeval_s_cleaned.json"
        if not path.exists():
            with dataset_download_notice("LongMemEval", _DATA_URL, path):
                urllib.request.urlretrieve(_DATA_URL, path)
        # On EVERY read, not only after download: an indefinitely-cached file that upstream (or
        # the disk) changed must fail before a single unit loads from it. This dataset has
        # already shipped three named revisions; silent drift is the expected failure, not a
        # hypothetical one.
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
                f"LongMemEval dataset sha256 mismatch at {path}: expected {_DATA_SHA256}, "
                f"got {actual}. Upstream drifted or the cache is corrupt; delete the file to "
                "re-download, or point LONGMEMEVAL_DATA_PATH at a deliberate local variant.")

    @staticmethod
    def _question_type(item: dict[str, Any]) -> str:
        """One item's question type, as a string, so it can key a count and a stratum.

        A record missing the field keys under the empty string, which no declared type uses. It
        is neither dropped nor folded into a real type: `_verify_fingerprint` reports it as the
        mismatch it is, and `_strata` gives it a stratum of its own -- the same treatment an
        unrecognised type already gets.
        """
        return str(item.get("question_type") or "")

    @staticmethod
    def _verify_fingerprint(raw: list[dict[str, Any]]) -> None:
        counts: dict[str, int] = defaultdict(int)
        for item in raw:
            counts[LongMemEvalBenchmark._question_type(item)] += 1
        if dict(counts) != _FINGERPRINT:
            raise RuntimeError(
                f"LongMemEval dataset fingerprint mismatch: expected {_FINGERPRINT}, "
                f"got {dict(counts)}")

    def _load_raw(self) -> list[dict[str, Any]]:
        path = self._data_path()
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        # Canonical data only -- LONGMEMEVAL_DATA_PATH is the deliberate escape hatch, and it is
        # already marked (question_text_public flips false, and the receipt digest differs).
        if path is not self._local_path:
            self._verify_fingerprint(raw)
        return raw

    def config_for_receipt(self) -> dict[str, Any]:
        """The digest of the file actually read joins the HASHED config: two artifacts share a
        config hash only when they scored identical data. That matters more here than on any
        other benchmark -- see `_DATA_SHA256` on why this artifact moves."""
        return {**super().config_for_receipt(),
                "longmemeval_dataset_sha256": self._file_sha256(self._data_path()),
                # Stated explicitly rather than left to the hash, on BEAM's precedent
                # (beam.py:config_for_receipt): an artifact has to be classifiable years later by
                # someone who did not run it, and a capped and an uncapped run of this benchmark
                # are different experiments that must not read alike.
                "context_policy": self.context_policy}

    @staticmethod
    def _strata(raw: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        """Group ``raw`` by (question_type, is-abstention), non-abstention strata first.

        Two dimensions, not one, because BOTH are positional in the file: types are grouped, and
        within a type the `_abs` items sit at the END (ranks 56-127 of their group on the
        canonical artifact). Stratifying by type alone would produce a slice that covers all six
        types and still never reaches the abstention grader.

        Non-abstention strata come first so a slice smaller than the stratum count still spans
        the type vocabulary before it starts spending slots on abstention.
        """
        groups: dict[tuple[str, bool], list[dict[str, Any]]] = defaultdict(list)
        for item in raw:
            is_abs = str(item.get("question_id", "")).endswith(_ABSTENTION_SUFFIX)
            groups[(LongMemEvalBenchmark._question_type(item), is_abs)].append(item)
        declared = list(_QUESTION_TYPES) + \
                   [t for t, _ in groups if t not in _QUESTION_TYPES]
        keys = [(t, False) for t in declared] + [(t, True) for t in declared]
        return [groups[k] for k in keys if groups.get(k)]

    @classmethod
    def _stratified(cls, raw: list[dict[str, Any]], size: int) -> list[dict[str, Any]]:
        """Take ``size`` items round-robin across the strata, preserving file order within each.

        Deterministic by construction -- no sampling, no seed -- so a slice is reproducible across
        runs and across machines. Returns everything when ``size`` exceeds the corpus rather than
        padding or raising.
        """
        strata = cls._strata(raw)
        taken: list[dict[str, Any]] = []
        for rank in range(max((len(s) for s in strata), default=0)):
            for stratum in strata:
                if rank < len(stratum):
                    taken.append(stratum[rank])
                    if len(taken) == size:
                        return taken
        return taken

    @staticmethod
    def _parse_date(value: str) -> str | None:
        """Return ISO-8601 from LongMemEval's ``YYYY/MM/DD (Ddd) HH:MM`` format.

        One format, because the dataset ships exactly one. The previous implementation split on
        "(" -- discarding the weekday AND the time -- then tried four formats behind a bare
        `except`, so `%Y/%m/%d` always succeeded and every timestamp in the benchmark was
        MIDNIGHT. That made same-day sessions indistinguishable to any engine sorting by time
        (211 of 500 instances have intra-day inversions) and stripped the hour from the reader's
        `Current date` block.

        Raises on anything else rather than returning None: a format we do not recognise is a
        dataset change we must notice, and silently degrading is what hid this for months
        (AGENTS.md: no fallback or degraded modes).
        """
        if not value:
            return None
        try:
            parsed = datetime.strptime(value.strip(), _DATE_FORMAT)
        except ValueError as exc:
            raise ValueError(
                f"LongMemEval date {value!r} does not match {_DATE_FORMAT!r}. The canonical "
                "artifact uses that format for every question_date and haystack_date; a "
                "mismatch means the dataset changed shape.") from exc
        return parsed.replace(tzinfo=timezone.utc).isoformat()

    @staticmethod
    def _aligned_sessions(item: dict[str, Any]) -> list[tuple[list[Any], str, str]]:
        """The three positionally-aligned haystack lists, zipped and truncated to the shortest.

        `haystack_sessions[i]`, `haystack_session_ids[i]` and `haystack_dates[i]` describe one
        session; a short list would silently mis-pair a session with another's date.
        """
        sessions = item.get("haystack_sessions", []) or []
        session_ids = item.get("haystack_session_ids", []) or []
        dates = item.get("haystack_dates", []) or []
        n = min(len(sessions), len(session_ids), len(dates))
        return list(zip(sessions[:n], session_ids[:n], dates[:n], strict=True))

    @staticmethod
    def _strip_evidence_labels(session_turns: list[Any]) -> list[Any]:
        """Turns with `has_answer` removed -- the dataset's evidence label, which is harness-side.

        The official harness pops exactly this key from every turn of every retrieved chunk
        before prompting (run_generation.py:178-190). It is the easiest leak to ship by accident,
        because the flag sits INSIDE the turn dicts a naive ingestion would pass through whole.
        """
        return [
            {k: v for k, v in turn.items() if k != "has_answer"} if isinstance(turn, dict)
            else turn
            for turn in session_turns
        ]

    @staticmethod
    def _has_user_target(item: dict[str, Any]) -> bool:
        """Whether any evidence turn is user-side, which is what retrieval can target.

        section 5.1: "when sessions or rounds are used as the key, we only keep the USER-SIDE
        utterances", so a question whose evidence lives in an assistant turn has nothing for a
        user-side index to retrieve, and the official retrieval scripts exclude it.

        Verified on the canonical file: exactly 51 non-abstention questions fail this, and all 51
        are `single-session-assistant` -- which is the shape the rule predicts.
        """
        return any(turn.get("has_answer") and turn.get("role") == "user"
                   for session in (item.get("haystack_sessions") or [])
                   for turn in session if isinstance(turn, dict))

    @staticmethod
    def _render(index: int, date_str: str, messages: list[dict[str, Any]]) -> tuple[str, str]:
        """The session's ``(context header, content)`` -- a dated, role-attributed transcript.

        The date is in-band, and that is load-bearing rather than cosmetic. The official reader
        renders ``Session Date: {date}`` above every session, and two question types are defined
        by it: ``knowledge-update`` asks which of two contradictory facts is CURRENT, and
        ``temporal-reasoning`` is specified as reasoning over the metadata timestamp.
        ``Document.timestamp`` does not reach the reader -- engines return their own documents,
        see the module docstring -- so text is the only reliable channel.

        The dataset's own date wording is passed through unreformatted, on LoCoMo's precedent:
        rewriting it to resemble a gold answer would be tuning to the scorer.
        """
        header = f"Session {index + 1}, {date_str}" if date_str else f"Session {index + 1}"
        lead = f"Session date: {date_str}" if date_str else header
        body = [f"{t.get('role', 'unknown')}: {t.get('content', '')}" for t in messages]
        return header, "\n".join([lead] + body)

    def _documents_for(
        self, item: dict[str, Any],
    ) -> tuple[list[Document], list[str], dict[str, str], dict[str, str]]:
        """One Document per haystack session, the evidence handles, and the two id maps.

        Four values: the documents, the `has_answer` gold handles, handle->session-id, and its
        inverse session-id->handle. The inverse is what resolves `answer_session_ids` in `load()`,
        and the signature said three for as long as it went unmentioned.

        Documents are keyed by OPAQUE POSITIONAL HANDLES, never by the dataset's own session
        ids. LongMemEval prefixes every evidence session id with `answer` and its official
        retrieval scorer keys on that prefix (run_retrieval.py:272), so the id is a noise-free
        oracle: ranking on it alone scores recall_all@10 = 1.000, above the paper's best
        retriever. Passing it to an engine -- as `Document.id`, which adapters forward as metadata
        per docs/system-contract.md:71, or inside `context`, which two adapters map to
        semantically indexed fields -- hands over the answer location.

        The real ids stay harness-side in the returned map, which `load()` puts on the unit's
        metadata so retrieval scoring can still resolve `answer_session_ids`.
        """
        qid = item.get("question_id", "unknown")
        documents: list[Document] = []
        gold_ids: list[str] = []
        handles: dict[str, str] = {}
        by_session: dict[str, str] = {}
        for index, (session_turns, session_id, date_str) in enumerate(
                self._aligned_sessions(item)):
            handle = f"{qid}_s{index:03d}"
            handles[handle] = session_id
            by_session[session_id] = handle
            cleaned_turns = self._strip_evidence_labels(session_turns)
            if any(isinstance(t, dict) and t.get("has_answer") for t in session_turns):
                gold_ids.append(handle)
            messages = [t for t in cleaned_turns if isinstance(t, dict)]
            header, content = self._render(index, date_str, messages)
            documents.append(Document(
                id=handle,
                content=content,
                # Already {role, content} -- pass the structure through rather than letting
                # adapters fall back to parsing the rendered text. No `speaker` key: this is a
                # user/assistant transcript, and inventing speaker names for it would be fiction.
                messages=messages,
                user_id=qid,
                # Position plus date: the date is a legitimate input the protocol hands the
                # reader, the session ID is the oracle it must never see (audit F1).
                context=header,
                timestamp=self._parse_date(date_str),
            ))
        return documents, gold_ids, handles, by_session

    def load(self) -> list[BenchmarkUnit]:
        """Return one BenchmarkUnit per LongMemEval question."""
        raw = self._load_raw()
        if self.slice in _SLICE_SIZES:
            raw = self._stratified(raw, _SLICE_SIZES[self.slice])
        units: list[BenchmarkUnit] = []
        for item in raw:
            qid = item.get("question_id", "unknown")
            qtype = item.get("question_type", "unknown")
            documents, gold_ids, handles, by_session = self._documents_for(item)
            # The OFFICIAL session-level evidence, which is `answer_session_ids` -- not the
            # `has_answer` turn flags `gold_ids` is built from. The two disagree on 62 of 500
            # questions and `has_answer` is always the strict subset, so scoring retrieval
            # against it would make `recall_all@k` EASIER (a missing gold id is one fewer
            # document that must appear in top-k). The README assigns them different jobs:
            # `has_answer` for turn-level recall, `answer_session_ids` for session-level, and
            # memrank's documents are sessions.
            evidence = [by_session[sid] for sid in (item.get("answer_session_ids") or [])
                        if sid in by_session]
            # Abstention overrides question type, matching the official dispatch
            # (`get_anscheck_prompt(..., abstention='_abs' in question_id)`). A
            # temporal-reasoning question whose id ends `_abs` is graded as an abstention, not
            # as a temporal question. `category` keeps the PARENT type, because the protocol
            # counts these 30 items inside their type bucket as well as reporting them
            # separately -- abstention is a slice, not a seventh type.
            is_abs = qid.endswith(_ABSTENTION_SUFFIX)
            queries = [{
                "id": qid,
                "text": item.get("question", ""),
                "user_id": qid,
                "gold_ids": gold_ids,
                # For `_abs` items this is an EXPLANATION of why the premise is false, and for
                # preference items a RUBRIC -- not an answer. The prompt keyed below names it
                # correctly to the judge.
                "gold_answers": [item.get("answer", "")],
                "category": qtype,
                "judge_prompt_key": "abstention" if is_abs else qtype,
                "evidence_doc_ids": evidence,
                # The official retrieval scripts skip the 30 `_abs` items (they refer to
                # non-existing events, so there is no ground-truth location) AND the 51
                # questions with no user-side target turn -- the paper indexes user-side
                # utterances only, so an assistant-side answer has no target to retrieve.
                # Verified on the canonical file: exactly 51, all single-session-assistant.
                # RETRIEVAL ONLY. Issue #16 records the author confirming these filters must
                # not reach the QA denominator, which stays 500; several publications made
                # exactly that error.
                "retrieval_scoreable": not is_abs and self._has_user_target(item),
                # Skips the sufficiency check, which on a question built to have no evidence can
                # only ever answer "no" and was polluting the sufficiency metric with 30 such.
                **({"kind": "negative"} if is_abs else {}),
                "query_timestamp": self._parse_date(item.get("question_date", "")),
            }]
            units.append(BenchmarkUnit(
                unit_id=qid, isolation_id=qid, documents=documents, queries=queries,
                # `session_handles` is HARNESS-SIDE ONLY: unit metadata is never forwarded to an
                # adapter. It exists so retrieval scoring can resolve the official
                # `answer_session_ids` without the engine ever seeing one.
                metadata={"question_type": qtype, "session_handles": handles},
            ))
        return units

    # ------------------------------------------------------------------ #
    # Scoring
    # ------------------------------------------------------------------ #

    def score(self, unit: BenchmarkUnit, responses: list[AdapterResponse]) -> dict[str, Any]:
        """Report what was asked and what came back; the quality verdict is the judge's.

        No composite. Two retrieval proxies were published from here and both were constants
        rather than measurements: gold-id recall compared an engine's own ids against ours, and
        the answer-span recall that replaced it is the exact matching LongMemEval's paper
        rejects -- and scored every abstention item 0 for every engine, an abstention having no
        answer text to find.

        A query that retrieved nothing is counted rather than dropped: a shrinking denominator
        nobody reports is how the previous broken metric survived unnoticed.
        """
        answered = {r.query_id for r in responses if r.documents}
        per_type: dict[str, int] = defaultdict(int)
        for q in unit.queries:
            per_type[q["category"]] += 1
        from memrank.metrics.retrieval import retrieval_metrics
        return {
            "composite": None,
            "per_category_counts": dict(per_type),
            "n_queries": len(unit.queries),
            "n_retrieved_nothing": sum(1 for q in unit.queries if q["id"] not in answered),
            "metric": "judged_answer_correctness (judge required)",
            # The benchmark's two OFFICIAL retrieval metrics, reported BESIDE the judged number
            # and never folded into it -- `composite` stays None and `composite_rankable` stays
            # False. They are the only LongMemEval numbers that measure the memory system rather
            # than the reader or the judge, which is exactly why they must not become a quality
            # headline: publishing a retrieval proxy as an accuracy is the category error the
            # field survey documents in MemPalace ("100%" = recall_any@5) and Supermemory
            # ("Recall@15 = 95%" printed beside QA-accuracy numbers).
            "retrieval": retrieval_metrics(unit, responses),
        }

    # ------------------------------------------------------------------ #
    # Reporting
    # ------------------------------------------------------------------ #

    def rollup(self, per_unit_scores: list[dict[str, Any]], *,
               ranked: bool = True) -> dict[str, Any]:
        """The official retrieval metrics as ONE number each, micro-averaged over 419 queries.

        Not 500: the 30 abstention items and the 51 assistant-side questions are excluded from
        retrieval only, and each unit reports its own `n_retrieval_scoreable` (0 or 1, since a
        LongMemEval unit holds exactly one query) so they weigh nothing instead of scoring zero.
        `n_retrieval_scoreable` rides along so the denominator is on the artifact rather than
        inferred -- the numerator/denominator mix-up is the failure behind the field's one
        retracted number.
        """
        from memrank.metrics.retrieval import DEFAULT_KS, weighted_mean

        if not ranked:
            # `fixed-context` and `full-context` hand back all 40-53 sessions in ingest order. Both metrics
            # are cut at k, so they are undefined here -- and scoring anyway produced
            # `recall_all@5` = 0.0, which reads as total retrieval failure for the two arms that
            # retrieved everything. The official harness excludes its long-context modes from
            # retrieval scoring by construction.
            unscored: dict[str, Any] = {"n_retrieval_scoreable": 0,
                                        "retrieval_metrics_apply": False}
            for k in DEFAULT_KS:
                unscored[f"recall_all@{k}"] = None
                unscored[f"ndcg_any@{k}"] = None
            return unscored

        cells = [u.get("retrieval") or {} for u in per_unit_scores]
        out: dict[str, Any] = {
            "n_retrieval_scoreable": sum(int(c.get("n_retrieval_scoreable") or 0) for c in cells),
            "retrieval_metrics_apply": True,
        }
        for k in DEFAULT_KS:
            for name in (f"recall_all@{k}", f"ndcg_any@{k}"):
                out[name] = weighted_mean(
                    [(c.get(name), int(c.get("n_retrieval_scoreable") or 0)) for c in cells])
        return out

    def report_template(self) -> str:
        return (
            "# LongMemEval-S report -- {adapter}\n"
            "\n"
            "- Composite: **{composite}**\n"
            "- Dataset: {dataset_version}\n"
            "- k: {k}\n"
            "\n"
            # Questions per type, NOT recall: `score()` emits `per_category_counts` and this
            # benchmark deliberately publishes no retrieval proxy as a quality number. The old
            # "Per-question-type recall@k" heading survived the withdrawal of the substring
            # proxy and named a metric nothing computed -- the same category error the field
            # survey documents in MemPalace ("100%" = recall_any@5) and Supermemory
            # ("Recall@15 = 95%" printed beside QA-accuracy numbers).
            "## Questions per type\n"
            "{per_category_md}\n"
        )
