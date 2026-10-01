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
"""Task -- one thing to put to the system.

Plain data, and nothing about correctness: the same task can be measured many ways, which is
what moving scoring out of the run loop bought (decision 0009).

The fields are the ones the five registered loaders actually fill. Their query dicts carry
``id``, ``text``, ``user_id``, ``category``, ``kind``, ``gold_answers``, ``required_spans``,
``forbidden_spans``, ``evidence_doc_ids``, ``gold_ids``, ``rubric``, ``judge_prompt_key``,
``retrieval_scoreable``, ``ordering_tested`` and ``query_timestamp``. What a correct outcome
looks like is gathered into :class:`Expected`; the rest that a loader genuinely uses stays in
``metadata``, which is deliberately the only untyped corner and the only place a benchmark's
own protocol knobs live.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from memrank.core import Document


class Expected(BaseModel):
    """What a correct outcome looks like, as the loaders carry it.

    Nothing here decides anything: a measure reads it and decides. ``polarity`` is the loaders'
    ``kind`` -- a negative task is correct when the wrong memory is NOT surfaced.
    """

    model_config = ConfigDict(frozen=True)

    #: The reference answer(s). `gold_answers` in every loader.
    answers: tuple[str, ...] = ()
    #: Spans that must appear verbatim in one retrieved document for a hit.
    required_spans: tuple[str, ...] = ()
    #: Spans whose presence is the failure (and, for a negative task, the whole test).
    forbidden_spans: tuple[str, ...] = ()
    #: Source document ids the evidence gate is taken over (`metadata['doc_id']`, never the
    #: engine's opaque id).
    evidence_doc_ids: tuple[str, ...] = ()
    #: BEAM's atomic nuggets: the scoring key for a rubric-shaped question.
    rubric: tuple[str, ...] = ()
    polarity: Literal["positive", "negative"] = "positive"


class Task(BaseModel):
    """One thing to put to the system: what to give it first, what to ask, what is expected."""

    model_config = ConfigDict(frozen=True)

    id: str
    #: The question, as the system is asked it.
    prompt: str
    expected: Expected = Expected()
    #: Tasks that share state. A run clears between groups, never inside one.
    group: str | None = None
    #: Documents given before the prompt. Tasks of one group normally carry the same tuple;
    #: the run ingests a group's documents once, in first-seen order, deduplicated by id.
    context: tuple[Document, ...] = ()
    category: str | None = None
    #: Only what a loader actually uses: `judge_prompt_key`, `retrieval_scoreable`,
    #: `ordering_tested`, `query_timestamp`, `gold_ids`.
    metadata: dict[str, Any] = Field(default_factory=dict)
