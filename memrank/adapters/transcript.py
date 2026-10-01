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
"""One rendering of a Document into the text an engine ingests.

Every text-ingesting adapter used to render documents its own way, and none of them
carried WHEN the conversation happened. Engines therefore stamped memories with
ingestion time: a 2023 LoCoMo conversation was stored as "August 2026", and 321 of
1540 temporal queries were unanswerable by construction.

Only hindsight has a native ingest-timestamp field. The AtomicMemory wire contract has
none (am-wire IngestRequest is user_id/conversation/source_site/source_url/session_id),
and supermemory's documented path does not either -- so **the text is the only channel
every engine has**. That makes this the chokepoint per CLAUDE.md: one renderer, used by
every adapter, rather than the date being remembered in three places and forgotten in a
fourth. Adapters with a native field still send it; the two carry different weight, since
the native field drives temporal filtering while the text drives extraction.

This is also what the vendors ask for. Hindsight's ingest guidance is that a conversation
should convey "who said what, and when"; supermemory's own benchmark harness prepends the
session date to the content it stores.
"""

from __future__ import annotations

from typing import Any

from memrank.core import Document


def _speaker_line(message: dict[str, Any]) -> str:
    """One turn, attributed however the source allows.

    A message carrying ``speaker`` is already attributed in its content (LoCoMo renders
    "Caroline: ..."), so adding the role would produce "User: Caroline: ...". A transcript
    without speakers (LongMemEval) has only its role, which is then the attribution.
    """
    content = str(message.get("content", ""))
    if message.get("speaker"):
        return content
    return f"{str(message.get('role', 'user')).capitalize()}: {content}"


def header(doc: Document) -> str:
    """The '[who, and when]' line for a CONVERSATION, or '' for anything else.

    The date is whatever the benchmark put in ``context`` -- deliberately the SOURCE's own
    wording rather than a reformatting of ``timestamp``. LoCoMo records "1:56 pm on 8 May,
    2023"; passing that through is honest, whereas choosing a format because it resembles
    a gold answer ("7 May 2023") would be tuning to the scorer. The ISO ``timestamp`` stays
    the machine field, sent natively by adapters whose engine accepts one.

    Gated on ``messages`` because ``context`` means different things for different
    benchmarks: for a conversation it is "who, and when", but for a relation-graph document
    it is a description that adapters already pass separately (supermemory sends it as
    ``entity_context`` metadata). Prepending it there would change what a non-conversational
    benchmark stores, for a fix that has nothing to do with it.
    """
    return f"[{doc.context}]" if doc.context and doc.messages else ""


def render(doc: Document) -> str:
    """The text to hand an engine for ``doc``.

    Documents with no ``messages`` -- synthetic benchmarks, relation-graph seeds, anything
    non-conversational -- render as their content **byte-identically**, so this is safe to
    put on every adapter's ingest path.
    """
    if not doc.messages:
        return doc.content
    body = "\n".join(_speaker_line(m) for m in doc.messages)
    prefix = header(doc)
    return f"{prefix}\n{body}" if prefix else body
