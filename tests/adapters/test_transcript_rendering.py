"""One rendering of a Document, carrying WHEN the conversation happened.

Every text-ingesting adapter used to render documents its own way and none carried the
session date, so engines stamped 2023 conversations with ingestion time and 321 of 1540
LoCoMo temporal queries were unanswerable by construction (audit F15). Only hindsight has
a native ingest-timestamp field; the AtomicMemory wire contract has none, so the text is
the only channel every engine shares.
"""

from __future__ import annotations

from memrank.adapters import transcript
from memrank.adapters.atomicmemory import AtomicMemory
from memrank.adapters.mem0 import Mem0
from memrank.core import Document

_DATED = Document(
    id="s1", content="raw-json-fallback",
    context="Conversation between Caroline and Melanie (session_1), 1:56 pm on 8 May, 2023",
    timestamp="2023-05-08T13:56:00+00:00",
    messages=[{"role": "user", "content": "Caroline: I went to a support group.", "speaker": "Caroline"},
              {"role": "assistant", "content": "Melanie: How was it?", "speaker": "Melanie"}],
)


def test_the_session_date_leads_the_transcript():
    """The whole point: an engine that is not told when cannot answer when."""
    first = transcript.render(_DATED).splitlines()[0]
    assert first.startswith("[") and "8 May, 2023" in first


def test_turns_follow_the_header_attributed():
    lines = transcript.render(_DATED).splitlines()
    assert lines[1] == "Caroline: I went to a support group."
    assert lines[2] == "Melanie: How was it?"


def test_a_document_without_messages_renders_its_content_unchanged():
    """Synthetic benchmarks and relation-graph seeds must pass through untouched."""
    doc = Document(id="d", content="a plain fact")
    assert transcript.render(doc) == "a plain fact"


def test_context_on_a_non_conversational_document_is_not_prepended():
    """`context` means "who and when" for a conversation but "what this document is" for a
    relation-graph document -- which adapters already send separately (supermemory as
    entity_context metadata). Prepending it there would change what an unrelated benchmark
    stores, so the header is gated on there being turns."""
    doc = Document(id="d", content="Avery prefers green dashboards.",
                   context="Dashboard preference memo.")
    assert transcript.render(doc) == "Avery prefers green dashboards."


def test_a_document_without_context_gets_no_empty_header():
    doc = Document(id="d", content="x", messages=[{"role": "user", "content": "hello"}])
    assert transcript.render(doc) == "User: hello"


def test_roles_attribute_a_transcript_that_has_no_speakers():
    """LongMemEval is user/assistant with no names; the role is its only attribution."""
    doc = Document(id="q1", content="x",
                   messages=[{"role": "user", "content": "Where did I park?"},
                             {"role": "assistant", "content": "Level 3."}])
    assert transcript.render(doc).splitlines() == ["User: Where did I park?", "Assistant: Level 3."]


def test_atomicmemory_ingests_the_dated_transcript():
    """Any wire-compatible engine subclassing this adapter inherits the path, so the one
    assertion covers them too -- including subclasses outside this repository."""
    text = AtomicMemory._document_to_conversation_text(_DATED)
    assert "8 May, 2023" in text
    assert text == transcript.render(_DATED)


def test_mem0_gets_the_date_on_its_first_turn_without_a_fabricated_speaker():
    """mem0 takes messages, not text, so the header rides on turn one -- the turn count
    stays honest and no invented speaker appears in the transcript."""
    messages = Mem0._document_to_messages(_DATED)
    assert len(messages) == 2                       # not 3: no synthetic header turn
    assert "8 May, 2023" in messages[0]["content"]
    assert messages[0]["content"].endswith("Caroline: I went to a support group.")
    assert messages[0]["role"] == "user" and messages[1]["role"] == "assistant"
