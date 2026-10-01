"""Conversational benchmarks must hand engines structured turns, not a JSON dump.

Before this, no benchmark populated ``Document.messages`` while two adapters branched on
it, so every fact-extracting engine received ``json.dumps(turns)`` as a single message with
role "user" -- an entire session attributed to one speaker, with dia_id and blip_caption
inline. These tests pin the structured path so the fallback cannot silently return.

No dataset download: turns are constructed inline and passed through the same helpers the
loaders use.
"""

from __future__ import annotations

from memrank.adapters.atomicmemory import AtomicMemory
from memrank.adapters.mem0 import Mem0
from memrank.benchmarks.locomo import LoCoMoBenchmark
from memrank.core import Document

_TURNS = [
    {"speaker": "Caroline", "dia_id": "D1:1", "text": "I went to a support group yesterday."},
    {"speaker": "Melanie", "dia_id": "D1:2", "text": "That's great, what happened?"},
]


def _locomo_doc() -> Document:
    return Document(id="s1_session_1", content="ignored-by-these-tests",
                    messages=LoCoMoBenchmark._turns_to_messages(_TURNS, "Caroline"))


def test_locomo_turns_become_role_tagged_messages():
    """Two named humans map to wire-valid roles: speaker A is user, B is assistant."""
    messages = LoCoMoBenchmark._turns_to_messages(_TURNS, "Caroline")
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert [m["speaker"] for m in messages] == ["Caroline", "Melanie"]


def test_speaker_name_survives_in_content():
    """LoCoMo questions name their speakers, and adapters forward only role+content --
    so a name held anywhere else is dropped before it reaches the engine."""
    messages = LoCoMoBenchmark._turns_to_messages(_TURNS, "Caroline")
    assert messages[0]["content"].startswith("Caroline: ")
    assert "support group" in messages[0]["content"]


def test_dataset_bookkeeping_does_not_reach_the_engine():
    """dia_id is join metadata for scoring, not something an extractor should read."""
    rendered = " ".join(m["content"] for m in LoCoMoBenchmark._turns_to_messages(_TURNS, "Caroline"))
    assert "dia_id" not in rendered and "D1:1" not in rendered


def test_image_captions_are_content_and_survive():
    """28% of LoCoMo turns carry a blip_caption and some gold answers appear ONLY there;
    dropping them cost ~3 points of recall when this helper first shipped without them.
    The image-search `query` field is construction metadata and stays out."""
    turns = [{"speaker": "Caroline", "dia_id": "D1:5", "text": "Look at this!",
              "img_url": "https://example/x.jpg", "query": "pride flag mural",
              "blip_caption": "a wall with a painting of a woman"}]
    content = LoCoMoBenchmark._turns_to_messages(turns, "Caroline")[0]["content"]
    assert "a wall with a painting of a woman" in content
    assert "Look at this!" in content
    assert "pride flag mural" not in content and "example/x.jpg" not in content


def test_a_caption_only_turn_still_carries_its_image():
    """Some turns are an image with no text; they must not become an empty message."""
    turns = [{"speaker": "Melanie", "text": "", "blip_caption": "a sunrise over water"}]
    content = LoCoMoBenchmark._turns_to_messages(turns, "Caroline")[0]["content"]
    assert "a sunrise over water" in content


def test_mem0_receives_structured_turns_not_the_fallback():
    """The regression this file exists for: one JSON blob with role 'user'."""
    forwarded = Mem0._document_to_messages(_locomo_doc())
    assert len(forwarded) == 2                      # not one blob
    assert forwarded[0]["role"] == "user" and forwarded[1]["role"] == "assistant"
    assert forwarded[0]["content"] == "Caroline: I went to a support group yesterday."


def test_atomicmemory_does_not_double_prefix_an_attributed_turn():
    """Content already says who spoke, so adding the role would give 'User: Caroline: ...'."""
    text = AtomicMemory._document_to_conversation_text(_locomo_doc())
    assert text.splitlines()[0] == "Caroline: I went to a support group yesterday."
    assert "User: Caroline" not in text


def test_a_transcript_without_speakers_keeps_its_role_prefix():
    """LongMemEval turns are user/assistant with no names; the role is their only
    attribution, so it must stay."""
    doc = Document(id="q1_s1", content="unused",
                   messages=[{"role": "user", "content": "Where did I park?"},
                             {"role": "assistant", "content": "Level 3."}])
    text = AtomicMemory._document_to_conversation_text(doc)
    assert text.splitlines() == ["User: Where did I park?", "Assistant: Level 3."]
