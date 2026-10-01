"""mem0 can keep a separate memory per speaker, the way its own published eval does.

memrank's matched mode gives every engine one partition per benchmark unit -- that shared scope is
what makes a cross-engine row mean anything. mem0's published LoCoMo runs did something else: a
memory per SPEAKER, searched separately, answering from roughly twice the memories one partition
returns (audit F13). Until now no target could express that, so the vendor target reproduced mem0's
models and not its scoping, and said so only in a comment.

These tests pin both paths against a fake transport: no mem0 server, no OpenAI key, no network.
The unpartitioned assertions are the regression gate -- matched mode must be untouched, because it
is the one that reaches the leaderboard. That gate matters MORE since 2026-08-18: the partitioned
path is what a bare `mem0` now does, and the unpartitioned one is reached only by `mem0:voyage` and
`mem0:bge-tei` clearing the inherited block.
"""

from __future__ import annotations

from memrank.adapters.mem0 import Mem0
from memrank.benchmarks.locomo import LoCoMoBenchmark
from memrank.core import Document

_TURNS = [
    {"speaker": "Caroline", "text": "I went to a support group yesterday."},
    {"speaker": "Melanie", "text": "That's great, what happened?"},
    {"speaker": "Caroline", "text": "It was powerful."},
]


class _FakeMemory:
    """Stands in for the mem0 SDK handle, recording what it was asked to do."""

    def __init__(self) -> None:
        self.added: list[tuple[str, list[dict]]] = []
        self.searched: list[tuple[str, str, int]] = []

    def add(self, messages, user_id, metadata=None):
        self.added.append((user_id, messages))

    def search(self, query, top_k=10, filters=None, threshold=0.0):
        self.searched.append(((filters or {}).get("user_id"), query, top_k))
        return {"results": [{"id": f"m-{len(self.searched)}", "memory": "a fact"}]}


def _adapter(partitioning=None) -> tuple[Mem0, _FakeMemory]:
    adapter = Mem0(mode="sdk", partitioning=partitioning)
    memory = _FakeMemory()
    adapter._memory = memory
    adapter.prepare("run-1")
    return adapter, memory


def _doc() -> Document:
    return Document(id="s1_session_1", content="unused", user_id="run-1",
                    context="Conversation between Caroline and Melanie (session_1), 8 May, 2023",
                    messages=LoCoMoBenchmark._turns_to_messages(_TURNS, "Caroline"))


def test_matched_mode_ingests_one_partition():
    """The regression gate: the path that reaches the leaderboard is unchanged."""
    adapter, memory = _adapter()
    adapter.ingest([_doc()])

    assert [uid for uid, _ in memory.added] == ["run-1"]
    assert len(memory.added[0][1]) == 3          # all three turns, one call


def test_per_speaker_ingest_splits_by_speaker():
    adapter, memory = _adapter({"by": "speaker"})
    adapter.ingest([_doc()])

    assert sorted(uid for uid, _ in memory.added) == ["run-1-Caroline", "run-1-Melanie"]
    by_uid = dict(memory.added)
    assert len(by_uid["run-1-Caroline"]) == 2    # her two turns, not the whole session
    assert len(by_uid["run-1-Melanie"]) == 1


def test_every_partition_carries_the_session_date():
    """A partition without the header dates its memories to ingestion time -- F15, per speaker."""
    adapter, memory = _adapter({"by": "speaker"})
    adapter.ingest([_doc()])

    for _, messages in memory.added:
        assert "8 May, 2023" in messages[0]["content"]


def test_per_speaker_retrieve_searches_each_and_does_not_trim():
    """k per partition, concatenated. Cutting to k would be the hindsight [:k] defect again."""
    adapter, memory = _adapter({"by": "speaker"})
    adapter.ingest([_doc()])
    recall = adapter.retrieve("what happened?", 10, "run-1")
    docs, raw = recall.documents, recall.declared

    assert sorted(uid for uid, _, _ in memory.searched) == ["run-1-Caroline", "run-1-Melanie"]
    assert [top_k for _, _, top_k in memory.searched] == [10, 10]
    assert len(docs) == 2                         # one per partition, nothing dropped
    assert raw["partitions"] == ["run-1-Caroline", "run-1-Melanie"]


def test_matched_mode_retrieves_from_one_scope():
    adapter, memory = _adapter()
    adapter.ingest([_doc()])
    adapter.retrieve("what happened?", 10, "run-1")

    assert [uid for uid, _, _ in memory.searched] == ["run-1"]


def test_a_document_without_speakers_is_not_given_invented_ones():
    """LongMemEval renders prose, not a dialogue. Scoping by a fabricated key would be worse
    than not partitioning at all."""
    adapter, memory = _adapter({"by": "speaker"})
    adapter.ingest([Document(id="d1", content="Some prose.", user_id="run-1",
                             messages=[{"role": "user", "content": "Some prose."}])])

    assert [uid for uid, _ in memory.added] == ["run-1"]


def test_partitions_do_not_leak_between_units():
    """prepare() starts a unit; a partition left over from the previous one would make the next
    unit search another conversation's memories."""
    adapter, memory = _adapter({"by": "speaker"})
    adapter.ingest([_doc()])
    adapter.prepare("run-2")

    assert adapter._partitions == []


def test_a_single_speaker_conversation_is_searched_where_it_was_written():
    """Ingest writes `run-1-Caroline`; retrieving from `run-1` would query a scope nothing was
    written to and score a clean zero -- a wrong number, not an error."""
    adapter, memory = _adapter({"by": "speaker"})
    solo = [{"speaker": "Caroline", "text": "I went alone."}]
    adapter.ingest([Document(id="d1", content="unused", user_id="run-1",
                             messages=LoCoMoBenchmark._turns_to_messages(solo, "Caroline"))])
    adapter.retrieve("what happened?", 10, "run-1")

    assert [uid for uid, _ in memory.added] == ["run-1-Caroline"]
    assert [uid for uid, _, _ in memory.searched] == ["run-1-Caroline"]
