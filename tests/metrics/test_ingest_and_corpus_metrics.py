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
"""What a run pushed through, and what ingesting it cost.

Retrieval had a full summary -- count, mean, IQR, five percentiles. Ingest had three percentiles
and nothing else, which cannot say what ingest COST: p50=12,750 ms is the same number whether
seventy documents paid it or one did. And ingest is where engines differ most -- median ingest p50
across the runs on the machine where this was written spans 1.1 ms (myengine, regex extraction) to
12,750 ms (atomicmemory, an LLM call per document). Four orders of magnitude, invisible beside a
quality score.

Corpus size was not recorded at all: "70 documents, 319,797 bytes" had to be recomputed from the
artifact every time it was wanted.
"""
from __future__ import annotations

import pytest

from memrank import runner
from memrank.metrics import cost


def test_throughput_says_what_percentiles_cannot(monkeypatch):
    """Documents per second and the total seconds they took. A p50 alone never says how many
    documents paid it."""
    summary = {"count": 70, "mean_ms": 200.0}

    throughput = runner._ingest_throughput(summary)

    assert throughput["documents"] == 70
    assert throughput["total_seconds"] == pytest.approx(14.0)
    assert throughput["documents_per_second"] == pytest.approx(5.0)


def test_a_run_that_ingested_nothing_reports_no_throughput():
    """The read-only controls (`none`, `icl`) ingest nothing. 0 docs/sec would read as an engine
    that is infinitely slow rather than one that was never asked -- and it divides by zero."""
    assert runner._ingest_throughput({"count": 0, "mean_ms": 0.0}) is None
    assert runner._ingest_throughput(None) is None


def test_corpus_size_counts_documents_bytes_and_tokens():
    """Bytes are UTF-8, which is what crossed the wire; tokens use the same pinned encoding as
    context_tokens_mean, so the corpus and what was retrieved from it divide into each other."""
    ingested = [{"content": "héllo world"}, {"content": "second document"}]

    size = runner._corpus_size(ingested)

    assert size["corpus_documents"] == 2
    assert size["corpus_bytes"] == len("héllo world".encode()) + len(b"second document")
    assert size["corpus_tokens"] == (cost.count_tokens("héllo world")
                                     + cost.count_tokens("second document"))


def test_a_document_with_no_content_is_counted_but_contributes_nothing():
    """An empty document is still a document the engine was asked to store."""
    size = runner._corpus_size([{"content": None}, {"content": ""}])

    assert size["corpus_documents"] == 2
    assert size["corpus_bytes"] == 0 and size["corpus_tokens"] == 0


def test_bytes_are_utf8_not_characters():
    """The distinction only shows on non-ASCII, which is most of LoCoMo's non-English content."""
    size = runner._corpus_size([{"content": "日本語"}])

    assert size["corpus_bytes"] == 9, "three characters, nine bytes"
