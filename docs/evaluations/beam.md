# BEAM -- `beam`

<!-- runnable: no -- a one-time dataset download the first time it is constructed -->
```python
from memrank.evaluations import BEAM

evaluation = BEAM()
```

Long conversations with questions testing ten memory abilities, at supported sizes of 100K,
500K and 1M tokens.

## How it works

Each conversation is a single user's narrative rather than stitched-together chats, and comes in
size tiers -- memrank supports 100K, 500K and 1M tokens. Memrank gives one conversation to the
[system](../reference/system.md), then asks that conversation's probing questions; each
conversation is its own group and is cleared afterwards.

The ten abilities include three that the other evaluations here do not test at all: noticing that
a later statement contradicts an earlier one, putting events in the order they happened, and
following an instruction given long ago.

Memrank supplies speaker labels and conversation text, without rendered dates or machine
timestamps, following the dataset protocol.

## Why it matters

Use BEAM to assess memory over long conversations. When a conversation exceeds the selected
reader's context window, retrieval must select a subset for answer generation.

**Quality here requires a judge.** The authors grade an answer against a rubric of small atomic
facts, one judgement each, averaged within the question -- and for the ordering ability, by how
well the answer's order matches the rubric's, because averaging would score a reversed answer as
highly as a correct one. The tracked judging pipeline follows that protocol. Without a judge, this evaluation reports
latency, failures and counts; its previous retrieval proxy was withdrawn.

Its protocol gives the reader all retrieved context. Tracked runs therefore lift the shared
retrieval token budget for every system on BEAM.


The answer is generated with BEAM's own reader prompt (`answer_generation_for_rag`), verbatim:
answer only from the context, directly and concisely. It has no abstention instruction and no
date line; see [the reader prompt](../methodology.md#the-reader-prompt).

## What it needs

A one-time download of the dataset, cached after the first run, or a local copy pointed at with
`BEAM_DATA_PATH`. The download goes through Hugging Face's `datasets` package, which is the
`benchmarks` extra rather than part of the base install: `pip install 'memrank[benchmarks]'`, or
`uv add 'memrank[benchmarks]'`. Without it, an uncached tier is refused with that instruction
before anything is fetched. The largest tier lives in a separate dataset and is not covered here.

Quality also needs a judge, which means a model key and sending question text to that model.

The data is share-alike licensed, which is stricter than its code license and carries over to
anything derived from it.

## References

- [mohammadtavakoli78/BEAM](https://github.com/mohammadtavakoli78/BEAM) -- the authors' code, MIT.
- [arXiv:2510.27246](https://arxiv.org/abs/2510.27246) -- *Beyond a Million Tokens: Benchmarking
  and Enhancing Long-Term Memory in LLMs*, ICLR 2026.
- [Methodology](../methodology.md) -- the token budget, and the evaluations exempt from it.
