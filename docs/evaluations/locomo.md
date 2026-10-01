# LoCoMo -- `locomo`

<!-- runnable: no -- a one-time dataset download the first time it is constructed -->
```python
from memrank.evaluations import LoCoMo

evaluation = LoCoMo()
```

Ten dated conversations with about 1,500 scored questions about their content.

## How it works

Each conversation runs over weeks of dated sessions between two speakers -- hundreds of turns,
tens of thousands of words. Memrank gives one conversation at a time to the
[system](../reference/system.md), session by session, then asks that conversation's questions.
Each conversation is its own group, so the memory is cleared between them and one conversation
cannot answer another's questions.

The questions come in four kinds: one fact from one session, a fact assembled from several, a
question about *when* something happened, and one that needs outside knowledge as well. A fifth
kind in the data -- deliberately unanswerable questions -- is excluded, because the release ships
almost none of them with an answer to grade against.

## Why it matters

LoCoMo supports comparisons on multi-session conversation recall. Its conversations may fit
within the reader's context window, so include a [full-context](../systems/full-context.md)
comparison when assessing the benefit of retrieval. A high score alone does not establish that
retrieval was necessary.

**Quality here requires a judge.** The published protocol grades the generated answer against a
reference. The reference answers can be derived -- *"three times"*,
*"2022"* -- rather than quotes, so literal retrieval matching can miss correct evidence. Without a judge this evaluation reports latency, failures and question
counts, and no quality number at all.


The answer is generated with LoCoMo's own reader prompt (`QA_PROMPT`, after the context), verbatim:
a short phrase, in the context's own words where possible. See
[the reader prompt](../methodology.md#the-reader-prompt).

## What it needs

A one-time download of the dataset, cached after the first run, or a local copy pointed at with
`LOCOMO_DATA_PATH`. The cached file is checked against a pinned checksum on every load, so a
changed copy raises a checksum error before scoring.

Quality also needs a judge, which means a model key and sending question text to that model.

The data is licensed for non-commercial use. Check its terms before using or redistributing it.

## References

- [snap-research/locomo](https://github.com/snap-research/locomo) -- the data and the authors'
  own harness.
- [arXiv:2402.17753](https://arxiv.org/abs/2402.17753) -- *Evaluating Very Long-Term
  Conversational Memory of LLM Agents*, ACL 2024.
- [Methodology](../methodology.md) -- why quality needs a judge, and limits on interpreting
  conversation-recall scores.
