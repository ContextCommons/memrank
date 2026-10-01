# LongMemEval -- `longmemeval`

<!-- runnable: no -- a one-time dataset download the first time it is constructed -->
```python
from memrank.evaluations import LongMemEval

evaluation = LongMemEval()
```

Five hundred questions, each paired with its own collection of dated chat sessions.

## How it works

Each question has a separate collection of user-and-assistant sessions, called a *haystack*.
Only some sessions contain relevant evidence. Memrank gives one question's haystack to the
[system](../reference/system.md) in the order the data lists it, one session at a time, and asks
the question after ingesting all sessions, following the authors' task definition. Each
question is its own group, so its haystack is cleared before the next.

The dates travel inside the text of each session, because memrank cannot reorder what a system
returns: a system hands back its own documents, not memrank's, so ordering by timestamp would
only be possible for the controls and would make context order depend on which system was under
test. This differs from the authors' context-ordering procedure.

## Why it matters

The same 500 questions are available at several context sizes, so
what changes between runs is how much irrelevant material the system had to hold -- not the
questions. The authors' headline is that assistants and long-context models lose about a third of
their accuracy when the questions move from a clean context into a large haystack.

**Quality here requires a judge.** The paper rejects string matching outright -- correct answers
take flexible forms -- and grades semantic and temporal consistency with a model instead. A span
proxy was tried here and withdrawn: it scored every question whose correct response is an
abstention at zero by construction, since an abstention has no answer text to find. Without a
judge this evaluation reports latency, failures and counts.

It is also one of two evaluations whose own protocol caps context at the model's window rather
than at a fairness budget, so memrank's token budget is lifted for it, symmetrically across every
system. This applies equally to all systems, but removes budget normalization against other evaluations.


The answer is generated with LongMemEval's own direct reader prompt, verbatim, including its
`Current Date` line. See [the reader prompt](../methodology.md#the-reader-prompt).

## What it needs

A one-time download of the dataset, cached after the first run, or a local copy pointed at with
`LONGMEMEVAL_DATA_PATH`.

Quality also needs a judge, which means a model key and sending question text to that model.

The questions and code are MIT licensed; the haystack filler is drawn from other corpora whose
licenses are not. Check the source-corpus licenses before redistributing the data.

## References

- [xiaowu0162/LongMemEval](https://github.com/xiaowu0162/LongMemEval) -- the authors' code, MIT.
- [arXiv:2410.10813](https://arxiv.org/abs/2410.10813) -- *Benchmarking Chat Assistants on
  Long-Term Interactive Memory*, ICLR 2025.
- [Methodology](../methodology.md) -- the token budget, and the two evaluations exempt from it.
