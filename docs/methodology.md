# Methodology

Memrank reports four axes for every system it measures -- quality, latency, tokens and cost.
Each is produced by a named [measure](measures.md), and a measure declares *before* it runs what
it reads from a [trace](reference/trace.md) and who decided the number it produces.

| Axis | What produces it | Scope | Reads | Decider |
|---|---|---|---|---|
| Quality | `<evaluation>-score` | run, one value per group | `recalled`, `declared` | rule -- the evaluation's own `score()` |
| Quality | `word-match` | task | `recalled` | rule -- span recall |
| Quality | `judge` | task | `answered` | a model |
| Latency | `latency` | run | `timings_ms` | memrank's own clock |
| Completeness | `failure-rate` | run | `error` | memrank's own bookkeeping |
| Tokens | a declaration, not a measure | -- | `declared` | the system, recorded as a declaration |
| Cost | an estimate, not a measure | -- | context tokens and a pinned price table | memrank, from a versioned table |

The last two rows say what they are rather than implying a measure that does not exist. Token
usage is what a system says a provider billed it; cost is derived arithmetic over a price table.

[`docs/measures.md`](measures.md) is the contract behind those columns. This page explains how to interpret the values and their limitations.

## Quality

Three shipped measures produce a quality value, and they are not interchangeable.

**`<evaluation>-score`** is one value per group and nothing that combines them, so no run-level
value is invented on an evaluation's behalf.

**`word-match`** is **retrieval correctness and never answer correctness**, and says so in the
`why` of every value it produces, so the caveat travels with the value.

**`judge`** scores generated-answer correctness using a model. The Python API requires callers
to add it explicitly. Applying it without the required API key raises an error. `memrank run`
judges with Claude Haiku (`claude-haiku-4-5`) unless `--judge-model` names another Anthropic
model, and every result it produces names the model that judged it. Scores judged by different
models are not comparable.

**Metric applicability.** `substring_recall_supported` says whether the answer-substring proxy
applies. `composite_rankable` independently says whether the evaluation's own composite may be
shown and ranked without a judge. Graph and passage retrieval scores can have the first flag
False and the second True. A passage score measures **full-passage retrieval recall, not
answer-span or end-to-end answer correctness**.

When `composite_rankable` is False, rankable surfaces withhold the raw composite and its
per-ability/per-category tables; a valid judged run supplies the quality number. Detailed JSON
may carry a raw composite for inspection, paired with its applicability declarations.
**Contract for consumers: use `composite_rankable` to decide whether a raw composite is valid
for ranking, and `quality_metric` to say what it measures.**

### SQuAD, the quick-start evaluation

`squad-score` is full-passage retrieval recall over a fixed subset of SQuAD v1.1's development
set: 32 passages supplied to the system in one pool, then 64 questions, two per passage in source
order. A question is a hit when its entire source passage comes back inside a returned document
(whitespace collapsed, case kept), from at most 10 documents per question by default. Its
`quality_metric` is `passage_recall`, and every value says what it measures.

Interpretation limits:

- **That an answer was right.** The scorer never reads an answer. This is not SQuAD's native
  exact-match or answer-token F1.
- **That returning more is better.** Recall does not penalise irrelevant passages, so two
  systems compare only at the same retrieval limit.
- **Anything against a published retrieval number.** Among 32 passages the right one is nearly
  always found; open-domain question answering retrieves over millions. The quick start's high
  score describes that pool, not the retriever. `slice="full"` (2,067 passages) is a harder pool
  and still not an open-domain corpus.

**A published row requires a complete record, not a score.** There is no standing public
leaderboard ([SPEC.md section 8](SPEC.md#8-hosting-and-licensing)); where a result is published,
a run qualifies when its provenance and quality declarations are complete, and it reports
whatever quality it has: the judged score, a self-contained composite, or `withheld`. It does not require a `composite`,
which the three external benchmarks stopped defining -- requiring one silently dropped every
judged run of them between 2026-08-13 and 2026-08-25. A run with no quality value still
publishes its latency, cost and token measurements.

**A failed unit leaves the denominator, and says so.** A cell attempts every unit and
records what happened to each. A unit whose ingest, retrieval or scoring raises is recorded as
failed and the run continues; it contributes no per-unit score and no per-query rows, so it is
excluded from the composite's mean and from judging rather than counted as a zero. **A composite
is therefore a mean over the units that RAN, not over the units the evaluation defines.** What
says which is in the same artifact, beside the composite:

- `unit_outcomes` -- one record per unit attempted: `unit_id` and `outcome` (`ok` or `failed`),
  plus the `stage` (`ingest` / `retrieve` / `score`), exception class and message for a failure.
- `units_total`, `units_failed`, `unit_failure_rate` -- the counts, also shown as `units failed`
  by `memrank runs show`, and narrated during the run as a warning per lost unit.

A cell in which every unit failed reports `composite: null` rather than `0.0`, on the same rule
as everywhere else: a score a system did not earn is not zero. The `failure-rate` measure is the
same fact stated as a value, and [`docs/measures.md`](measures.md) has the whole of the rule.
`memrank submit --fail-fast` restores the older behaviour of ending the cell at the first unit
that raises. Two conditions end a run whatever that flag says: an exhausted provider rate limit
(an account-wide condition, so remaining units would encounter the same exhausted limit) and an operator interrupt.

**One system under two configurations is two rows on one board.** Row identity is the
system, the transport, *and* the configuration it drives its engine with -- the extractor LLM, embedder,
context budget and retrieval settings it ran with. So two arms of one system that differ only
in their extractor -- a deterministic built-in versus a hosted model -- rank against each other
rather than one replacing the
other. Engine *version* is deliberately excluded: a rebuild of one configuration is a
newer run of the same system configuration and replaces its row. Both rows currently publish
under the adapter's name, and are told apart by `config_hash` in their provenance.

**Slices are for debugging, not full-dataset performance claims.** A `smoke`/`mini` slice
(`beam:100k-smoke`, `locomo:mini`) exists to debug ingest, retrieval, judging and cloud dispatch
cheaply. It selects the *first* N units of the dataset, and the first units are not a fair
sample: measured on the first full judged BEAM 100k tier (run
`20260813-184519__beam__100k__c37986`), conversation 1 scores 0.318, the first five average
0.217 and the full 20-conversation tier 0.158, so a smoke figure flatters every system by ~2x.
**A slice figure is never quoted as a score; measurement claims come from full tiers.** Two
structural guards make that more than a convention: the leaderboard's `board_key` segregates
rows by tier and slice, so slice figures cannot blend into tier comparisons; and the bias is
positional and therefore identical across systems, so same-slice *rankings* stay directionally
useful for debugging even though their absolute values are not. (Decision recorded closing M6 of
the BEAM protocol-fidelity plan.)

## Latency

`latency` is wall-clock time (`perf_counter`) taken at memrank's own call boundary around each
step, never a figure a system reported about itself. It reports p50 and p95 per step by nearest
rank, and every value carries the sample count it was taken over, because a percentile over
three samples is a different object from one over three hundred. [The command
line](misc/command-line.md)'s older run loop takes the same measurement through
`LatencyCollector.track` and reports p50/p95/p99 per `ingest()` and `retrieve()` by linear
interpolation on the sorted distribution. Latency is reported, never asserted on, and ranked
only within a transport class -- see **Transport matters** below.

Systems MAY record additional internal-only timings (e.g., AtomicMemory's extraction phase) by
calling `LatencyCollector.record` directly. These appear in the per-cell JSON but are not part of
the headline.

**Transport matters.** A system that drives an in-process SDK incurs no network or
serialization overhead; one that drives an HTTP API does. Latency is therefore only
directly comparable *within* a transport class. Every row records its
`transport` (`http` / `sdk` / `in-process` / `translator`); read latency alongside it and do
not rank an SDK engine against an HTTP engine on raw latency without saying so.

`translator` is the class for an engine reached through a vendor-written program implementing
[the system contract](system-contract.md) -- the route by which memrank evaluates an engine it has
never seen. A translator is an extra process and an extra hop, so its measured latency includes
overhead a directly-driven engine never pays. Naming it as its own class is what stops a translator
row being ranked against a direct one; quality metrics are unaffected and compare across everything.

Translators MAY report `engine_ms` per call -- their own claim about time spent inside the
engine, excluding the translator. It reaches the trace as `declared.engine_timings`, through
`Memory.declared_latency()`, as samples memrank pools and renders itself beside the wall-clock
figure as `ingest_engine_p50_ms` / `retrieve_engine_p50_ms` (and p95), so a reader can see how
much of the measurement was harness rather than engine. It is *reported*, not measured: memrank
cannot verify it, its decider is the system, and it is never the headline. The six wall-clock
keys are memrank's own measurement and a system cannot declare them at all -- a bucket that
would render one is refused.

## Tokens

Usage is the one number memrank asks a system for, and it is a DECLARATION rather than a
measurement: only the system under test can be told what a provider billed it. Latency, by contrast, memrank
times itself at its own call boundary and never asks for. A system that knows records into a
`TokenCollector` per LLM call, under the buckets `query` and `ingest`, and returns the four keys
of `token_metrics()`; they reach the trace as `declared.tokens`, whose decider is the system.

**Absent is not zero.** A system whose engine surfaces no usage data records nothing, and the
bucket reports `null`. Only some of the systems measured so far report usage at all, so
conflating the two would fabricate an efficiency win for every system that stays quiet. An
unmeasured cell renders as `n/a`; a measured zero renders as zero.

## Cost

Cost is an **estimate**, not a measurement, and it is **prompt cost only**. `memrank/metrics/cost.py`
counts the tokens a system injects into context per query under a pinned encoding (`o200k_base`)
and prices them at the named model's input rate from a versioned table. The table carries its own
`PRICING_TABLE_VERSION` and `PRICING_EFFECTIVE_DATE`, and both travel with the artifact, so a
figure can be read back at the rates that produced it. A model the table does not hold raises
rather than being priced at zero.

What it excludes matters as much as what it counts: **ingest-time extraction and answer generation
are not priced**. The figure is the cost of the context a system chose to inject, and every surface
that renders it is labelled accordingly.

## Retrieval token budget -- the fairness control

`--token-budget` (default 5000) caps **two** things: the `$/query` estimate, and the context text
actually handed to the reader on a judged run. The second is the one that matters methodologically.

Without a shared cap, "better recall" and "returned more text" are indistinguishable: a system
could win by injecting a larger prompt. Holding every target to the same budget means a row
reflects *what a system chose to retrieve*, not *how much*. That is the field problem memrank
exists to answer -- vendors self-publish at a different retrieval token budget per row, and no
disinterested party has re-run them under one fixed judge and matched budgets.

Truncation is hard: the concatenated context is cut at the budget, mid-sentence if necessary. Each
judged query records `context_tokens_sent` and `context_truncated`, so a context-starved run is
visible in the artifact rather than silent.

**One evaluation-scoped exception.** An evaluation whose own protocol hands the reader everything
retrieval returned may declare it (`Benchmark.context_policy = "uncapped"`); **BEAM and
LongMemEval both do.** BEAM, because every published BEAM harness -- including BEAM's own baselines
-- runs uncapped, and a capped run of it is not that evaluation. LongMemEval, because its harness
truncates only at the model window and never at a fairness budget: measured, the 5,000-token
default bit on 8/8 queries and the evidence reached the reader on **0 of 5** scoreable questions
while retrieval itself scored recall_all@10 = 1.0, so a judged run under that cap would have
measured the truncation rather than the memory. The promotion applies to every target
symmetrically, so rows within the evaluation stay comparable; what is given up is
budget-normalization against other evaluations' rows
(a standing decision recorded with the evaluation).

## The reader prompt

On a judged evaluation, a model -- the reader -- turns the question and the context a system
supplied into the answer the judge grades. The reader's wording is part of the benchmark's
protocol, so **each evaluation answers with its benchmark's own published reader prompt**,
verbatim, sent as its authors' code sends it: one user message, no system prompt.

| Evaluation | Reader prompt | Source |
|---|---|---|
| `beam` | `answer_generation_for_rag` | [BEAM `src/prompts.py` @ b2da22e](https://github.com/mohammadtavakoli78/BEAM/blob/b2da22eac88bb0874c64665f13457eb99835774a/src/prompts.py), filled as in `src/answer_probing_questions/long_term_memory_methods.py` |
| `locomo` | `QA_PROMPT`, after the context | [LoCoMo `task_eval/gpt_utils.py` @ 3eb6f2c](https://github.com/snap-research/locomo/blob/3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376/task_eval/gpt_utils.py) |
| `longmemeval` | the direct (no chain-of-thought) reader, with its `Current Date` line | [LongMemEval `src/generation/run_generation.py` @ 9e0b455](https://github.com/xiaowu0162/LongMemEval/blob/9e0b455f4ef0e2ab8f2e582289761153549043fc/src/generation/run_generation.py) |
| `demo`, `relation_graph`, evaluation files | memrank's own reader | no official reader exists |

LongMemEval's reader is the variant its harness uses both for retrieved chunks and for the full
history, so it describes any system's context truthfully; its fact-only variant presumes extracted
facts, which only some systems return. LoCoMo's adversarial category has a reader of its own, a
multiple choice built from the reference answer; memrank excludes that category because the
release ships almost none of those questions with an answer.

Two things memrank's own reader adds are deliberately **absent** from the official prompts:

- **No current-date line where the official prompt has none.** BEAM's and LoCoMo's readers state
  no date, so neither does memrank's rendering of them; LongMemEval's does, and keeps it. (LoCoMo's
  temporal questions still carry the official "Use DATE of CONVERSATION" suffix in their text.)
- **No untrusted-data wrapping.** memrank's own reader delimits the question and context and tells
  the model not to follow instructions inside them. The official readers do not, and the dataset
  text is trusted benchmark data rather than user input, so fidelity wins.

Why this matters: until 2026-09-29 every evaluation shared memrank's own reader, which told the
model *"If they are insufficient, reply exactly: I don't know."* No benchmark's reader says that.
On BEAM-100k it produced "I don't know"-style answers for 46%, 82% and 88% of 400 questions across
three memory systems, of which only 40 are abstention questions -- a measurement of the
instruction, not of the memory. memrank's own reader, instruction included, remains only where no
official reader exists.

Every judged run records the reader prompt it answered with -- its name and a fingerprint of its
bytes, its source, and whether it is official -- and results answered under different reader
prompts are never ranked together.

## Baseline arms

Three controls belong beside every system under test. A row without them does not test the hypothesis --
*external memory makes the decisive difference when the knowledge base exceeds the model's context
window, and is worth its tokens against a naive baseline that just reads what it can.*

| Target | Retrieves | Budget | Question it answers |
|---|---|---|---|
| `no-context` | nothing | n/a | Does the reader already know the answer? |
| `fixed-context` | the corpus, unranked, ingest order | **matched** | Does ranked retrieval beat reading the first N tokens? |
| `full-context` | the corpus, uncapped | uncapped | Is memory needed at all when everything fits? |

`full-context` is not the only uncapped target. Vendor-configuration targets are uncapped too, by
definition -- they reproduce a vendor's own depth rather than this cap -- and they never enter the
leaderboard.
Today those are `mem0` and `hindsight` -- every BARE vendor ref, because since 2026-08-19 a bare
ref means the configuration that vendor ships. memrank's comparison arm carries a suffix:
`hindsight:matched` is the ranked hindsight row, and mem0 has no ranked row at all, since both of
its matched variants moved to the research lane. A target's `context_budget`, not its name, is what
says which it is.

`fixed-context` is the arm that belongs on every row, and it is the *token-matched* one by
design: same context size as the system under test, so the only variable is selection.
`full-context` is deliberately rare -- an evaluation whose knowledge base fits inside a current
context window cannot hold headline status, which is why LoCoMo (16k-26k tokens) is runnable but
labelled non-discriminative.

Two limitations affect interpretation. **Ingest order is not neutral**: `fixed-context` reads from the start, so a
corpus whose answers cluster late is disadvantaged in a way ranked retrieval is not. And
**`no-context` needs answer scoring to assess answer quality.** With retrieval-only scoring,
it can receive credit for negative queries requiring a forbidden span to remain absent.
The runner emits a notice when that happens.

`word-overlap` uses naive token-overlap retrieval, a simple retrieval baseline, distinct from the
empty-context and unranked-context controls.

`tfidf` and `bm25` are weighted keyword retrieval baselines. `tfidf` weights each shared
word by how rare it is and scores the cosine between query and document; `bm25` also saturates
repeated words and normalises by document length, at `k1 = 1.5` and `b = 0.75`. The two share a
tokeniser, so the gap between their rows is the weighting and nothing else. None of the three is
a control: each is a `Memory` that retrieves by the query, so a higher score shows an improvement over that keyword method on the selected evaluation.
Use diagnostic controls to assess the contribution of retrieved context. All three run in your own process, so their latency is not comparable
with a system reached over HTTP.

### What each arm is called in the literature

Use this mapping when comparing control names across publications. The arms were `none`, `baseline` and
`icl` until 2026-08-19; two were renamed again on 2026-08-20.

**Two of these names are ours, not the field's**, and the table says which. Where our name differs
from the published one, the citation identifies the CONDITION -- it is not the source of the name.

| Ours | Published counterpart |
|---|---|
| `no-context` | **Our name, chosen for clarity.** The condition is well established: open-domain QA calls it "closed-book" ([Roberts et al., EMNLP 2020](https://aclanthology.org/2020.emnlp-main.437/)), used verbatim as an arm name by [LOFT](https://aclanthology.org/2025.findings-naacl.374.pdf); agent-memory work calls it "no-memory" ([MemDelta](https://arxiv.org/html/2606.29914)'s lower bound is `S0: No Memory`). No paper uses "no-context" as a defined condition name. We prefer it because "closed-book" is an exam metaphor that is opaque to a reader who does not already know it. |
| `word-overlap` | Attested as an arm name: LoCoMo-style baseline sets list "BM25 retrieval" and "word-overlap retrieval" as **separate** arms, and ours is the second. **Not BM25** -- the score is an unweighted set intersection, no IDF, no term frequency, no length normalisation. [BEIR](https://arxiv.org/abs/2104.08663)'s term for the family above this one is "lexical", which is the level, not the instance. |
| `fixed-context` | Closest to LoCoMo's "Base" and MemGPT's "fixed-context baselines". The budget-matching practice itself is argued for in [Laitenberger, Manning & Liu, EMNLP 2025](https://arxiv.org/abs/2506.03989). **Caveat:** MemGPT's term means bounded by the *model's context window*; ours is bounded by the *retrieval token budget*. |
| `full-context` | Standard as-is in agent-memory work -- [Mem0](https://arxiv.org/abs/2504.19413) and [Zep](https://arxiv.org/html/2501.13956v1) both use the phrase. Core NLP calls the same condition "long-context (LC)". |

It was `icl` until 2026-08-19, which was simply wrong: in-context learning means few-shot
*demonstrations* that induce a task ([Brown et al. 2020](https://arxiv.org/pdf/2005.14165);
[Dong et al., ACL 2024](https://arxiv.org/abs/2301.00234)), not injected knowledge. No memory
benchmark surveyed runs a few-shot control at all, so nothing was lost by dropping the name.

`word-overlap` was briefly `lexical` (2026-08-19 to 2026-08-20). Two problems: it named BEIR's
family rather than the instance, and this repo already uses "lexical" for its ordinary meaning --
real engines describe their own retrieval as "lexical and graph" walks. One word cannot be both an arm
name and a property several engines have.

### One arm we do not have, and one we measure but do not ship

Recorded so the status of each reads as a decision rather than an oversight.

**Oracle / gold-evidence retrieval** -- the ceiling that separates "retrieval is hard" from "reading
is hard". [LongMemEval](https://arxiv.org/abs/2410.10813) ships it as a dataset variant and builds
its headline claim on it: models decline 30-60% reading full history versus oracle evidence. We do
not have it because it needs per-evaluation gold-evidence labels, which not every evaluation here
carries.

**Random retrieval** -- [MemDelta](https://arxiv.org/html/2606.29914)'s control, "random chunks...
controls for 'having text' vs. 'relevant text'". **Measured since 2026-09-10**, as a seeded
arm over one retrieval grid on `locomo`: k documents drawn uniformly at random per query from
the same unit's corpus, ignoring the query, scored by the same evidence-recall diagnostic as
the systems beside it. The control is MemDelta's; ours is a reproduction of it.

What it found there is worth stating because it is the reason the control exists: at that
grid's shipped settings a majority of a retrieval score was reachable by drawing at random,
and a retriever that ignores the query did not clear the floor at any setting. A level quoted without its random floor
overstates what retrieval contributed.

It is **not a shipped arm**: `memrank.catalog()` does not list it, no evaluation run adds
it, and it is not a registered adapter. It is a measurement arm, and what remains undone is
promoting it to one -- which needs a decision about how a floor is reported beside a published
score, not just an adapter.

## Reproducibility receipt

Every tracked run -- one the command line submits -- writes a `Receipt` (see
`memrank/provenance/receipt.py`). `evaluation.run()` in Python returns a
[result](reference/result.md), which carries none; it records the system, the evaluation, the
traces and the values, and `result.save(path)` keeps them. A receipt captures:

- Memrank version + git SHA
- System name and version (recorded as `adapter_name` and `adapter_version`) + engine version
- Evaluation name (recorded as `benchmark_name`) + dataset version (HuggingFace id or file SHA)
- Seed
- Wall-clock timestamps + duration
- Host fingerprint (hostname, platform, Python version)
- Env-var fingerprints (sha256 of values, never plaintext)
- Evidence class and publication eligibility, derived from how the engine actually executed

### Development observations versus reproducible evidence

`memrank submit TARGET EVAL` starts a source-bound named target from the checkout declared in
its central target file. It is the engine-development loop: each run gets a fresh native process,
but it does not run an immutable executable artifact. Its receipt records the source commit,
dirty flag, an opaque hash of tracked and untracked changes, native platform and launcher
identity as `development_observation` with `publishable: false`. The machine-local checkout path
is excluded from the receipt and the synchronized record.

Clean development observations retain their scores and may synchronize into a private
organization for comparison; dirty-tree observations stay local; public leaderboard ingestion
excludes both. A clean git tree does not promote one to reproducible evidence: a commit
identifies source, not the compiled executable or its toolchain.

Likewise, `--on none` against an already-running HTTP engine is an `endpoint_observation` unless
the engine exposes a resolved executable identity. It is useful for rapid diagnostics, and its
long-lived state and unpinned process make it ineligible for publication.

Artifact-backed local and cloud runs remain the evaluator path. Their image digest identifies the
executable artifact; publication still requires the dataset, configuration, models, platform and
methodology pins described elsewhere in this document.

The vendor-neutral charter requires every published value to ship with its receipt. Where a third
party cannot reproduce a memrank-published value within the receipt's noise band, AtomicStrata
investigates within 7 days.

## Determinism

Systems MUST be deterministic given the same seed, or document their non-determinism explicitly.
The runner re-seeds `random` per cell. Stochastic systems -- LLM extraction at a non-zero
temperature -- should either pin temperature to 0 or run multiple seeds and report the spread.
