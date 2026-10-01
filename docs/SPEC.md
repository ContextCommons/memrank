---
title: "Memrank -- Public Specification"
status: active
last_reviewed: 2026-09-22
audience: external
public_repo: https://github.com/ContextCommons/memrank
license: Apache 2.0
maintainer: AtomicStrata (vendor-neutral charter)
---

# Memrank -- Public Specification

> **Deprecated:** this page describes memrank's older, pre-agent surface, which still ships but will be removed. To evaluate an agent, use `memrank run`; see the [README](../README.md).

**One line:** Evaluate your agent on your tasks. Memrank runs
evaluations against supplied systems and records measurements with their supporting evidence.
Tracked runs additionally record reproducibility receipts.

This document is the canonical statement of what memrank measures, what it refuses to claim, the
contracts a contribution must satisfy, and the governance the maintainer commits to. It is the
document to cite in a disagreement about method. Where an implementation detail is needed,
[`docs/methodology.md`](methodology.md) is authoritative and this specification defers to it.

---

## 1. Mission

The AI-agent ecosystem chooses memory engines from numbers nobody outside the vendor can check.
Each vendor publishes through its own harness, at its own retrieval budget, under its own judge,
against its own dataset snapshot. Latency, ingestion cost and token efficiency are frequently not
measured at all.

Shared harnesses also need transparent methods and configurations. **Every shared harness we could find is operated by a company that
appears in its own results**, memrank included. What memrank offers instead of a claim of
neutrality is a method that can be checked: the configuration is declared, the protocol is
documented, and the result carries what is needed to re-run it. Published results must include the information needed to reproduce them.

## 2. Scope

### 2.1 What memrank measures

A run records a **trace** per task -- everything observed while that task ran, including on
failure -- and **measures** turn those traces into **values**. Every value carries the name of the
measure that produced it and its **decider**: memrank's own clock and bookkeeping, a fixed rule, a
model that adjudicated, or a system declaration. Each value identifies its source.

Five measures ship (`memrank/instrument/measures.py`). The contract they satisfy, and how to write
one of your own, is [`docs/measures.md`](measures.md).

| Measure | The values it produces | Scope | Reads | Decider |
|---|---|---|---|---|
| `WordMatch` | `word-match` | task | `recalled` | rule |
| `Judge` | `judge` | task | `answered` | model |
| `Latency` | `latency.<step>.p50`, `latency.<step>.p95` | run | `timings_ms` | memrank |
| `FailureRate` | `failure-rate` | run | `error` | memrank |
| `BenchmarkScore` | `<evaluation>-score`, one per group | run | `recalled`, `declared` | rule |

`WordMatch` asks whether the expected spans appear verbatim in something the system recalled: a
retrieval proxy, never answer correctness, and it says so on every value it produces. `Latency`
reports p50 and p95 per step over timings memrank took at its own call boundary, never a figure a
system reported about itself (section 4), each with the sample count it was taken over.
`BenchmarkScore` applies a named in-tree evaluation's own scoring, including the per-category
breakdown where that evaluation has one (section 5.1).

Three figures a memrank row carries are not measures, and are named here because a reader of one
meets them:

- **Tokens** -- what a provider billed the system, per query and per ingest, mean and p95. Only the
  system can be told this, so it is **declared** by the system and recorded as the system's word
  (section 4). A bucket it did not measure is `null`, never `0.0`.
- **Cost** -- mean USD per query, priced from the declared model (`memrank/metrics/cost.py`) by the
  tracked run the command line and the cloud drive.
- **Provenance** -- configuration, dataset version, model, seed, instrument version and evidence
  class, which travel in the receipt rather than in a value (section 6).

**No number memrank publishes travels without its receipt** (section 6).

### 2.2 What is under test

A **system** is an implementation of one of four supported interfaces (section 4). The client that drives a memory engine is a system; so is a model, a retriever, an assistant, or
a custom implementation around a private store. `evaluation.run(system=...)` takes the
instance, so a system does not have to be registered, named, or known to memrank to be measured.

What a *published* row names is never the engine alone. A memory engine's result is dominated by
its configured embedder and LLM, so a row naming only the engine names two different
systems at two different times. A system that is named rather than constructed -- in a config file,
on the wire, in a stored artifact -- therefore has to carry that configuration in its name, and the
word for a named composition of an engine with its embedder and LLM is a **target**: the command
line's word, resolved from a manifest and told apart by `config_hash`. An engine under two
configurations is two targets and two rows. The catalog of targets, and the rest of the command
line's vocabulary, is [`docs/misc/command-line.md`](misc/command-line.md).

Systems ship for the services `atomicmemory`, `mem0`, `hindsight` and `supermemory`, for any
translator of the system contract through `native`, and in-process for `tfidf`, `bm25` and the
`word-overlap` retrieval baseline, beside the control arms `no-context`, `fixed-context` and
`full-context`; `memrank.catalog()` prints them and `memrank.systems` holds them as classes. Not
every engine image is obtainable, and the ones that are not are named as such: see
[engine images](misc/engine-images.md). Evaluations ship for the synthetic `demo`,
`relation_graph`, `locomo`, `longmemeval` and `beam`.

### 2.3 Control arms are part of the measurement

A comparison must include these three diagnostic controls:

- `NoContext` (`no-context`) retrieves nothing -- does the reader already know the answer?
- `FixedContext` (`fixed-context`) reads the corpus unranked, token-matched to the system under
  test -- does ranked retrieval beat reading the first N tokens?
- `FullContext` (`full-context`) reads everything uncapped -- is memory needed at all when the
  corpus fits?

Their known limits are recorded rather than hidden: `fixed-context` reads from the start, so a
corpus whose answers cluster late disadvantages it, and `no-context` needs an answer writer and answer-scoring measure to test answer quality.

### 2.4 Out of scope

- Systems with no isolation primitive. Measurement requires that nothing ingested for one group of
  tasks is visible to another; a system that cannot guarantee that cannot be measured here.
- Closed engines with no documented API, because a result nobody else can reproduce is not one
  this instrument will publish.
- Latency of hosted-only services where network round-trip dominates the measurement. memrank
  measures local-deploy and self-host paths.

## 3. What a score means

This section is normative. A consumer of memrank output that ignores it will misreport.

**`composite` is not answer correctness.** It is whatever the evaluation declares in
`quality_metric`. `demo` computes a deterministic substring-recall
proxy; `relation_graph` computes a structural graph score. LoCoMo, LongMemEval and BEAM declare
judged metrics and compute no proxy at all: their gold answers are derived or written as prose
rubrics, so substring recall would be near zero for reasons that have nothing to do with the
system, and without a judge they report latency and failures only.

**Structurally invalid metrics are withheld, not caveated.** Each evaluation declares
`substring_recall_supported` and `composite_rankable`. Where the composite is not rankable without
a judge, the rankable surfaces refuse to show one and report `withheld (judge required)`. The
flags are independent: `substring_recall_supported` describes the answer-substring proxy,
while `composite_rankable` describes the evaluation's own composite. Graph and passage retrieval
scores do not require that proxy. Contract for consumers: **rank a raw `composite` only when
`composite_rankable` is true, and interpret it through `quality_metric`; otherwise use a valid
judged run.**

**The context budget is declared and symmetric.** The default holds every arm to the same
retrieval token budget, because without it "retrieved better" and "returned more text" are the
same number. An evaluation whose own protocol hands the reader everything retrieval returned may
declare `context_policy = "uncapped"`, and two do -- BEAM and LongMemEval -- because a capped run of
those is not those evaluations. What is enforced is that the policy applies to every arm alike and
travels in the receipt, so a capped and an uncapped run cannot be mistaken for each other later.

**Absent is not zero.** A system reporting no token usage records `null`. `0.0` is the claim that
it spent nothing, and conflating the two fabricates an efficiency win for every system that stays
quiet.

**A slice is not a measurement.** `beam:100k-smoke`, `locomo:mini` and their kin take the *first*
N units, which is not a fair sample: measured on a full judged BEAM 100k tier, conversation 1
scores 0.318 against 0.158 for the tier -- a smoke number flatters every system by roughly 2x.
Slices exist to debug ingest, retrieval, judging and dispatch cheaply. A measurement claim comes
from a full tier.

**A methodology change is a documentation change.** A change to scoring, dataset version, or
latency protocol that does not carry a matching change to `docs/methodology.md` is not a change
this project accepts (section 7.3).

## 4. The system contract

A system's **kind** is determined by its base class. A run refuses,
**before touching the system**, when it is not a kind memrank knows or lacks a verb its kind
requires. The kinds are `memrank/instrument/system.py` and `memrank/instrument/kinds.py`; the
shortest statement of them is [`docs/reference/system.md`](reference/system.md), and the full
contract, including the metadata and engine-description surfaces, is
[`docs/system-contract.md`](system-contract.md).

| Kind | What it is for | The verbs it must implement |
|---|---|---|
| `memrank.Memory` | stores supplied documents and retrieves relevant ones | `prepare`, `ingest`, `retrieve`, `cleanup` |
| `memrank.Model` | given a prompt, returns text | `complete` |
| `memrank.Retriever` | given a query, orders its own corpus | `rank` |
| `memrank.Assistant` | responds to supplied messages | `respond` |

`Memory` is the kind a memory system implements, and these four verbs are the whole of it:

<!-- runnable: no -- an interface sketch: the ABC, its imports and its `...` bodies are the contract this section states, not a script -->
```python
from datetime import datetime

import memrank
from memrank import Document, Recall


class Notebook(memrank.Memory):
    """A memory in a Python list. Four verbs, and none of them reports a measurement."""

    name = "notebook"           # short and stable -- stored results use it
    version = "1.0.0"           # this implementation's version
    engine_version = "1.0.0"    # the version of whatever it wraps

    def prepare(self, isolation_unit: str) -> None:
        self.kept: list[Document] = []

    def ingest(self, documents: list[Document]) -> None:
        self.kept.extend(documents)

    def retrieve(self, query: str, k: int, user_id: str,
                 query_timestamp: datetime | str | None = None) -> Recall:
        wanted = set(query.lower().split())

        def overlap(document: Document) -> int:
            return len(wanted & set(document.content.lower().split()))

        return Recall(documents=sorted(self.kept, key=overlap, reverse=True)[:k])

    def cleanup(self) -> None:
        self.kept = []


result = memrank.evaluation("demo").run(system=Notebook())
print(result.values_of("failure-rate")[0].value, len(result.traces))
```

`Recall` (`memrank/contract.py`) contains `documents`, the ranked list, and `declared`, the
provider payload preserved as returned.

A system **must**:

- Guarantee isolation. Nothing ingested under one `isolation_unit` may be visible under another.
- Return documents **ranked best-first** from `retrieve`, and return fewer than `k` rather than
  padding.
- **Fail loudly.** Raise on error rather than returning an empty result -- an empty list is a valid
  answer meaning "nothing matched", so a disguised failure scores exactly like the no-memory
  control arm and reads as a real one. No silent fallback or degraded mode.
- Pass `tests/live/conformance/test_adapter_contract.py`.

Nothing a kind requires reports a measurement memrank takes itself. Latency is timed at memrank's
own call boundary, so a system does not report that measurement. Optional methods **declare** information
only the system can provide:

- `token_metrics()` -- what a provider billed it, as the four keys `tokens_per_query_mean`,
  `tokens_per_query_p95`, `tokens_per_ingest_mean` and `tokens_per_ingest_p95`, rendered by
  memrank's `TokenCollector` rather than a private metric shape. A bucket it did not measure is
  `None`, never `0.0`.
- `declared_latency()` -- time only it can see, inside the hop memrank timed around it, as samples
  per bucket rather than percentiles, so runs at any width report the same statistic of the same
  population.

A system **may** be written in any language and live outside this repository. A **translator**
speaks the same contract over HTTP; memrank launches it, drives it, and never imports it.
[`examples/more/native-adapter/`](../examples/more/native-adapter/README.md) is a working one in about
200 lines of standard-library Python.

> **The operator layer.** `MemoryAdapter` and `MemoryEngine` are the same class object as
> `memrank.Memory` under its previous names, so `isinstance` checks and every registered adapter
> keep holding. The command line calls a registered system an *adapter*, and the class that
> resolves a target's entry is what it means by the word:
> [`docs/misc/command-line.md`](misc/command-line.md).

## 5. The evaluation contract

An **evaluation** is a named, versioned bundle: its **tasks**, the **measures** it ships with, and
the rule for when the system's state is cleared (`memrank/instrument/evaluation.py`). A **task** is
one evaluation case: context, prompt and expected outcome. `memrank.evaluation("demo")` builds one of the shipped evaluations by name, and this is
the same object written by hand:

<!-- runnable: no -- an interface sketch: the ABC, its imports and its `...` bodies are the contract this section states, not a script -->
```python
import memrank
from memrank import Clearing, Document, Evaluation, Expected, Task
from memrank.systems import WordOverlap

note = Document(id="d1", content="Ana has been a marine biologist since 2019.", user_id="ana")
one_question = Evaluation(
    name="one-question",
    version="1.0.0",
    tasks=(Task(id="q_job", prompt="What does Ana do for a living?", group="ana",
                context=(note,),
                expected=Expected(answers=("marine biologist",),
                                  required_spans=("marine biologist",))),),
    measures=(memrank.WordMatch(), memrank.Latency(), memrank.FailureRate()),
    clearing=Clearing.PER_GROUP,
)

result = one_question.run(system=WordOverlap())
for value in result.values_of("word-match"):
    print(value.measure, value.value, value.decider.value)
```

An evaluation **must**:

- Give every task a stable `id`, and a `group` where state has to be isolated between sets of them.
  The `clearing` rule says when the system's state is cleared, and it is closed -- `per-task`,
  `per-group` or `at-end` -- so two runs can be compared on it.
- Declare its measures, which is what lets a run be refused before the system is touched: a measure
  reading a name nothing in the run produces is a refusal with a stated reason and no traces, not a
  failure halfway through a run that has already spent time and money.
- Bump its `version` on any breaking change to how it is loaded or scored, so
  version-against-version comparison can refuse instead of silently comparing two definitions.

### 5.1 Named in-tree evaluations

An evaluation memrank ships by name is built from a `Benchmark` (`memrank/core.py`), which loads
the dataset and brings that dataset's own scoring with it as a `BenchmarkScore` measure.
`Benchmark` is the operator layer: `memrank.evaluation(ref)` is the route to a named evaluation,
and [`docs/evaluations.md`](evaluations.md) is where its own contract is written. A
named in-tree evaluation **must** additionally:

- Run the dataset's own scoring protocol, and document every divergence from it with a rationale
  and an impact estimate.
- Declare its own judgeability. A category the judge gate has never heard of must raise, never
  silently shrink the denominator -- that failure ran undetected in this harness for months and
  cost 60.8% of one evaluation's questions (section 7.4).
- Declare its metric honestly through the flags section 3 names -- `substring_recall_supported`,
  `composite_rankable`, `quality_metric`, `context_policy`, `is_synthetic`, `requires_graph` --
  rather than letting a caller infer them.

### 5.2 Judging

`Judge` is a measure whose decider is a model (section 2.1). It requires `ANTHROPIC_API_KEY` and
nothing else: `anthropic` is a required dependency of the package, not an optional group, because
the evaluations whose only quality metric is a judge's are judged by default in the command line. Constructing `Judge` needs no key;
applying it requires the key and raises if it is missing.

The Python API does not bundle `Judge` into evaluations; callers add it explicitly. On the command line, an eval whose only quality metric is a judge's is
judged by default, and `--no-judge` leaves it measuring latency and cost with no quality score.

Two controls this specification published are retired, and are named here because they were
published:

- **A call cap.** `--max-judge-calls` refused a run whose ceiling was too small; it is gone
  (`memrank/runner.py`, `memrank/cli/retired.py`, `memrank/evaluation/judge_stage.py`). A cap on the
  judge bounds the measurement rather than the system under test -- which questions got graded would
  depend on where a counter ran out, which is to say on ordering -- and no evaluation framework caps
  its scorer. What a tracked run does instead is print the arithmetic before it spends anything:
  enough to decide whether to start. What bounds a judged run is the number of tasks in it.
- **An egress gate.** `--ack-egress` refused a judged run of a non-synthetic evaluation; it is gone
  (`memrank/orchestration/resolve.py`, `memrank/cli/evals.py`). An evaluation still declares
  `is_synthetic` and the declaration is still true, but it gates nothing: what it drives is the
  **disclosure** printed when judging is enabled -- what is sent, to which provider, and where the
  responses are cached. Asking for a judged run is the acknowledgement, which is what the browser
  path had always treated it as.

## 6. Reproducibility

Every result says what produced it. `evaluation.run(system=...)` returns a `Result`
(`memrank/instrument/result.py`) carrying the system -- name, kind, version and address -- the
evaluation and its version, the clearing rule and whether clearing was observed, the start and
finish times, one trace per task per attempt including on failure, and every value with its
decider. `result.save(path)` writes it and `Result.load(path)` reads it back, which is what lets a
measure thought of later run over traces recorded months ago, with the system never touched again.

A **tracked** run -- the one the command line and the cloud drive, and the only one whose output is
publishable -- emits a reproducibility receipt as well (`memrank/provenance/receipt.py`) recording:

- memrank version and commit, and the system, engine and dataset versions
- the resolved configuration and its `config_hash`, plus `config_hash_version`
- the answer and judge models, by provider and name
- the seed, the start and finish times, the duration
- the host and environment fingerprint -- **hashed, never plaintext**, and never a secret value
- the evidence class (section 6.1)

If any of these differ between two runs, they are two rows, not one number measured twice.

### 6.1 Evidence classes

Tracked runs classify their evidence by how the executable was identified:

| Class | What produced it | Publishable |
|---|---|---|
| `reproducible_evidence` | An immutable, resolvable executable artifact | yes |
| `development_observation` | A mutable local checkout | no |
| `endpoint_observation` | An already-running HTTP engine with no resolved executable identity | no |

**A clean git tree does not promote a development observation.** A commit identifies source, not
the executable that ran or the toolchain that built it. Development observations keep their scores
and are useful for comparison; they are ineligible for publication, and a run from a dirty tree
does not leave the machine.

## 7. Governance -- the vendor-neutral charter

memrank is maintained by **AtomicStrata**, which also ships a memory engine. That is a structural
conflict of interest and the charter exists to make it checkable rather than to deny it.

### 7.1 Open contribution

Anyone may submit a system or an evaluation by pull request. There is no approval gate beyond
review for correctness and conformance to section 4 and section 5. An engine can also be measured with no pull
request at all, through an out-of-tree translator.

### 7.2 Verbatim publication

Results are published as measured. AtomicStrata does not curate, omit, or asterisk results
unfavourable to its own engine. Where a result is anomalous, the methodology and the raw output
are published beside it.

### 7.3 Methodology changes are public

A change to scoring, dataset version, or the latency-measurement protocol requires a public
proposal with a comment period before it merges, and a matching change to the documentation.
Affected results are re-run and re-published; superseded numbers stay accessible with the
methodology version they were produced under.

### 7.4 The conflict is checked by reproduction, not asserted away

The mitigation is that every measurement is reproducible by anyone with a clone and the relevant
API keys. Two consequences the maintainer accepts:

- If a third party reports they cannot reproduce a published memrank number, AtomicStrata
  investigates and either fixes the discrepancy or publishes the gap.
- Findings against our own harness and our own engine are stated rather than buried. The record
  so far includes a LoCoMo path that silently scored 39.2% of the benchmark under labels wrong on
  three of four names, a published BEAM headline that turned out to be the high draw of a
  distribution, and a competitor scoring above our own engine when run properly in our own
  harness. All three were found by auditing ourselves against the benchmarks' own protocols.

### 7.5 Long-term governance

Sole maintainership by AtomicStrata is the starting position, not the intended end state. A public
steering committee with seats for third-party memory-engine maintainers is the next step, and
independent stewardship after that if adoption justifies it. Neither is claimed as done.

## 8. Hosting and licensing

- **Repository:** https://github.com/ContextCommons/memrank
- **License:** Apache 2.0 -- chosen for the patent grant and enterprise compatibility.
- **Install:** `uv add memrank` (or `pip install memrank`) into your own project, from PyPI:
  memrank is a library, and [installing memrank](install.md) is the whole procedure. The
  `memrank` command line is not core and is installed as described in
  [the command line](misc/command-line.md); contributors work from a clone, per
  [local development](local-development.md).
- **Results:** run output is written under `results/` on the machine that ran it. There is no
  standing public leaderboard, and this specification does not authorise one; publication is
  selective and evidence-led.
