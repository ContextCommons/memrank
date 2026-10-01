# Start here

> **Deprecated:** this page describes memrank's older, pre-agent surface, which still ships but will be removed. To evaluate an agent, use `memrank run`; see the [README](../README.md).

Evaluate your agent on your tasks.

Use Memrank to measure how well your agent performs on the tasks it has to handle.
Start with a small local evaluation to check the installation and learn to read a result.

A **system** is the implementation you test, such as a memory client. An **evaluation** supplies
the tasks, the rules for measuring them and when to clear state. You choose both; Memrank runs
the tasks, records responses and errors, and reports scores and timings. Running the same
evaluation on another system gives you results to compare.

## Run the installation check

Follow [Installing Memrank](install.md) to add the package to a project and run the complete
TFIDF/Demo example. Installation needs the network. Once installed, that example uses bundled
data and needs no network, running service or API key.

The example stores two short conversations, asks five questions and measures whether retrieval
returns the passage holding each answer. This is a local installation check and a way to learn the result
format. It is too small to establish performance on your own workload.

## Read the installation check

| Output | What to check |
|---|---|
| `system:` | Names `TFIDF`, the local retrieval implementation you selected. |
| `evaluation:` | Names `demo`, its dataset version and 5 tasks. |
| `demo-score` | Evidence recall: higher means more questions had the session holding their answer retrieved. It is not answer correctness. |
| `latency.retrieve.p50` and `.p95` | Median and 95th-percentile retrieval times in milliseconds at Memrank's call boundary. Read the sample counts in `why`. |
| `failure-rate` | The share of traces with an error; lower means fewer failures. |
| `traces:` | Expect `5 recorded, 0 with errors`. A printed score alone does not establish a successful run. |

The score does not penalize irrelevant passages returned alongside the relevant one. There is no
answer-writing model in this example, so it cannot measure answer quality. Timing reflects this
run on your machine, and a small timing difference from another run does not establish a speed
advantage. [Demo's page](evaluations/demo.md) explains the exact scenario.

If the run is refused or records errors, read the reason before interpreting any values.
[Understand results](results.md) explains the difference between a failed task, a refused run and
a measured zero. Include the package version and error text when [asking for help](contributing.md#get-help).

## Choose the next task

- **Compare two versions of your memory system:** follow [Compare memory systems and their versions](comparing.md),
  then run the existing [version comparison example](../examples/06-new-version-vs-old/README.md).
  The example uses a toy memory to teach the mechanics.
- **Try a different memory system:** check the [systems catalog](systems/README.md) for requirements,
  or [connect your implementation](systems.md). A live engine may require credentials and incur
  service costs; the local installation check does not exercise it.
- **Apply your own evaluation:** express your tasks and success criteria with
  [Adding an evaluation](evaluations.md). You decide whether they represent the problem you care
  about; running them does not validate that choice.
- **Inspect or keep the evidence:** [Understand results](results.md) walks from a value to its
  trace and shows how to save and reload a result.

The [documentation index](README.md) also links exact API reference, command-line workflows and
contributor setup.
