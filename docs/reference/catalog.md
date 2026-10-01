# catalog

> **Deprecated:** this page describes memrank's older, pre-agent surface, which still ships but will be removed. To evaluate an agent, use `memrank run`; see the [README](../../README.md).

`memrank.catalog()` lists the included systems and evaluations, their import names, string
keys and requirements.

```python
import memrank

memrank.catalog()
```

<!-- output: exact -->
```console
memrank ships 11 system(s) and 5 evaluation(s).

systems -- `from memrank.systems import TFIDF`, or `memrank.system("tfidf")`

  TFIDF         'tfidf'           memory   nothing -- in-process, offline
  BM25          'bm25'            memory   nothing -- in-process, offline
  WordOverlap   'word-overlap'    memory   nothing -- in-process, offline
  NoContext     'no-context'      control  nothing -- the floor: retrieves nothing, answers closed-book
  FixedContext  'fixed-context'   control  nothing -- the corpus unranked, capped by the token budget
  FullContext   'full-context'    control  nothing -- the corpus uncapped: the ceiling retrieval aims at
  AtomicMemory  'atomicmemory'    memory   a running engine, at base_url= or ATOMICMEMORY_API_URL
  Hindsight     'hindsight'       memory   a running engine, at base_url= or HINDSIGHT_API_URL
  Supermemory   'supermemory'     memory   a running engine, at base_url= or SUPERMEMORY_BASE_URL
  Mem0          'mem0'            memory   the mem0 SDK, or a running engine at MEM0_HTTP_URL
  Native        'native'          memory   a running translator of memrank's system contract, at NATIVE_API_URL

evaluations -- `from memrank.evaluations import Demo`, or `memrank.evaluation("demo")`

  Demo()           'demo'            nothing -- a bundled synthetic scenario
                                     measures a substring retrieval proxy and evidence recall
  RelationGraph()  'relation_graph'  nothing -- in-repo fixtures, and a graph-capable system
                                     measures a structural graph score
  LoCoMo()         'locomo'          a one-time download, or LOCOMO_DATA_PATH
                                     measures latency and failures; quality needs a judge
  LongMemEval()    'longmemeval'     a one-time download, or LONGMEMEVAL_DATA_PATH
                                     measures latency and failures; quality needs a judge
  BEAM()           'beam'            a one-time download (the benchmarks extra), or BEAM_DATA_PATH
                                     measures latency and failures; quality needs a judge
```

## What it is

Each row shows the **Python class**, **string key** and **requirements**. `TFIDF` and `Demo` need
no service, API key or download after installation.

System rows also show a role: `memory` for retrieval implementations and clients, or `control`
for diagnostic comparisons. This differs from the system's [kind](system.md): every included
control subclasses `Memory`, so its `result.system.kind` is `memory`.

`catalog()` prints a table and returns entries in `.systems` and `.evaluations`.

Import a class, such as `from memrank.systems import TFIDF`, when writing Python code.
Use `memrank.system("tfidf")` to construct an instance when the key comes from configuration.

## Who supplies what

Memrank supplies eleven [systems](system.md) and five [evaluations](evaluation.md). Listing them
needs no service or credentials; running a selected implementation may require both.

## The Python names

```python
import memrank
from memrank import evaluations, systems

print(len(systems.SHIPPED), "systems,", len(evaluations.SHIPPED), "evaluations")
print(memrank.system("tfidf").name)
print(memrank.evaluation("demo").name)
```

- `memrank.catalog()` -- prints the table and returns it.
- `memrank.systems` -- the module: `TFIDF`, `BM25`, `WordOverlap`, `NoContext`, `FixedContext`,
  `FullContext`, `AtomicMemory`, `Hindsight`, `Supermemory`, `Mem0`, `Native`.
- `memrank.evaluations` -- the module: `Demo`, `RelationGraph`, `LoCoMo`, `LongMemEval`,
  `BEAM`.
- `memrank.system("<name>")` and `memrank.evaluation("<name>")` -- the same things by string.

## Going deeper

- [Supported systems](../systems/README.md) and [available evaluations](../evaluations/README.md)
  -- behavior and prerequisites.
- [Adding a system](../systems.md) -- getting your own into the catalog.
- [Engine images](../misc/engine-images.md) -- per shipped memory engine, whether you can obtain
  the container to run it.
