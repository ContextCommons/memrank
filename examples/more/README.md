<a id="more"></a>
# Command-line and service integration examples

These examples cover tracked runs, target registration and service integration. They use the
command-line pipeline and its result format, or need a live engine or Docker. For direct Python
`evaluation.run()` examples, see the [examples index](../README.md).

| | What it is | Needs |
| --- | --- | --- |
| [`explore-benchmark.py`](explore-benchmark.py) | Inspect a benchmark's tasks without running it or writing output. | nothing |
| [`run-existing-target.py`](run-existing-target.py) | Drive a catalog target through the tracked-run pipeline, resolving the ref exactly as `memrank submit` does. | nothing |
| [`native-adapter/`](native-adapter/README.md) | A translator: the contract over HTTP, in any language, without touching memrank. | nothing |
| [`3-line-example.py`](3-line-example.py) | The smallest run against a real engine: one adapter, one benchmark, smoke slice. | a live engine |
| [`supermemory-image/`](supermemory-image/README.md) | Build the engine image the `supermemory` target names. The image is built locally from the provided recipe. | Docker |
| `compare-am-mem0.py` | Kept as a record of the earlier comparison script. It does not run: it imports `memrank.adapters.baseline`, which no longer exists. | does not run |

`3-line-example.py` reads `ATOMICMEMORY_API_URL` (default `http://localhost:3070`),
`MEM0_HTTP_URL` (`http://localhost:8888`) or `HINDSIGHT_API_URL` (`http://localhost:8888`)
depending on the adapter it is given. An unreachable backend fails with a connection error;
memrank does not silently fall back.
