# 07 -- against a known engine

`uv run python examples/07-against-a-known-engine/run.py`

Compare the script's custom memory with the included `WordOverlap` implementation on the same
tasks and measures. The script runs offline and prints means, their difference and changed tasks.

To evaluate a service, replace `WordOverlap` with its client and configure the prerequisites.
`Mem0` needs a server at `MEM0_HTTP_URL` (default `http://localhost:8888`) or its SDK configured
with `mode="sdk"`. `AtomicMemory` needs a service at `ATOMICMEMORY_API_URL`
(default `http://localhost:3070`). See the [systems catalog](../../docs/systems/README.md).
An unreachable service raises an error; Memrank does not substitute another implementation.
