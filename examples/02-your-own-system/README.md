# 02 -- your own system

`uv run python examples/02-your-own-system/run.py`

Run the Demo evaluation from example 01 against a memory implemented in this script.
Subclass `memrank.Memory` and implement `prepare`, `ingest`, `retrieve` and `cleanup`.
No registration or changes to the Memrank package are required.

The script prints the result. Only the system differs from example 01. Optional declarations
such as `declared_version` return `None` when information is unavailable.
